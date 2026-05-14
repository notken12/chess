import torch
import torch.nn.functional as F
import torch.optim as optim

import copy

from hyperparams import (
    CONSISTENCY_LOSS_COEFF,
    LR,
    POLICY_ENTROPY_LOSS_COEFF,
    POLICY_LOSS_COEFF,
    REWARD_LOSS_COEFF,
    TARGET_NET_UPDATE_INTERVAL,
    UNROLL_STEPS,
    VALUE_LOSS_COEFF,
)
from model import Networks, ProjectionHead, PredictionHead
from batch_worker import provide_batch_transitions
from tree import value_bins, reward_bins  # For move encoding and categorical support


class Learner:
    def __init__(
        self,
        representation,
        dynamics,
        policy,
        value,
        lr=LR,
        target_net_update_interval=TARGET_NET_UPDATE_INTERVAL,
    ):
        # 1. Main Networks (The "Live" Weights)
        self.representation = representation
        self.dynamics = dynamics
        self.policy = policy
        self.value = value
        self.device = next(self.representation.parameters()).device
        self.projector = ProjectionHead(in_channels=256).to(self.device)
        self.predictor = PredictionHead().to(self.device)
        self.nets = Networks(
            representation, dynamics, policy, value, self.projector, self.predictor
        )
        self.target_nets = copy.deepcopy(self.nets)
        self._freeze_target_nets()
        self.target_net_update_interval = target_net_update_interval

        # 2. Optimization Setup (SGD for discrete/vision tasks per EZ-V2 Table 3)
        params = (
            list(self.representation.parameters())
            + list(self.dynamics.parameters())
            + list(self.policy.parameters())
            + list(self.value.parameters())
            + list(self.projector.parameters())
            + list(self.predictor.parameters())
        )
        self.optimizer = optim.SGD(
            params, lr=lr, momentum=0.9, weight_decay=1e-4
        )

        # 3. Helper Metadata
        self.value_bins = value_bins  # torch.linspace(-1, 1, num_bins)
        self.reward_bins = reward_bins

    def update_step(self, batch, training_step):
        batch = batch.to(self.device)

        obs_0 = batch["observations"][:, 0]
        latent = self.representation(obs_0)

        total_loss = torch.tensor(0.0, device=self.device)
        loss_accum = {
            "total": 0.0,
            "policy": 0.0,
            "value": 0.0,
            "reward": 0.0,
            "consistency": 0.0,
            "entropy": 0.0,
        }

        move_masks = batch["move_masks"]  # (B,K,4672)

        window_lengths = batch["window_lengths"]  # (B,)
        step_idx = torch.arange(UNROLL_STEPS, device=self.device).unsqueeze(0)  # (1,K)
        valid = step_idx < window_lengths.unsqueeze(1)  # (B,K)

        # Unroll Loop for BPTT
        for k in range(UNROLL_STEPS):
            # --- Predictions ---
            pred_value_logits = self.value(latent)
            pred_policy_logits = self.policy(latent, move_masks[:, k])

            target_policy = batch["target_policies"][:, k]
            target_value = batch["target_values"][:, k]

            p_loss = self.compute_policy_loss(pred_policy_logits, target_policy)
            v_loss = self.compute_value_loss(pred_value_logits, target_value)
            mask_k = valid[:, k]
            num_valid_k = mask_k.sum()
            if num_valid_k == 0:
                continue
            p_mean = (p_loss * mask_k).sum() / num_valid_k
            v_mean = (v_loss * mask_k).sum() / num_valid_k
            total_loss += POLICY_LOSS_COEFF * p_mean
            total_loss += VALUE_LOSS_COEFF * v_mean
            loss_accum["policy"] += p_mean.item()
            loss_accum["value"] += v_mean.item()

            # --- Transitions ---
            if k < UNROLL_STEPS - 1:
                action_tensor = self.encode_action(batch["actions"][:, k])
                next_latent, pred_reward_logits = self.dynamics(latent, action_tensor)
                # Reward loss at step k: valid if step k itself is real
                r_loss = self.compute_reward_loss(
                    pred_reward_logits, batch["rewards"][:, k]
                )
                # Consistency loss: needs step k+1 to also be real
                real_obs_k = batch["observations"][:, k + 1]
                with torch.no_grad():
                    real_latent = self.target_nets.representation(real_obs_k)
                c_loss = self.compute_consistency_loss(next_latent, real_latent)
                # Entropy regularization on policy at step k
                e_loss = self.compute_entropy_loss(pred_policy_logits)
                # Reward and entropy: supervise if step k is real
                r_mean = (r_loss * mask_k).sum() / num_valid_k
                e_mean = (e_loss * mask_k).sum() / num_valid_k
                total_loss += REWARD_LOSS_COEFF * r_mean
                total_loss += POLICY_ENTROPY_LOSS_COEFF * e_mean
                loss_accum["reward"] += r_mean.item()
                loss_accum["entropy"] += e_mean.item()
                # Consistency: only if next state is also real
                cons_mask = valid[:, k] & valid[:, k + 1]
                if cons_mask.any():
                    c_mean = (c_loss * cons_mask).sum() / cons_mask.sum()
                    total_loss += CONSISTENCY_LOSS_COEFF * c_mean
                    loss_accum["consistency"] += c_mean.item()
                # Gradient scaling for BPTT
                latent = next_latent
                latent.register_hook(lambda grad: grad * 0.5)

        # Optimization Step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        if training_step % self.target_net_update_interval == 0:
            self.update_target_networks()

        loss_accum["total"] = float(total_loss.item())
        return loss_accum

    # --- Loss Helper Functions ---
    # these all return all per-sample losses (B,)

    def compute_policy_loss(self, pred_logits, target_policy):
        return -(target_policy * F.log_softmax(pred_logits, dim=1)).sum(dim=1)  # (B,)

    def compute_value_loss(self, pred_logits, target_value):
        target_dist = self.scalar_to_categorical(target_value)
        return -(target_dist * F.log_softmax(pred_logits, dim=1)).sum(dim=1)  # (B,)

    def compute_reward_loss(self, pred_logits, target_reward):
        target_dist = self.scalar_to_categorical(target_reward, bins=self.reward_bins)
        return -(target_dist * F.log_softmax(pred_logits, dim=1)).sum(dim=1)  # (B,)

    def compute_consistency_loss(self, pred_latent, real_latent):
        # SimSiam-style: predictor on pred side, stop-grad on target side.
        z_pred = self.nets.projector(pred_latent)
        p_pred = self.nets.predictor(z_pred)
        with torch.no_grad():
            z_real = self.target_nets.projector(real_latent)
        return -F.cosine_similarity(p_pred, z_real, dim=-1)

    def compute_entropy_loss(self, pred_logits):
        probs = F.softmax(pred_logits, dim=1)
        log_probs = F.log_softmax(pred_logits, dim=1)
        return -(probs * log_probs).sum(dim=1)  # (B,)

    def update_target_networks(self):
        self.target_nets = copy.deepcopy(self.nets)
        self._freeze_target_nets()

    def _freeze_target_nets(self):
        """Set target_nets to eval (BN uses running stats, no NaN with batch=1
        in MCTS expansion) and disable grads (no accidental updates)."""
        self.target_nets.eval()
        for m in (
            self.target_nets.representation,
            self.target_nets.dynamics,
            self.target_nets.policy,
            self.target_nets.value,
            self.target_nets.projector,
            self.target_nets.predictor,
        ):
            for p in m.parameters():
                p.requires_grad = False

    def encode_action(self, action_indices):
        """convert one hot encoding to full tensor"""
        B = action_indices.shape[0]
        action_tensor = torch.zeros((B, 73, 8, 8), device=action_indices.device)
        for b in range(B):
            idx = action_indices[b].item()
            z = idx // 64
            x = (idx % 64) // 8
            y = idx % 8
            action_tensor[b, z, x, y] = 1.0
        return action_tensor

    def scalar_to_categorical(self, v, bins=None):
        # Maps float value to the 51-bin histogram
        if bins is None:
            bins = self.value_bins
        bins = bins.to(v.device)
        num_bins = len(bins)
        v = v.clamp(bins[0], bins[-1])
        bin_width = bins[1] - bins[0]
        centered_v = (v - bins[0]) / bin_width
        low = centered_v.floor().long().clamp(0, num_bins - 1)
        high = (low + 1).clamp(0, num_bins - 1)
        w_high = centered_v - low.float()
        dist = torch.zeros((v.size(0), num_bins), device=v.device)
        # scatter_add_ instead of scatter_: when low == high (v on a bin
        # boundary, e.g. v ≈ +1), both writes go to the same slot. Adding
        # accumulates to (1 - w_high) + w_high = 1; plain scatter would
        # overwrite the first write with w_high ≈ 0.
        dist.scatter_add_(1, low.unsqueeze(1), (1 - w_high).unsqueeze(1))
        dist.scatter_add_(1, high.unsqueeze(1), w_high.unsqueeze(1))
        return dist
