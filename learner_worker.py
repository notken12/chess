import torch
import torch.nn.functional as F
import torch.optim as optim

import copy

from hyperparams import (
    CONSISTENCY_LOSS_COEFF,
    POLICY_ENTROPY_LOSS_COEFF,
    POLICY_LOSS_COEFF,
    REWARD_LOSS_COEFF,
    TARGET_NET_UPDATE_INTERVAL,
    VALUE_LOSS_COEFF,
)
from model import Networks
from batch_worker import provide_batch_transitions
from replay_buffer import K_STEPS
from tree import value_bins  # For move encoding and categorical support


class Learner:
    def __init__(
        self,
        representation,
        dynamics,
        policy,
        value,
        lr=1e-4,
        target_net_update_interval=TARGET_NET_UPDATE_INTERVAL,
    ):
        # 1. Main Networks (The "Live" Weights)
        self.representation = representation
        self.dynamics = dynamics
        self.policy = policy
        self.value = value
        self.nets = Networks(representation, dynamics, policy, value)
        self.target_nets = copy.deepcopy(self.nets)
        self.target_net_update_interval = target_net_update_interval

        self.device = next(self.representation.parameters()).device

        # 2. Optimization Setup
        self.optimizer = optim.Adam(
            list(self.representation.parameters())
            + list(self.dynamics.parameters())
            + list(self.policy.parameters())
            + list(self.value.parameters()),
            lr=lr,
        )

        # 3. Helper Metadata
        self.value_bins = value_bins  # torch.linspace(-1, 1, num_bins)

    def update_step(self, batch, training_step):
        batch = provide_batch_transitions(training_step, self.nets, self.target_nets)
        if batch is None:
            return
        batch = batch.to(self.device)

        obs_0 = batch["observations"][:, 0]
        latent = self.representation(obs_0)

        total_loss = torch.tensor(0.0, device=self.device)

        move_masks = batch["move_masks"]  # (B,K,4672)
        # Unroll Loop for BPTT
        for k in range(K_STEPS):
            # --- Predictions ---
            pred_value_logits = self.value(latent)
            pred_policy_logits = self.policy(latent, move_masks[:, k])

            target_policy = batch["target_policies"][:, k]
            target_value = batch["target_values"][:, k]

            p_loss = self.compute_policy_loss(pred_policy_logits, target_policy)
            v_loss = self.compute_value_loss(pred_value_logits, target_value)
            total_loss += POLICY_LOSS_COEFF * p_loss + VALUE_LOSS_COEFF * v_loss

            # --- Transitions ---
            if k < K_STEPS - 1:
                action_indices = batch["actions"][:, k]
                action_tensor = self.encode_action(action_indices)

                # Dynamics returns (nextState, rewardProbs)
                latent, pred_reward_logits = self.dynamics(latent, action_tensor)

                # Reward Loss
                target_reward = batch["rewards"][:, k]
                r_loss = self.compute_reward_loss(pred_reward_logits, target_reward)

                # Consistency Loss (Grounding the dream)
                real_obs_k = batch["observations"][:, k + 1]
                with torch.no_grad():
                    real_latent = self.target_nets.representation(real_obs_k)

                c_loss = self.compute_consistency_loss(latent, real_latent)
                total_loss += (
                    REWARD_LOSS_COEFF * r_loss + CONSISTENCY_LOSS_COEFF * c_loss
                )

                policy_entropy_loss = self.compute_entropy_loss(pred_policy_logits)
                total_loss += POLICY_ENTROPY_LOSS_COEFF * policy_entropy_loss

                # Gradient Scaling: Scale by 1/2 at each step of the chain
                latent.register_hook(lambda grad: grad * 0.5)

        # Optimization Step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        if training_step % self.target_net_update_interval == 0:
            self.update_target_networks()

    # --- Loss Helper Functions ---

    def compute_policy_loss(self, pred_logits, target_policy):
        return -(target_policy * F.log_softmax(pred_logits, dim=1)).sum(dim=1).mean()

    def compute_value_loss(self, pred_logits, target_value):
        target_dist = self.scalar_to_categorical(target_value)
        return -(target_dist * F.log_softmax(pred_logits, dim=1)).sum(dim=1).mean()

    def compute_reward_loss(self, pred_logits, target_reward):
        # Maps -1, 0, 1 to class indices 0, 1, 2
        return F.cross_entropy(pred_logits, (target_reward + 1).long())

    def compute_consistency_loss(self, pred_latent, real_latent):
        p = F.normalize(pred_latent.flatten(1), p=2, dim=1)
        r = F.normalize(real_latent.flatten(1), p=2, dim=1)
        return F.mse_loss(p, r)

    def compute_entropy_loss(self, pred_logits):
        probs = F.softmax(pred_logits, dim=1)
        log_probs = F.log_softmax(pred_logits, dim=1)
        return (probs * log_probs).sum(dim=1).mean()

    def update_target_networks(self):
        self.target_nets = copy.deepcopy(self.nets)

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

    def scalar_to_categorical(self, v):
        # Maps float value to the 51-bin histogram
        v = v.clamp(-1, 1)
        bin_width = self.value_bins[1] - self.value_bins[0]
        centered_v = (v - self.value_bins[0]) / bin_width
        low = centered_v.floor().long()
        high = (low + 1).clamp(0, len(self.value_bins) - 1)
        w_high = centered_v - low
        dist = torch.zeros((v.size(0), len(self.value_bins)), device=v.device)
        dist.scatter_(
            1,
            low.unsqueeze(1).clamp(0, len(self.value_bins) - 1),
            (1 - w_high).unsqueeze(1),
        )
        dist.scatter_(1, high.unsqueeze(1), w_high.unsqueeze(1))
        return dist
