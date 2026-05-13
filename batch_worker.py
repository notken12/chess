from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import torch

from hyperparams import BATCH_SIZE, GAMMA, T1, T2, TD_STEPS, UNROLL_STEPS
from model import Networks
from replay_buffer import get_num_episodes, sample_batch
from tree import get_target_policy, value_bins


class _ReanalysisResult(NamedTuple):
    target_policy: torch.Tensor
    mcts_value: float


def _reanalyze(
    latent: torch.Tensor, policy_logits: torch.Tensor, target_nets: Networks
) -> _ReanalysisResult:
    """Run MCTS for one latent state (without batch dimension); return the improved policy and root value."""
    _, target_policy, mcts_value = get_target_policy(
        latent.unsqueeze(0), policy_logits, target_nets
    )
    return _ReanalysisResult(target_policy, mcts_value)


def _build_td_targets(
    rewards: torch.Tensor,
    bootstrap_values: torch.Tensor,
) -> torch.Tensor:
    """
    Compute l-step TD targets for every position in the K-step unroll window.

    For each unroll position j in [0, K):
        G[b, j] = Σ_{i=0}^{L-1} γ^i · rewards[b, j+i]  +  γ^L · V_boot[b, j]

    rewards has shape (B, K+L) so that rewards[b, j:j+L] is always in-bounds.
    bootstrap_values[b, j] = V(s_{t+j+L}), obtained by running the representation
    and value functions on the stored observation at position j+L in the window.

    Fully vectorised — no Python loop over K or L.
    """
    K = bootstrap_values.shape[1]
    L = rewards.shape[1] - K
    device = rewards.device

    # Build gather indices: idx[j, i] = j+i for j in [0,K), i in [0,L)
    j_idx = torch.arange(K, device=device).unsqueeze(1)  # (K, 1)
    i_idx = torch.arange(L, device=device).unsqueeze(0)  # (1, L)
    gather_idx = j_idx + i_idx  # (K, L)

    reward_windows = rewards[:, gather_idx]  # (B, K, L)
    discounts = GAMMA ** torch.arange(L, dtype=torch.float32, device=device)  # (L,)
    td_from_rewards = (reward_windows * discounts).sum(dim=-1)  # (B, K)

    td_targets = td_from_rewards + GAMMA**L * bootstrap_values  # (B, K)
    return td_targets


def provide_batch_transitions(
    training_step: int, nets: Networks, target_nets: Networks
):
    num_games = get_num_episodes()
    if num_games < BATCH_SIZE:
        return None
    device = next(nets.representation.parameters()).device
    batch = sample_batch(batch_size=BATCH_SIZE).to(device)
    step_idx = torch.arange(UNROLL_STEPS, device=device).unsqueeze(0)
    valid = step_idx < batch["window_lengths"].unsqueeze(1)

    observations = batch["observations"]  # (B, K+L, 8, 8, 119)
    # observations shape: (B, K+L, 8, 8, 119)
    obs_chw = observations.permute(0, 1, 4, 2, 3)
    # new shape: (B, K+L, 119, 8, 8)
    rewards = batch["rewards"]  # (B, K+L)
    B = observations.shape[0]

    # --- Reanalysis: MCTS on valid unroll steps only ---
    unroll_obs = obs_chw[:, :UNROLL_STEPS]  # (B, K, 119, 8, 8)
    unroll_masks = batch["move_masks"]

    valid_flat = valid.flatten()
    unroll_obs_flat = unroll_obs.flatten(0, 1)[valid_flat]  # (N_valid, 119, 8, 8)
    unroll_masks_flat = unroll_masks.flatten(0, 1)[valid_flat]  # (N_valid, 4672)

    with torch.no_grad():
        latents_flat = target_nets.representation(
            unroll_obs_flat
        )  # (N_valid, 256, 8, 8)
        logits_flat = target_nets.policy(
            latents_flat, unroll_masks_flat
        )  # (N_valid, A)

    # Run MCTS on valid unroll positions
    with ThreadPoolExecutor() as executor:
        results = list(
            executor.map(
                lambda lp: _reanalyze(lp[0], lp[1], target_nets),
                zip(latents_flat, logits_flat),
            )
        )

    # Reconstruct (B, K) tensors, zero/one-hot fill for padded positions
    A = logits_flat.shape[-1]
    target_policies = torch.zeros(B, UNROLL_STEPS, A, device=device)
    mcts_values = torch.zeros(B, UNROLL_STEPS, device=device)
    target_policies[valid] = torch.stack([r.target_policy for r in results])
    mcts_values[valid] = torch.tensor([r.mcts_value for r in results], device=device)

    # --- Bootstrap values: only on valid positions ---
    boot_valid = (step_idx + TD_STEPS) < batch["window_lengths"].unsqueeze(1)  # (B, K)
    boot_obs = obs_chw[:, TD_STEPS : UNROLL_STEPS + TD_STEPS]  # (B, K, 119, 8, 8)
    boot_obs_flat = boot_obs.flatten(0, 1)[boot_valid.flatten()]  # (N_boot, 119, 8, 8)
    with torch.no_grad():
        boot_latents = target_nets.representation(boot_obs_flat)
        boot_value_logits = target_nets.value(boot_latents)
        boot_value_probs = torch.softmax(boot_value_logits, dim=-1)
    boot_values_flat = (boot_value_probs * value_bins.to(device)).sum(dim=-1)
    bootstrap_values = torch.zeros(B, UNROLL_STEPS, device=device)
    bootstrap_values[boot_valid] = boot_values_flat

    # TD targets: padded rewards are already 0, and invalid bootstraps are 0
    td_targets = _build_td_targets(rewards, bootstrap_values)

    # --- Mixed value target (EfficientZero V2) ---
    #   condition 1 — early training: value net not yet reliable, always use TD
    #   condition 2 — recent games: their MCTS ran on a stale network, use TD
    # Both evaluated without any Python-level branching over B*K.
    game_positions = batch["game_buffer_positions"]  # (B,)
    if training_step < T1:
        target_values = td_targets
    else:
        cond_recent = game_positions >= (num_games - T2)  # (B,)
        target_values = torch.where(
            cond_recent.unsqueeze(1), td_targets, mcts_values
        )  # (B, K)

    batch["observations"] = obs_chw
    batch["target_policies"] = target_policies  # (B, K, A)
    batch["target_values"] = target_values  # (B, K)
    return batch
