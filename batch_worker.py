from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import torch
from tensordict import TensorDict

from replay_buffer import K_STEPS, L_STEPS, get_num_episodes, sample_batch
from tree import get_target_policy, value_bins

# EfficientZero V2 mixed value target thresholds.
# Before T1 training steps, the value network is too inaccurate to rely on
# MCTS values, so the TD target is always used regardless of game recency.
T1 = 10_000
# The T2 most recently added games always use the TD target: their MCTS was
# run with an older network snapshot and is therefore less trustworthy.
T2 = 1_000
# Chess terminal rewards arrive only at the end of the game, so no discounting
# within the K-step window is necessary.
GAMMA = 1.0
BATCH_SIZE = 128


class _ReanalysisResult(NamedTuple):
    target_policy: torch.Tensor
    mcts_value: float


def _reanalyze(latent: torch.Tensor, policy_logits: torch.Tensor) -> _ReanalysisResult:
    """Run MCTS for one latent state; return the improved policy and root value."""
    _, target_policy, mcts_value = get_target_policy(latent, policy_logits)
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


def provide_batch_transitions(training_step: int) -> TensorDict | None:
    num_games = get_num_episodes()
    if num_games < BATCH_SIZE:
        return None
    batch = sample_batch(batch_size=BATCH_SIZE)
    observations = batch["observations"]  # (B, K+L, 8, 8, 119)
    rewards = batch["rewards"]  # (B, K+L)
    mask = batch["mask"]  # (B, K+L)
    B = observations.shape[0]

    # --- Reanalysis: MCTS on the K unroll steps only ---
    unroll_obs = observations[:, :K_STEPS]  # (B, K, 8, 8, 119)
    unroll_obs_flat = unroll_obs.flatten(0, 1)  # (B*K, 8, 8, 119)
    with torch.no_grad():
        latents_flat = representation_function(unroll_obs_flat)  # (B*K, 8, 8, 256)
        logits_flat = policy_function(latents_flat)  # (B*K, A)

    # Parallel MCTS for all B*K unroll positions.
    # tree.py uses thread-local RNGs so workers don't corrupt each other's state.
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(_reanalyze, latents_flat, logits_flat))

    target_policies = torch.stack([r.target_policy for r in results]).view(
        B, K_STEPS, -1
    )
    mcts_values = torch.tensor([r.mcts_value for r in results]).view(B, K_STEPS)

    # --- Bootstrap values: V(s_{t+j+L}) for each unroll position j ---
    # The bootstrap for position j sits at window index j+L, spanning L..K+L-1.
    boot_obs = observations[:, L_STEPS : K_STEPS + L_STEPS]  # (B, K, 8, 8, 119)
    boot_obs_flat = boot_obs.flatten(0, 1)  # (B*K, 8, 8, 119)
    with torch.no_grad():
        boot_latents = representation_function(boot_obs_flat)  # (B*K, 8, 8, 256)
        boot_value_probs = value_function(boot_latents)  # (B*K, num_bins)
    bins = value_bins.to(boot_value_probs.device)
    boot_values_flat = (boot_value_probs * bins).sum(dim=-1)  # (B*K,) expected value
    bootstrap_values = boot_values_flat.view(B, K_STEPS)  # (B, K)
    # Zero out positions where the game had already ended.
    bootstrap_values = bootstrap_values * mask[:, L_STEPS : K_STEPS + L_STEPS]

    td_targets = _build_td_targets(rewards, bootstrap_values)  # (B, K)

    # --- Mixed value target (EfficientZero V2) ---
    #   condition 1 — early training: value net not yet reliable, always use TD
    #   condition 2 — recent games: their MCTS ran on a stale network, use TD
    # Both evaluated without any Python-level branching over B*K.
    game_positions = batch["game_buffer_positions"]  # (B,)
    cond_recent = game_positions >= (num_games - T2)  # (B,)
    use_td = torch.as_tensor(training_step < T1) | cond_recent.unsqueeze(1)  # (B, K)

    target_values = torch.where(use_td, td_targets, mcts_values)  # (B, K)

    batch["target_policies"] = target_policies  # (B, K, A)
    batch["target_values"] = target_values  # (B, K)
    return batch
