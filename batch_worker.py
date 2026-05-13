from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import torch

from hyperparams import BATCH_SIZE, GAMMA, T1, T2
from model import Networks
from replay_buffer import K_STEPS, L_STEPS, get_num_episodes, sample_batch
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
    observations = batch["observations"]  # (B, K+L, 8, 8, 119)
    # observations shape: (B, K+L, 8, 8, 119)
    obs_chw = observations.permute(0, 1, 4, 2, 3)
    # new shape: (B, K+L, 119, 8, 8)
    rewards = batch["rewards"]  # (B, K+L)
    B = observations.shape[0]

    # --- Reanalysis: MCTS on the K unroll steps only ---
    unroll_obs = obs_chw[:, :K_STEPS]  # (B, K, 119, 8, 8)
    unroll_obs_flat = unroll_obs.flatten(0, 1)
    unroll_masks = batch["move_masks"]
    unroll_masks_flat = unroll_masks.flatten(0, 1)
    with torch.no_grad():
        latents_flat = nets.representation(unroll_obs_flat)  # (B*K, 256, 8, 8)
        logits_flat = nets.policy(latents_flat, unroll_masks_flat)  # (B*K, A)

    # Parallel MCTS for all B*K unroll positions.
    # tree.py uses thread-local RNGs so workers don't corrupt each other's state.
    with ThreadPoolExecutor() as executor:
        results = list(
            executor.map(
                lambda lp: _reanalyze(lp[0], lp[1], target_nets),
                zip(latents_flat, logits_flat),
            )
        )
    target_policies = torch.stack([r.target_policy for r in results]).view(
        B, K_STEPS, -1
    )
    mcts_values = torch.tensor(
        [r.mcts_value for r in results], device=observations.device
    ).view(B, K_STEPS)

    # --- Bootstrap values: V(s_{t+j+L}) for each unroll position j ---
    # The bootstrap for position j sits at window index j+L, spanning L..K+L-1.
    boot_obs = obs_chw[:, L_STEPS : K_STEPS + L_STEPS]  # (B, K, 8, 8, 119)
    boot_obs_flat = boot_obs.flatten(0, 1)  # (B*K, 8, 8, 119)
    with torch.no_grad():
        boot_latents = nets.representation(boot_obs_flat)  # (B*K, 256, 8, 8)
        boot_value_logits = nets.value(boot_latents)  # (B*K, num_bins)
        # Convert bins back to a single scalar value
        boot_value_probs = torch.softmax(boot_value_logits, dim=-1)

    bins = value_bins.to(boot_value_probs.device)
    boot_values_flat = (boot_value_probs * bins).sum(dim=-1)  # (B*K,) expected value
    bootstrap_values = boot_values_flat.view(B, K_STEPS)  # (B, K)

    td_targets = _build_td_targets(rewards, bootstrap_values)  # (B, K)

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
