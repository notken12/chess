from typing import List, Tuple
import torch
from collections import deque
from tensordict import TensorDict
import numpy as np

from hyperparams import REPLAY_CAPACITY, TD_STEPS, UNROLL_STEPS

# (observation (8,8,119), action index, reward, move_mask)
# Target policies are intentionally excluded: EfficientZero V2 uses reanalysis,
# recomputing them from stored observations with the current network at training
# time so the policy head always trains against up-to-date targets.
Step = Tuple[torch.Tensor, int, int, torch.Tensor]

K_STEPS = TD_STEPS  # unroll depth: number of dynamics steps during training
L_STEPS = UNROLL_STEPS  # TD horizon: number of real reward steps before bootstrapping


class EpisodeReplayBuffer:
    """
    Stores full game episodes and samples (K+L)-step windows for MuZero unrolling.

    Each sample contains K unroll steps followed by L extra steps needed to
    compute proper l-step TD targets for every position in the unroll window.
    For position j in [0, K), the TD target uses rewards at j..j+L-1 and
    bootstraps with V(s_{t+j+L}); all of these lie within the K+L window.

    Sampling is weighted by episode length so that every individual step has
    an equal probability of being selected as a window start, regardless of
    game length.

    Sample schema (batch_size B, unroll depth K, TD horizon L):
        observations          (B, K+L, 8, 8, 119) — K unroll steps + L extra steps.
                                                     observations[:, :K] seed the
                                                     dynamics chain and consistency loss.
                                                     observations[:, L:K+L] are encoded
                                                     by the representation + value functions
                                                     to produce per-position bootstraps.
        actions               (B, K)               — actions for the K unroll steps only;
                                                     the extra L steps are not needed since
                                                     bootstraps come from the representation
                                                     function, not the dynamics function.
        rewards               (B, K+L)             — rewards for all K+L steps, needed to
                                                     compute l-step TD targets for each of
                                                     the K unroll positions.
        game_buffer_positions (B,)                 — deque index of each sampled game
                                                     (0 = oldest, N-1 = newest); used
                                                     by the batch worker to identify
                                                     recently-added games that should
                                                     use TD targets instead of MCTS values.
    """

    def __init__(
        self,
        max_episodes: int = REPLAY_CAPACITY,
        k_steps: int = K_STEPS,
        l_steps: int = L_STEPS,
    ) -> None:
        self.max_episodes = max_episodes
        self.k_steps = k_steps
        self.l_steps = l_steps
        self._episodes: deque[TensorDict] = deque()
        self._total_steps = 0

    def add_episode(self, history: List[Step]) -> None:
        if len(history) < self.k_steps + self.l_steps:
            return  # Too short for a full K+L window; skip entirely
        observations, actions, rewards, move_masks = zip(*history)
        episode = TensorDict(
            {
                "observation": torch.stack(observations),  # (T, 8, 8, 119)
                "action": torch.tensor(actions),  # (T,)
                "reward": torch.tensor(rewards, dtype=torch.float32),  # (T,)
                "move_mask": torch.stack(move_masks),
            },
            batch_size=[len(history)],
        )
        if len(self._episodes) == self.max_episodes:
            evicted = self._episodes.popleft()
            self._total_steps -= evicted.batch_size[0]
        self._episodes.append(episode)
        self._total_steps += len(history)

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    def sample(self, batch_size: int) -> TensorDict:
        episodes = list(self._episodes)
        lengths = np.array([ep.batch_size[0] for ep in episodes], dtype=np.float64)
        probs = lengths / lengths.sum()
        # Sample indices (not the TensorDict objects) so we can track buffer positions.
        episode_indices = np.random.choice(
            len(episodes), size=batch_size, p=probs, replace=True
        )

        total_steps = self.k_steps + self.l_steps
        observations, actions_out, rewards_out, masks_out, buffer_positions = (
            [],
            [],
            [],
            [],
            [],
        )

        for ep_idx in episode_indices:
            ep = episodes[ep_idx]
            buffer_positions.append(ep_idx)
            T = ep.batch_size[0]
            # Reject windows that would exceed episode length.
            max_t = max(0, T - total_steps)
            t = np.random.randint(0, max_t + 1) if max_t > 0 else 0

            ep_obs, ep_actions, ep_rewards, ep_masks = [], [], [], []
            for k in range(total_steps):
                ep_obs.append(ep["observation"][t + k])
                ep_rewards.append(ep["reward"][t + k])
                # Actions are only needed for the K unroll steps, not the extra L.
                if k < self.k_steps:
                    ep_actions.append(ep["action"][t + k])
                    ep_masks.append(ep["move_mask"][t + k])

            observations.append(torch.stack(ep_obs))  # (K+L, 8, 8, 119)
            actions_out.append(torch.stack(ep_actions))  # (K,)
            rewards_out.append(torch.stack(ep_rewards))  # (K+L,)
            masks_out.append(torch.stack(ep_masks))  # (K,)

        # target_policies are not included: the training loop recomputes them
        # via reanalysis (MCTS with the current network) on sampled observations.
        return TensorDict(
            {
                "observations": torch.stack(observations),  # (B, K+L, 8, 8, 119)
                "actions": torch.stack(actions_out),  # (B, K)
                "rewards": torch.stack(rewards_out),  # (B, K+L)
                "game_buffer_positions": torch.tensor(buffer_positions),  # (B,)
                "move_masks": torch.stack(masks_out),  # (B, K, 4672)
            },
            batch_size=[batch_size],
        )

    def __len__(self) -> int:
        return self._total_steps


rb = EpisodeReplayBuffer(max_episodes=10_000, k_steps=K_STEPS, l_steps=L_STEPS)


def save_to_replay_buffer(history: List[Step]) -> None:
    rb.add_episode(history)


def get_num_episodes() -> int:
    return rb.num_episodes


def sample_batch(batch_size: int = 128) -> TensorDict:
    return rb.sample(batch_size)
