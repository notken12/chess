import os
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

REPLAY_BUFFER_PATH = "replay_buffer.pt"


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
    ) -> None:
        self.max_episodes = max_episodes
        self._episodes: deque[TensorDict] = deque()
        self._total_steps = 0

    def add_episode(self, history: List[Step]) -> None:
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

        total_steps = UNROLL_STEPS + TD_STEPS
        (
            observations,
            actions_out,
            rewards_out,
            masks_out,
            buffer_positions,
            window_lengths,
        ) = ([], [], [], [], [], [])

        for ep_idx in episode_indices:
            ep = episodes[ep_idx]
            buffer_positions.append(ep_idx)
            T = ep.batch_size[0]
            t = np.random.randint(0, T)
            window_lengths.append(min(T - t, total_steps))

            ep_obs, ep_actions, ep_rewards, ep_masks = [], [], [], []
            for k in range(total_steps):
                if t + k >= T:
                    ep_obs.append(torch.zeros_like(ep_obs[0]))
                    ep_rewards.append(torch.tensor(0.0, dtype=torch.float32))
                    if k < UNROLL_STEPS:
                        ep_actions.append(torch.tensor(0, dtype=torch.long))
                        ep_masks.append(torch.full_like(ep_masks[0], -1e9))
                    continue

                ep_obs.append(ep["observation"][t + k])
                ep_rewards.append(ep["reward"][t + k])
                # Actions are only needed for the K unroll steps, not the extra L.
                if k < UNROLL_STEPS:
                    ep_actions.append(ep["action"][t + k])
                    ep_masks.append(ep["move_mask"][t + k])

            observations.append(torch.stack(ep_obs))  # (K+L, 8, 8, 119)
            rewards_out.append(torch.stack(ep_rewards))  # (K+L,)
            actions_out.append(torch.stack(ep_actions))  # (K,)
            masks_out.append(torch.stack(ep_masks))  # (K,)

        # target_policies are not included: the training loop recomputes them
        # via reanalysis (MCTS with the current network) on sampled observations.
        return TensorDict(
            {
                "observations": torch.stack(observations),  # (B, K+L, 8, 8, 119)
                "actions": torch.stack(actions_out),  # (B, K)
                "rewards": torch.stack(rewards_out),  # (B, K+L)
                "move_masks": torch.stack(masks_out),  # (B, K, 4672)
                "game_buffer_positions": torch.tensor(buffer_positions),  # (B,)
                "window_lengths": torch.tensor(window_lengths),  # (B,)
            },
            batch_size=[batch_size],
        )

    def __len__(self) -> int:
        return self._total_steps

    def save(self, path: str = REPLAY_BUFFER_PATH) -> None:
        """Persist buffer to disk as a single .pt file."""
        torch.save(
            {
                "episodes": list(self._episodes),
                "total_steps": self._total_steps,
                "max_episodes": self.max_episodes,
            },
            path,
        )

    def load(self, path: str = REPLAY_BUFFER_PATH) -> bool:
        """Restore buffer from disk. Returns True if loaded."""
        if not os.path.exists(path):
            return False
        data = torch.load(path, map_location="cpu", weights_only=False)
        episodes = data.get("episodes", [])
        self.max_episodes = data.get("max_episodes", self.max_episodes)
        # Trim to max capacity in case max_episodes was lowered
        if len(episodes) > self.max_episodes:
            episodes = episodes[-self.max_episodes :]
        self._episodes = deque(episodes, maxlen=self.max_episodes)
        self._total_steps = sum(ep.batch_size[0] for ep in self._episodes)
        return True


rb = EpisodeReplayBuffer(max_episodes=10_000)


def save_to_replay_buffer(history: List[Step]) -> None:
    rb.add_episode(history)


def get_num_episodes() -> int:
    return rb.num_episodes


def sample_batch(batch_size: int = 128) -> TensorDict:
    return rb.sample(batch_size)


def save_replay_buffer(path: str = REPLAY_BUFFER_PATH) -> None:
    rb.save(path)


def load_replay_buffer(path: str = REPLAY_BUFFER_PATH) -> bool:
    return rb.load(path)
