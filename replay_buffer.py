from typing import List, Tuple
import torch
from collections import deque
from tensordict import TensorDict
import numpy as np

# (observation (8,8,119), action index, reward)
# Target policies are intentionally excluded: EfficientZero V2 uses reanalysis,
# recomputing them from stored observations with the current network at training
# time so the policy head always trains against up-to-date targets.
Step = Tuple[torch.Tensor, int, int]

K_STEPS = 5   # unroll depth: number of dynamics steps during training
L_STEPS = 5   # TD horizon: number of real reward steps before bootstrapping


class EpisodeReplayBuffer:
    """
    Stores full game episodes and samples (K+L)-step windows for MuZero unrolling.

    Each sample contains K unroll steps followed by L extra steps needed to
    compute proper l-step TD targets for every position in the unroll window.
    For position j in [0, K), the TD target uses rewards at j..j+L-1 and
    bootstraps with V(s_{t+j+L}); all of these lie within the K+L window.

    Sampling is weighted by episode length so that every individual step has
    an equal probability of being selected as a window start, regardless of
    game length. Windows that extend past the end of an episode are zero-padded
    with a boolean mask marking which steps are real.

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
        mask                  (B, K+L)             — 1 for real steps, 0 for padding
        game_buffer_positions (B,)                 — deque index of each sampled game
                                                     (0 = oldest, N-1 = newest); used
                                                     by the batch worker to identify
                                                     recently-added games that should
                                                     use TD targets instead of MCTS values.
    """

    def __init__(
        self,
        max_episodes: int = 10_000,
        k_steps: int = K_STEPS,
        l_steps: int = L_STEPS,
    ) -> None:
        self.max_episodes = max_episodes
        self.k_steps = k_steps
        self.l_steps = l_steps
        self._episodes: deque[TensorDict] = deque()
        self._total_steps = 0

    def add_episode(self, history: List[Step]) -> None:
        observations, actions, rewards = zip(*history)
        episode = TensorDict(
            {
                "observation": torch.stack(observations),  # (T, 8, 8, 119)
                "action": torch.tensor(actions),           # (T,)
                "reward": torch.tensor(rewards, dtype=torch.float32),  # (T,)
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
        episode_indices = np.random.choice(len(episodes), size=batch_size, p=probs, replace=True)

        total_steps = self.k_steps + self.l_steps
        obs_blank = None  # lazily set to a zero observation on first pad
        observations, actions_out, rewards_out, masks, buffer_positions = [], [], [], [], []

        for ep_idx in episode_indices:
            ep = episodes[ep_idx]
            buffer_positions.append(ep_idx)
            T = ep.batch_size[0]
            t = np.random.randint(0, T)

            if obs_blank is None:
                obs_blank = torch.zeros_like(ep["observation"][0])

            ep_obs, ep_actions, ep_rewards, ep_mask = [], [], [], []
            for k in range(total_steps):
                valid = t + k < T
                ep_obs.append(ep["observation"][t + k] if valid else obs_blank)
                ep_rewards.append(ep["reward"][t + k] if valid else torch.zeros(()))
                ep_mask.append(1.0 if valid else 0.0)
                # Actions are only needed for the K unroll steps, not the extra L.
                if k < self.k_steps:
                    ep_actions.append(
                        ep["action"][t + k] if valid else torch.zeros((), dtype=torch.long)
                    )

            observations.append(torch.stack(ep_obs))       # (K+L, 8, 8, 119)
            actions_out.append(torch.stack(ep_actions))    # (K,)
            rewards_out.append(torch.stack(ep_rewards))    # (K+L,)
            masks.append(torch.tensor(ep_mask))            # (K+L,)

        # target_policies are not included: the training loop recomputes them
        # via reanalysis (MCTS with the current network) on sampled observations.
        return TensorDict(
            {
                "observations": torch.stack(observations),              # (B, K+L, 8, 8, 119)
                "actions": torch.stack(actions_out),                    # (B, K)
                "rewards": torch.stack(rewards_out),                    # (B, K+L)
                "mask": torch.stack(masks),                             # (B, K+L)
                "game_buffer_positions": torch.tensor(buffer_positions),# (B,)
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
