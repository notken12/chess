import threading
from typing import List, TYPE_CHECKING
import torch
import numpy as np

from hyperparams import NUM_SAMPLED_ACTIONS, NUM_SIMS_IN_SEARCH

if TYPE_CHECKING:
    from model import Networks


class MinMaxStats:
    def __init__(self) -> None:
        self.minimum = float("inf")
        self.maximum = -float("inf")

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class TreeNode:
    """An expanded node in the search tree.

    Invariant: all network outputs (latent, policy_logits, value_pred,
    reward_pred) are non-None.  Unexpanded actions are tracked only in
    ``sampled_actions``; there are no placeholder child objects.
    """

    def __init__(
        self,
        latent: torch.Tensor,
        policy_logits: torch.Tensor,
        value_pred: torch.Tensor,
        reward_pred: torch.Tensor,
        q_value_min_max: MinMaxStats,
    ) -> None:
        self.latent = latent
        self.policy_logits = policy_logits
        self.policy_probs = torch.softmax(policy_logits, dim=0)
        self.value_pred = value_pred
        self.reward_pred = reward_pred

        self.children: dict[int, TreeNode] = {}
        self.sampled_actions: set[int] = set()

        self.q_value_min_max = q_value_min_max
        self.num_visits = 0
        self.average_value = 0.0
        self.most_visited_child_num_visits = 0

    def policy_weighted_average_values(self) -> float:
        total = 0.0
        for a in self.sampled_actions:
            if a in self.children:
                total += self.policy_probs[a].item() * self.children[a].average_value
        return total

    def pick_action(self) -> int:
        pwav = self.policy_weighted_average_values()
        valid_actions = list(self.sampled_actions)
        if not valid_actions:
            raise RuntimeError("pick_action called on node with no sampled_actions")
        scores = [self.action_score(a, pwav) for a in valid_actions]
        max_score = max(scores)
        best = [a for a, s in zip(valid_actions, scores) if s == max_score]
        return int(np.random.choice(best))

    def sigma(self, q: float) -> float:
        c_visit = 50
        c_scale = 0.1
        return (c_visit + self.most_visited_child_num_visits) * c_scale * q

    def completedQ(
        self, action_idx: int, policy_weighted_average_values: float
    ) -> float:
        if action_idx not in self.children:
            return self.q_value_min_max.normalize(policy_weighted_average_values)
        return self.q_value_min_max.normalize(self.children[action_idx].average_value)

    def action_score(
        self, action_idx: int, policy_weighted_average_values: float
    ) -> float:
        completedQ = self.completedQ(action_idx, policy_weighted_average_values)
        logProb = torch.log_softmax(self.policy_logits, dim=0)[action_idx].item()
        return logProb + self.sigma(completedQ)


action_space_size = 8 * 8 * 73
value_bins = torch.linspace(-1, 1, 51)
reward_bins = torch.tensor([-1, 0, 1])

_thread_local = threading.local()


def _rng() -> np.random.Generator:
    if not hasattr(_thread_local, "rng"):
        _thread_local.rng = np.random.default_rng()
    return _thread_local.rng


gamma = 1


def get_target_policy(
    cur_latent: torch.Tensor, cur_policy_logits: torch.Tensor, nets: Networks
):
    """call with tensors without batch dimension"""
    qmm = MinMaxStats()
    device = cur_latent.device

    with torch.no_grad():
        value_logits = nets.value(cur_latent)
        value_pred = torch.softmax(value_logits, dim=-1).squeeze(0)

    reward_pred = torch.zeros(3, device=device)

    root = TreeNode(
        latent=cur_latent,
        policy_logits=cur_policy_logits,
        value_pred=value_pred,
        reward_pred=reward_pred,
        q_value_min_max=qmm,
    )

    # Gumbel top-k at the root
    gumbel = torch.from_numpy(_rng().gumbel(size=action_space_size)).to(device)
    samples = gumbel + cur_policy_logits
    k = NUM_SAMPLED_ACTIONS
    top_k_idx = np.argpartition(samples.detach().cpu().numpy(), -k)[-k:]
    root.sampled_actions = set(int(a) for a in top_k_idx)

    # Sequential halving over root actions
    roots = [int(a) for a in top_k_idx]
    remain = k
    simulation_budget_per_node = NUM_SIMS_IN_SEARCH // NUM_SAMPLED_ACTIONS

    while remain > 1:
        for action_idx in roots:
            for _ in range(simulation_budget_per_node):
                _simulate([root], action_idx, nets)
        # Eliminate worse half (keep higher completedQ)
        roots.sort(
            key=lambda a: root.completedQ(a, root.policy_weighted_average_values()),
            reverse=True,
        )
        roots = roots[: remain // 2]
        remain = len(roots)
        simulation_budget_per_node *= 2

    root_action = roots[0]

    if root.children:
        root.most_visited_child_num_visits = max(
            c.num_visits for c in root.children.values()
        )

    target_policy = torch.ones_like(cur_policy_logits) * -float("inf")
    for action_idx in top_k_idx:
        cq = root.completedQ(action_idx, root.policy_weighted_average_values())
        target_policy[action_idx] = root.policy_logits[action_idx] + root.sigma(cq)
    target_policy = torch.softmax(target_policy, dim=0)
    return root_action, target_policy, root.average_value


def _simulate(path: List[TreeNode], action: int, nets: Networks):
    """Traverse from the last node in *path* via *action*, expanding if needed."""
    parent = path[-1]
    if action not in parent.children:
        _expand(path, action, nets)
        return

    child = parent.children[action]
    path.append(child)
    if not child.sampled_actions:
        return
    next_action = child.pick_action()
    _simulate(path, next_action, nets)


def _expand(path: List[TreeNode], action: int, nets: Networks):
    """Expand *action* from path[-1], backpropagate, and sample children."""
    parent = path[-1]
    device = parent.latent.device
    v_bins = value_bins.to(device)
    r_bins = reward_bins.to(device)

    with torch.no_grad():
        action_one_hot = encode_single_action(action, device)
        next_latent, reward_logits = nets.dynamics(parent.latent, action_one_hot)
        policy_logits = nets.policy(next_latent, masks=None).squeeze(0)
        value_logits = nets.value(next_latent)

        value_pred = torch.softmax(value_logits, dim=-1).squeeze(0)
        reward_pred = torch.softmax(reward_logits, dim=-1).squeeze(0)

        node = TreeNode(
            latent=next_latent,
            policy_logits=policy_logits,
            value_pred=value_pred,
            reward_pred=reward_pred,
            q_value_min_max=parent.q_value_min_max,
        )

        # Sample child candidates for this new node
        gumbel = torch.from_numpy(_rng().gumbel(size=action_space_size)).to(device)
        samples = gumbel + policy_logits
        child_k = 10
        child_top_k = torch.topk(samples, child_k).indices.cpu().numpy()
        node.sampled_actions = set(int(a) for a in child_top_k)

        parent.children[action] = node
        path.append(node)

        # Backpropagation (negamax)
        q_value = torch.sum(value_pred * v_bins).item()
        for i in range(len(path) - 1, -1, -1):
            node_i = path[i]
            immediate_reward = torch.sum(node_i.reward_pred * r_bins).item()
            new_avg = (node_i.num_visits * node_i.average_value + q_value) / (
                node_i.num_visits + 1
            )
            node_i.average_value = new_avg
            node_i.num_visits += 1

            q_value = immediate_reward - gamma * q_value
            node_i.q_value_min_max.update(q_value)

            if i > 0 and node_i.num_visits > path[i - 1].most_visited_child_num_visits:
                path[i - 1].most_visited_child_num_visits = node_i.num_visits


def encode_single_action(action_idx: int, device: torch.device):
    action_tensor = torch.zeros((1, 73, 8, 8), device=device)
    z = action_idx // 64
    x = (action_idx % 64) // 8
    y = action_idx % 8
    action_tensor[0, z, x, y] = 1.0
    return action_tensor
