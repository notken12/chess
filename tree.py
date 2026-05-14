import threading
from typing import List, TYPE_CHECKING
import torch
import numpy as np
from model import Networks
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
reward_bins = torch.linspace(-1, 1, 51)

_thread_local = threading.local()


def _rng() -> np.random.Generator:
    if not hasattr(_thread_local, "rng"):
        _thread_local.rng = np.random.default_rng()
    return _thread_local.rng


gamma = 1


def encode_actions(action_indices, device):
    """Batched action encoding: (N,) indices -> (N, 73, 8, 8) one-hot."""
    N = len(action_indices)
    action_tensor = torch.zeros((N, 73, 8, 8), device=device)
    if N == 0:
        return action_tensor
    indices = torch.tensor(action_indices, device=device)
    z = indices // 64
    x = (indices % 64) // 8
    y = indices % 8
    action_tensor[torch.arange(N, device=device), z, x, y] = 1.0
    return action_tensor


def _batched_expand(expansions, nets):
    """Expand multiple (path, action) pairs with a single batched network call.
    Deduplicates identical (parent, action) requests within the batch."""
    if not expansions:
        return

    device = expansions[0][0][-1].latent.device
    seen = set()
    unique = []
    for path, action in expansions:
        key = (id(path[-1]), action)
        if key not in seen:
            seen.add(key)
            unique.append((path, action))

    if not unique:
        return

    parent_latents = torch.cat([path[-1].latent for path, _ in unique], dim=0)
    actions = [action for _, action in unique]
    action_tensor = encode_actions(actions, device)

    with torch.no_grad():
        next_latents, reward_logits = nets.dynamics(parent_latents, action_tensor)
        policy_logits = nets.policy(next_latents, masks=None)
        value_logits = nets.value(next_latents)
        value_preds = torch.softmax(value_logits, dim=-1)
        reward_preds = torch.softmax(reward_logits, dim=-1)

    for i, (path, action) in enumerate(unique):
        parent = path[-1]
        node = TreeNode(
            latent=next_latents[i : i + 1],
            policy_logits=policy_logits[i],
            value_pred=value_preds[i],
            reward_pred=reward_preds[i],
            q_value_min_max=parent.q_value_min_max,
        )
        gumbel = torch.from_numpy(
            _rng().gumbel(size=action_space_size).astype(np.float32)
        ).to(device)
        samples = gumbel + policy_logits[i]
        _, child_top_k = torch.topk(samples, NUM_SAMPLED_ACTIONS)
        node.sampled_actions = set(int(a) for a in child_top_k.cpu().numpy())
        parent.children[action] = node


def _backprop(path, leaf_value, v_bins, r_bins):
    """Backpropagate a leaf value up a path (negamax)."""
    q_value = leaf_value
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


def batched_get_target_policy(latents, policy_logits, nets):
    """Batched Gumbel search.  Inputs have batch dimension.

    Args:
        latents: (B, C, H, W)
        policy_logits: (B, A)  -- already masked (illegal = -1e9)
        nets: Networks

    Returns:
        root_actions: (B,) int64
        target_policies: (B, A)
        root_values: (B,)
    """
    B = latents.shape[0]
    device = latents.device

    with torch.no_grad():
        value_logits = nets.value(latents)
        value_preds = torch.softmax(value_logits, dim=-1)

    roots = []
    for b in range(B):
        qmm = MinMaxStats()
        root = TreeNode(
            latents[b : b + 1],
            policy_logits[b],
            value_preds[b],
            torch.zeros(51, device=device),
            qmm,
        )
        roots.append(root)

    # Sample root actions per position
    gumbel = torch.from_numpy(
        _rng().gumbel(size=(B, action_space_size)).astype(np.float32)
    ).to(device)
    perturbed = gumbel + policy_logits

    k = NUM_SAMPLED_ACTIONS
    active = []
    for b in range(B):
        legal_mask = policy_logits[b] > -1e8
        legal_actions = torch.where(legal_mask)[0]
        if len(legal_actions) == 0:
            legal_actions = torch.tensor(
                [policy_logits[b].argmax().item()], device=device
            )
        kb = min(k, len(legal_actions))
        top = legal_actions[torch.topk(perturbed[b, legal_actions], kb).indices]
        active.append([int(a) for a in top.cpu().numpy()])
        roots[b].sampled_actions = set(active[-1])

    # Gumbel noise per sampled action per tree must persist across every
    # halving round: Gumbel sequential halving selects on
    # g(a) + logit(a) + sigma(q(a)), and reusing the original noise is what
    # makes the selection consistent across rounds.
    g_roots = [{a: gumbel[b, a].item() for a in active[b]} for b in range(B)]

    num_rounds = max(1, int(np.ceil(np.log2(k))))
    remain = k

    v_bins = value_bins.to(device)
    r_bins = reward_bins.to(device)

    def gumbel_score(root, g_root, a, policy_weighted_average_values):
        cq = root.completedQ(a, policy_weighted_average_values)
        return g_root[a] + root.policy_logits[a].item() + root.sigma(cq)

    while remain > 1:
        sim_budget = max(1, NUM_SIMS_IN_SEARCH // (num_rounds * remain))
        for _ in range(sim_budget):
            # One simulation per active action per tree
            paths = [(b, [roots[b]], a) for b in range(B) for a in active[b]]
            completed = []

            while paths:
                need_expand = []
                next_paths = []

                for b, path, action in paths:
                    node = path[-1]
                    if action not in node.children:
                        need_expand.append((path, action))
                    else:
                        child = node.children[action]
                        path.append(child)
                        if not child.sampled_actions:
                            completed.append((b, path))
                        else:
                            next_paths.append((b, path, child.pick_action()))

                if need_expand:
                    _batched_expand(need_expand, nets)
                    for path, action in need_expand:
                        child = path[-1].children[action]
                        path.append(child)
                        # After expansion the simulation ends (backprop below)
                        completed.append((None, path))

                paths = next_paths

            # Backpropagate all completed paths
            for _b, path in completed:
                leaf = path[-1]
                leaf_value = torch.sum(leaf.value_pred * v_bins).item()
                _backprop(path, leaf_value, v_bins, r_bins)

        # Eliminate worse half per tree by Gumbel score
        # g(a) + logit(a) + sigma(q(a))
        for b in range(B):
            pwav = roots[b].policy_weighted_average_values()
            active[b].sort(
                key=lambda a, b=b, pwav=pwav: gumbel_score(
                    roots[b], g_roots[b], a, pwav
                ),
                reverse=True,
            )
            active[b] = active[b][: remain // 2]
        remain = remain // 2

    root_actions = torch.tensor(
        [active[b][0] for b in range(B)], device=device, dtype=torch.long
    )

    # Build target policy over full action space for each tree
    target_policies = torch.zeros(B, action_space_size, device=device)
    for b in range(B):
        pwav = roots[b].policy_weighted_average_values()
        for a in range(action_space_size):
            cq = roots[b].completedQ(a, pwav)
            target_policies[b, a] = roots[b].policy_logits[a] + roots[b].sigma(cq)
        if roots[b].children:
            roots[b].most_visited_child_num_visits = max(
                c.num_visits for c in roots[b].children.values()
            )
    target_policies = torch.softmax(target_policies, dim=-1)
    root_values = torch.tensor(
        [roots[b].average_value for b in range(B)], device=device
    )

    return root_actions, target_policies, root_values


def get_target_policy(
    cur_latent: torch.Tensor, cur_policy_logits: torch.Tensor, nets: Networks
):
    """Single-instance wrapper around the batched version.

    ``cur_latent`` already has a batch dimension of 1 from the caller.
    ``cur_policy_logits`` is unbatched (A,) and needs unsqueeze.
    """
    actions, policies, values = batched_get_target_policy(
        cur_latent, cur_policy_logits.unsqueeze(0), nets
    )
    return int(actions[0].item()), policies[0], float(values[0].item())


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
        gumbel = torch.from_numpy(
            _rng().gumbel(size=action_space_size).astype(np.float32)
        ).to(device)
        samples = gumbel + policy_logits
        child_k = NUM_SAMPLED_ACTIONS
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
