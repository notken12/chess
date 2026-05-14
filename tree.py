import threading
from typing import List, TYPE_CHECKING, Optional
import torch
import numpy as np
from model import Networks
import hyperparams

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
            value = (value - self.minimum) / max(self.maximum - self.minimum, 1e-6)
        value = max(min(value, 1.0), 0.0)
        return value


class TreeNode:
    """An expanded node in the search tree.

    Invariant: all network outputs (latent, policy_logits, value_pred,
    reward_pred) are non-None.
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
        # policy_probs computed on demand — not stored as a persistent tensor
        self.value_pred = value_pred
        self.reward_pred = reward_pred

        self.children: dict[int, TreeNode] = {}
        self.sampled_actions: set[int] = set()
        self.selected_children_idx: list[int] = []  # root only

        self.q_value_min_max = q_value_min_max
        self.num_visits = 0
        self.average_value = 0.0
        self.most_visited_child_num_visits = 0
        self.reward_scalar = 0.0
        self.policy_probs = torch.softmax(policy_logits, dim=0)

    def policy_weighted_average_values(self) -> float:
        """v_mix from Appendix B — blends node value with empirical Q average."""
        total = 0.0
        pi_sum = 0.0
        visit_sum = 0
        for a, child in self.children.items():
            q = child.reward_scalar - gamma * child.average_value
            prob = self.policy_probs[a].item()
            total += prob * q
            pi_sum += prob
            visit_sum += child.num_visits
        if pi_sum < 1e-6:
            return self.average_value
        pi_qsa_avg = total / pi_sum
        v_mix = (1.0 / (1.0 + visit_sum)) * (
            self.average_value + visit_sum * pi_qsa_avg
        )
        return v_mix

    def pick_action(self, value_min_max: MinMaxStats) -> int:
        """Non-root action selection over sampled_actions only."""
        valid_actions = list(self.sampled_actions)
        if not valid_actions:
            raise RuntimeError("pick_action called on node with no sampled_actions")

        pwav = self.policy_weighted_average_values()
        device = self.policy_logits.device

        cqs = []
        for a in valid_actions:
            if a in self.children:
                child = self.children[a]
                q = child.reward_scalar - gamma * child.average_value
                cqs.append(value_min_max.normalize(q))
            else:
                cqs.append(value_min_max.normalize(pwav))
        cqs_t = torch.tensor(cqs, device=device)

        max_visits = max(
            (c.num_visits for c in self.children.values()), default=0
        )
        sigmas = (50 + max_visits) * 0.1 * cqs_t
        valid_logits = self.policy_logits[valid_actions] + sigmas
        improved_policy = torch.softmax(valid_logits, dim=0).cpu().numpy()

        visits = np.array(
            [self.children[a].num_visits if a in self.children else 0
             for a in valid_actions],
            dtype=np.float32,
        )
        visit_sum = visits.sum()
        scores = improved_policy - visits / (1.0 + visit_sum)
        max_score = scores.max()
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
        child = self.children[action_idx]
        q = child.reward_scalar - gamma * child.average_value
        return self.q_value_min_max.normalize(q)


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

    # Precompute reward scalar once per expansion batch
    reward_scalars = torch.sum(reward_preds * reward_bins.to(device), dim=-1)

    for i, (path, action) in enumerate(unique):
        parent = path[-1]
        node = TreeNode(
            latent=next_latents[i : i + 1],
            policy_logits=policy_logits[i],
            value_pred=value_preds[i],
            reward_pred=reward_preds[i],
            q_value_min_max=parent.q_value_min_max,
        )
        node.reward_scalar = reward_scalars[i].item()
        # Sample child candidates for this new node
        gumbel = torch.from_numpy(
            _rng().gumbel(size=action_space_size).astype(np.float32)
        ).to(device)
        legal_mask = policy_logits[i] > -1e8
        legal_actions = torch.where(legal_mask)[0]
        if len(legal_actions) == 0:
            legal_actions = torch.tensor([policy_logits[i].argmax().item()], device=device)
        child_k = min(hyperparams.NUM_SAMPLED_ACTIONS, len(legal_actions))
        samples = gumbel[legal_actions] + policy_logits[i][legal_actions]
        _, top_local = torch.topk(samples, child_k)
        node.sampled_actions = set(int(legal_actions[idx].item()) for idx in top_local.cpu().numpy())
        parent.children[action] = node


def _backprop(path, leaf_value, v_bins, r_bins):
    """Backpropagate a leaf value up a path (negamax)."""
    q_value = leaf_value
    for i in range(len(path) - 1, -1, -1):
        node_i = path[i]
        immediate_reward = node_i.reward_scalar
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
    """Batched Gumbel search matching official EZ-V2 discrete implementation.

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
        # Official initialization: root gets 1 visit and its value estimate
        root_value = torch.sum(value_preds[b] * value_bins.to(device)).item()
        root.average_value = root_value
        root.num_visits = 1
        qmm.update(root_value)
        roots.append(root)

    # Gumbel noise (temperature = 1.0)
    gumbel = torch.from_numpy(
        _rng().gumbel(size=(B, action_space_size)).astype(np.float32)
    ).to(device)

    # Initialize root selected_children_idx with ALL legal actions sorted by gumbel + logit
    for b in range(B):
        legal_mask = policy_logits[b] > -1e8
        legal_actions = torch.where(legal_mask)[0].cpu().numpy()
        if len(legal_actions) == 0:
            legal_actions = np.array([policy_logits[b].argmax().item()])
        scores = gumbel[b, legal_actions] + policy_logits[b, legal_actions]
        sorted_idx = torch.argsort(scores, descending=True).cpu().numpy()
        roots[b].selected_children_idx = [int(legal_actions[i]) for i in sorted_idx]

    # Phase tracking (identical to official EZ-V2)
    k = hyperparams.NUM_SAMPLED_ACTIONS
    n = hyperparams.NUM_SIMS_IN_SEARCH
    current_phase = [0] * B
    current_num_top_actions = [k] * B
    used_visit_num = [0] * B
    visit_num_for_next_phase = []
    for _ in range(B):
        v = max(1, int(np.floor(n / (np.log2(k) * k)))) * k
        visit_num_for_next_phase.append(v)

    v_bins = value_bins.to(device)
    r_bins = reward_bins.to(device)

    def do_equal_visit(node: TreeNode) -> int:
        """Pick the least-visited action from selected_children_idx."""
        min_visits = float("inf")
        best_action = -1
        for a in node.selected_children_idx:
            visits = node.children[a].num_visits if a in node.children else 0
            if visits < min_visits:
                min_visits = visits
                best_action = a
        assert best_action >= 0
        return best_action

    def sequential_halving(
        root: TreeNode,
        gumbel_noise: torch.Tensor,
        value_min_max: MinMaxStats,
        phase: int,
        num_top: int,
    ):
        """Narrow selected_children_idx exactly like the official code."""
        if phase == 0:
            # Phase 0 setup: slice the already-sorted full list to top num_top
            root.selected_children_idx = root.selected_children_idx[:num_top]
        else:
            selected = root.selected_children_idx
            pwav = root.policy_weighted_average_values()
            scores = []
            for a in selected:
                cq = root.completedQ(a, pwav)
                score = (
                    gumbel_noise[a].item()
                    + root.policy_logits[a].item()
                    + root.sigma(cq)
                )
                scores.append(score)
            sorted_idx = np.argsort(scores)[::-1]
            root.selected_children_idx = [
                selected[i] for i in sorted_idx[:num_top]
            ]

    # Main search loop: one simulation at a time
    for simulation_idx in range(n):
        # 1. Selection: traverse one path per tree
        selections = []  # (b, path, action_to_expand)
        for b in range(B):
            root = roots[b]
            action = do_equal_visit(root)
            path = [root]
            # Descend until we select an unexpanded action
            while action in path[-1].children:
                child = path[-1].children[action]
                path.append(child)
                action = child.pick_action(child.q_value_min_max)
            selections.append((b, path, action))

        # 2. Expansion (batched across the batch)
        paths_actions = [(path, action) for _b, path, action in selections]
        _batched_expand(paths_actions, nets)

        # 3. Backpropagation
        for _b, path, action in selections:
            child = path[-1].children[action]
            path.append(child)
            leaf_value = torch.sum(child.value_pred * v_bins).item()
            _backprop(path, leaf_value, v_bins, r_bins)

        # 4. Phase transition check (identical to official ready_for_next_gumble_phase)
        for b in range(B):
            if simulation_idx + 1 >= visit_num_for_next_phase[b]:
                current_phase[b] += 1
                current_num_top_actions[b] //= 2

                current_m = current_num_top_actions[b]
                if current_m > 2:
                    extra_visit = (
                        int(np.floor(n / (np.log2(k) * current_m))) * current_m
                    )
                else:
                    extra_visit = n - used_visit_num[b]

                used_visit_num[b] += extra_visit
                visit_num_for_next_phase[b] += extra_visit
                visit_num_for_next_phase[b] = min(
                    visit_num_for_next_phase[b], n
                )

                sequential_halving(
                    roots[b],
                    gumbel[b],
                    roots[b].q_value_min_max,
                    current_phase[b],
                    current_num_top_actions[b],
                )

    root_actions = torch.tensor(
        [roots[b].selected_children_idx[0] for b in range(B)],
        device=device,
        dtype=torch.long,
    )

    # Build target policy over full action space — vectorised
    target_policies = torch.zeros(B, action_space_size, device=device)
    for b in range(B):
        if roots[b].children:
            roots[b].most_visited_child_num_visits = max(
                c.num_visits for c in roots[b].children.values()
            )
        pwav = roots[b].policy_weighted_average_values()
        # Default completedQ for unvisited actions
        cq_default = roots[b].q_value_min_max.normalize(pwav)
        cq_vals = torch.full((action_space_size,), cq_default, device=device)
        for a, child in roots[b].children.items():
            q = child.reward_scalar - gamma * child.average_value
            cq_vals[a] = roots[b].q_value_min_max.normalize(q)
        sigmas = (50 + roots[b].most_visited_child_num_visits) * 0.1 * cq_vals
        target_policies[b] = roots[b].policy_logits + sigmas
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
    next_action = child.pick_action(child.q_value_min_max)
    _simulate(path, next_action, nets)


def _expand(path: List[TreeNode], action: int, nets: Networks):
    """Expand *action* from path[-1], backpropagate."""
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
        node.reward_scalar = torch.sum(reward_pred * r_bins).item()
        # Sample child candidates for this new node
        gumbel = torch.from_numpy(
            _rng().gumbel(size=action_space_size).astype(np.float32)
        ).to(device)
        legal_mask = policy_logits > -1e8
        legal_actions = torch.where(legal_mask)[0]
        if len(legal_actions) == 0:
            legal_actions = torch.tensor([policy_logits.argmax().item()], device=device)
        child_k = min(hyperparams.NUM_SAMPLED_ACTIONS, len(legal_actions))
        samples = gumbel[legal_actions] + policy_logits[legal_actions]
        _, top_local = torch.topk(samples, child_k)
        node.sampled_actions = set(int(legal_actions[idx].item()) for idx in top_local.cpu().numpy())
        parent.children[action] = node
        path.append(node)

        # Backpropagation (negamax)
        q_value = torch.sum(value_pred * v_bins).item()
        for i in range(len(path) - 1, -1, -1):
            node_i = path[i]
            immediate_reward = node_i.reward_scalar
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
