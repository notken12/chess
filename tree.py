import threading
from typing import List, cast
import torch
import numpy as np


class MinMaxStats:
    def __init__(self) -> None:
        self.minimum = float("inf")
        self.maximum = -float("inf")

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class TreeNode:
    def __init__(
        self,
        action_space_size: int,
        latent: torch.Tensor | None,
        policy_logits: torch.Tensor | None,
        value_pred: torch.Tensor | None,
        reward_pred: torch.Tensor | None,
        q_value_min_max: MinMaxStats,
        num_visits=0,
    ) -> None:
        self.action_space_size = action_space_size
        self.children: List[TreeNode | None] = [None] * action_space_size
        self.latent = latent
        self.policy_logits = policy_logits
        self.policy_probs = (
            torch.softmax(policy_logits, dim=0) if policy_logits is not None
        )
        self.value_pred = value_pred
        self.num_visits = num_visits
        self.reward_pred = reward_pred
        self.q_value_min_max = q_value_min_max
        self.average_value = 0
        self.most_visited_child_num_visits = 0
        pass

    def policy_weighted_average_values(self):
        probs = [
            self.policy_probs[a].item()
            for a in range(self.action_space_size)
            if self.children[a]
        ]
        values = [
            self.children[a].average_value
            for a in range(self.action_space_size)
            if self.children[a]
        ]
        return sum(p * v for p, v in zip(probs, values))

    def pick_action(self) -> int:
        policy_weighted_average_values = self.policy_weighted_average_values()
        valid_actions = [
            a for a in range(self.action_space_size) if self.children[a] is not None
        ]
        scores = [
            self.action_score(a, policy_weighted_average_values) for a in valid_actions
        ]
        max_score = max(scores)
        best_valid_actions = [
            valid_actions[i]
            for i in range(len(valid_actions))
            if scores[i] == max_score
        ]
        return np.random.choice(best_valid_actions)

    def sigma(self, q):
        c_visit = 50
        c_scale = 0.1
        return (c_visit + self.most_visited_child_num_visits) * c_scale * q

    def completedQ(self, action_idx: int, policy_weighted_average_values: float):
        child = cast(TreeNode, self.children[action_idx])
        n = child.num_visits
        return self.q_value_min_max.normalize(
            child.average_value if n > 0 else policy_weighted_average_values
        )

    def action_score(
        self,
        action_idx: int,
        policy_weighted_average_values: float,
    ):
        completedQ = self.completedQ(action_idx, policy_weighted_average_values)
        logProb = torch.log_softmax(self.policy_logits, dim=0)[action_idx].item()
        return logProb + self.sigma(completedQ)


action_space_size = 8 * 8 * 73
value_bins = torch.linspace(-1, 1, 51)
reward_bins = torch.tensor([-1, 0, 1])

# Each thread gets its own RNG so parallel reanalysis workers don't corrupt
# each other's random state.
_thread_local = threading.local()


def _rng() -> np.random.Generator:
    if not hasattr(_thread_local, "rng"):
        _thread_local.rng = np.random.default_rng()
    return _thread_local.rng


gamma = 1


# first we make a overall root that represents the current state
# then we sample a K root actions and attach them to the overall root
# then we run sequential halving rounds to eliminate half of those root actions each time and double our num simulations
def get_target_policy(cur_latent: torch.Tensor, cur_policy_logits: torch.Tensor, nets:Networks):
    q_value_min_max = MinMaxStats()
    root_node = TreeNode(
        action_space_size,
        cur_latent,
        cur_policy_logits,
        torch.zeros([51]), 
        torch.zeros([3]),
        q_value_min_max,
    )
    # sample Gumbel top K here and add as child nodes
    gumbel_noise = torch.from_numpy(_rng().gumbel(size=action_space_size)).to(cur_latent.device)
    samples = gumbel_noise + cur_policy_logits
    k = 16
    top_k_idx = np.argpartition(samples, -k)[-k:]
    roots = []
    for action_idx in top_k_idx:
        node = TreeNode(action_space_size, None, None, None, None, q_value_min_max)
        root_node.children[action_idx] = node
        roots.append((node, action_idx))
        pass
    remain = k
    simulation_budget_per_node = 100
    while remain > 1:
        for r, action_idx in roots:
            for i in range(simulation_budget_per_node):
                simulate([root_node], action_idx, r, nets)
        # eliminate worse half
        roots.sort(key=lambda r: r[0].average_value)
        roots = roots[remain // 2 :]
        remain = len(roots)
        # double budget for each
        simulation_budget_per_node *= 2

    root_action = roots[0][1]
    root_node.most_visited_child_num_visits = max(
        root_node.children[a].num_visits for a in top_k_idx
    )
    # discrete (original Gumbel MuZero): π'(a) = softmax(logits(a) + σ(completedQ(a)))
    target_policy = torch.ones_like(cur_policy_logits) * -float("inf")
    for action_idx in top_k_idx:
        completedQ = root_node.completedQ(
            action_idx, root_node.policy_weighted_average_values()
        )
        target_policy[action_idx] = root_node.policy_logits[
            action_idx
        ] + root_node.sigma(completedQ)
    target_policy = torch.softmax(target_policy, dim=0)
    return root_action, target_policy, root_node.average_value


def simulate(prev_path: List[TreeNode], last_action: int, start_node: TreeNode, nets:Networks):
    # models: {'representation': ..., 'dynamics': ..., 'policy': ..., 'value': ...}
    q_value_min_max = start_node.q_value_min_max
    device = start_node.latent.device
    v_bins = value_bins.to(device)
    r_bins = reward_bins.to(device)
    with torch.no_grad():
        search_path = list(prev_path)  # node
        cur_node: TreeNode = start_node
        reached_leaf = False
        last_action = last_action
        while not reached_leaf:
            if cur_node.num_visits == 0:   
                # this is a leaf
                # TODO: simulate action via the dynamics function
                # and get predicted value and reward
                # 1. Get the parent and action that led here
                parent_node, last_action = search_path[-1]

                # 2. Encode the scalar action into a (1, 73, 8, 8) tensor
                # This matches your Dynamics input: (latent + action_channels)
                action_one_hot = encode_single_action(last_action, device)

                # 3. Model Inference (Expansion)
                # next_latent: (1, 256, 8, 8), reward_logits: (1, 3)
                next_latent, reward_logits = nets.dynamics(parent_node.latent, action_one_hot)

                # 4. Leaf Predictions
                # policy_logits: (1, 4672), value_logits: (1, 51)
                policy_logits = nets.policy(next_latent, boards=None)
                value_logits = nets.value(next_latent)

                # 5. Populate the Leaf Node
                cur_node.latent = next_latent
                cur_node.policy_logits = policy_logits.squeeze(0) # Store as (4672,)

                # Convert categorical logits to probability distributions for the backprop
                cur_node.value_pred = torch.softmax(value_logits, dim=-1).squeeze(0) # (51,)
                cur_node.reward_pred = torch.softmax(reward_logits, dim=-1).squeeze(0) # (3,)

                # backpropagate to update average_value for this node and parents
                search_path.append(cur_node)
                q_value = torch.sum(cur_node.value_pred * value_bins).item()
                for i in range(len(search_path) - 1, -1, -1):
                    node = search_path[i]
                    immediate_reward=torch.sum(node.reward_pred * reward_bins).item()
                    # crucially, we subtract the q value of the child node because it belongs to the opponent
                    # and we want to minimize the opponent's reward (negamax)
                    q_value = immediate_reward - gamma*q_value
                    new_average = (node.num_visits * node.average_value + q_value) / (
                        node.num_visits + 1
                    )
                    node.average_value = new_average
                    node.num_visits += 1
                    q_value_min_max.update(q_value)

                    if i > 0 and (
                        node.num_visits
                        > search_path[i - 1].most_visited_child_num_visits
                    ):
                        search_path[i - 1].most_visited_child_num_visits = (
                            node.num_visits
                        )

                # sample Gumbel top K here and add as child nodes
                gumbel_noise = torch.from_numpy(_rng().gumbel(size=action_space_size)).to(device)
                samples = gumbel_noise + cur_node.policy_logits
                k = 10
                top_k_idx = torch.topk(samples, k).indices.cpu().numpy()
                for action_idx in top_k_idx:
                    cur_node.children[action_idx]=TreeNode(action_space_size, None, None, None, None, q_value_min_max   )

                reached_leaf = True
                break

            last_action = cur_node.pick_action()
            search_path.append(cur_node)
            cur_node = cast(
                TreeNode, cur_node.children[last_action]
            )  # non-leaf nodes will always have children and only choose actions that are in their children list

def encode_single_action(action_idx: int, device: torch.device):
    """
    Hardware adapter: Scalar Index -> 3D One-Hot Tensor
    """
    # Create empty action planes
    action_tensor = torch.zeros((1, 73, 8, 8), device=device)
    
    # Use your MOVE_LOOKUP logic to find the coordinates
    # For example: plane_idx = action_idx // 64; x = (action_idx % 64) // 8; y = action_idx % 8
    z = action_idx // 64
    x = (action_idx % 64) // 8
    y = action_idx % 8
    
    action_tensor[0, z, x, y] = 1.0
    return action_tensor