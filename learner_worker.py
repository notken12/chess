import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast  

from tensordict import TensorDict
from torch.utils.data import DataLoader
from collections import deque, namedtuple
import numpy as np

import threading
from concurrent.futures import ThreadPoolExecutor
import time
import copy  

from model import Representation, Dynamics, Policy, Value
from batch_worker import provide_batch_transitions
from replay_buffer import sample_batch, K_STEPS, L_STEPS
from tree import MOVE_LOOKUP, value_bins  # For move encoding and categorical support


class Learner:
    def __init__(self, representation, dynamics, policy, value, lr=1e-4):
        # 1. Main Networks (The "Live" Weights)
        self.representation = representation
        self.dynamics = dynamics
        self.policy = policy
        self.value = value
        
        self.model_dict = {
            'representation': self.representation,
            'dynamics': self.dynamics,
            'policy': self.policy,
            'value': self.value
        }
        self.device = next(self.representation.parameters()).device

        # 2. Target Network (The "Stable" Reference for Consistency)
        # We only need the representation head for EfficientZero consistency
        self.target_representation = copy.deepcopy(representation)
        self.target_representation.requires_grad_(False)
        
        # 3. Optimization Setup
        # We group all parameters so the optimizer can update the whole "brain" at once
        self.optimizer = optim.Adam(
            list(self.representation.parameters()) + 
            list(self.dynamics.parameters()) + 
            list(self.policy.parameters()) + 
            list(self.value.parameters()), 
            lr=lr
        )
        self.scaler = torch.amp.GradScaler(device_type='cuda') # For Mixed Precision (FP16)
        
        # 4. Helper Metadata
        self.value_bins = value_bins # torch.linspace(-1, 1, num_bins)
        # Invert the MOVE_LOOKUP for fast action encoding
        self.index_to_coord = {v[0]: (v[0], v[1], v[2]) for k, v in MOVE_LOOKUP.items()}

    def update_step(self, training_step):
        batch = provide_batch_transitions(training_step)
        if batch is None: return
        
        # Move batch to the correct device (GPU/CPU)
        batch = batch.to(next(self.representation.parameters()).device)

        with torch.amp.autocast(device_type='cuda'):
            obs_0 = batch["observations"][:, 0].permute(0, 3, 1, 2)
            latent = self.representation(obs_0)
            
            total_loss = 0
            
            # Unroll Loop for BPTT
            for k in range(K_STEPS):
                # --- Predictions ---
                pred_value_logits = self.value(latent)
                # Note: Policy requires board objects for masking in your implementation
                # If boards are not in batch, you'd use a raw policy head here
                pred_policy_probs = self.policy(latent, batch["boards"][k])
                
                # --- Losses ---
                target_policy = batch["target_policies"][:, k]
                target_value = batch["target_values"][:, k]
                
                p_loss = self.compute_policy_loss(pred_policy_probs, target_policy)
                v_loss = self.compute_value_loss(pred_value_logits, target_value)
                total_loss += (p_loss + v_loss)

                # --- Transitions ---
                if k < K_STEPS - 1:
                    action_indices = batch["actions"][:, k]
                    action_tensor = self.encode_action(action_indices)
                    
                    # Dynamics returns (nextState, rewardProbs)
                    latent, pred_reward_logits = self.dynamics(latent, action_tensor)
                    
                    # Reward Loss
                    target_reward = batch["rewards"][:, k]
                    r_loss = self.compute_reward_loss(pred_reward_logits, target_reward)
                    
                    # Consistency Loss (Grounding the dream)
                    real_obs_k = batch["observations"][:, k+1].permute(0, 3, 1, 2)
                    with torch.no_grad():
                        real_latent = self.target_representation(real_obs_k)
                    
                    c_loss = self.compute_consistency_loss(latent, real_latent)
                    total_loss += (r_loss + c_loss)
                    
                    # Gradient Scaling: Scale by 1/2 at each step of the chain
                    latent.register_hook(lambda grad: grad * 0.5)

        # Optimization Step
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Slowly move the target network toward the live network
        self.update_target_network()

    # --- Loss Helper Functions ---
    
    def compute_policy_loss(self, pred_probs, target_policy):
        return -(target_policy * torch.log(pred_probs + 1e-8)).sum(dim=1).mean()

    def compute_value_loss(self, pred_logits, target_value):
        target_dist = self.scalar_to_categorical(target_value)
        return -(target_dist * F.log_softmax(pred_logits, dim=1)).sum(dim=1).mean()

    def compute_reward_loss(self, pred_logits, target_reward):
        # Maps -1, 0, 1 to class indices 0, 1, 2
        return F.cross_entropy(pred_logits, (target_reward + 1).long())

    def compute_consistency_loss(self, pred_latent, real_latent):
        p = F.normalize(pred_latent.flatten(1), p=2, dim=1)
        r = F.normalize(real_latent.flatten(1), p=2, dim=1)
        return F.mse_loss(p, r)

    def update_target_network(self, tau=0.005):
        for t_p, p in zip(self.target_representation.parameters(), self.representation.parameters()):
            t_p.data.mul_(1.0 - tau).add_(p.data * tau)

    def encode_action(self, action_indices):
        B = action_indices.shape[0]
        action_tensor = torch.zeros((B, 73, 8, 8), device=action_indices.device)
        for b in range(B):
            idx = action_indices[b].item()
            if idx in self.index_to_coord:
                z, x, y = self.index_to_coord[idx]
                action_tensor[b, z, x, y] = 1.0
        return action_tensor

    def scalar_to_categorical(self, v):
        # Maps float value to the 51-bin histogram
        v = v.clamp(-1, 1)
        bin_width = self.value_bins[1] - self.value_bins[0]
        centered_v = (v - self.value_bins[0]) / bin_width
        low = centered_v.floor().long()
        high = (low + 1).clamp(0, len(self.value_bins)-1)
        w_high = centered_v - low
        dist = torch.zeros((v.size(0), len(self.value_bins)), device=v.device)
        dist.scatter_(1, low.unsqueeze(1).clamp(0, len(self.value_bins)-1), (1-w_high).unsqueeze(1))
        dist.scatter_(1, high.unsqueeze(1), w_high.unsqueeze(1))
        return dist