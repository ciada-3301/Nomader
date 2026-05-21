"""
networks.py
-----------
Network architecture for Nomader RL policy.

MaskEncoder  : lightweight CNN  →  32-dim visual embedding from cam frame
               (will be replaced by SAM2 mask encoding on real hardware)

NomaderPolicy: Actor-Critic for PPO
  Inputs (all concatenated after encoding):
    - cam_frame       → MaskEncoder → (32,)
    - target_centroid              → (2,)
    - target_bbox                  → (4,)
    - depth_estimate               → (1,)
    - imu                          → (6,)
    - wheel_encoders               → (6,)
    - prev_action                  → (2,)
  Total tabular dim: 2+4+1+6+6+2 = 21
  Total fused dim  : 32 + 21     = 53

Actor  → 53 → 128 → 64 → 2   (mean of Gaussian, tanh-squashed)
Critic → 53 → 128 → 64 → 1   (state value)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -5
LOG_STD_MAX =  2


# ── Mask / Camera Encoder ────────────────────────────────────────────────────
class MaskEncoder(nn.Module):
    """
    3-layer conv net that compresses (3, H, W) cam frame → (32,) embedding.
    Intentionally tiny so it runs fast on Pi during real deployment.
    """
    def __init__(self, in_h=120, in_w=160):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),  # →(16,60,80)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # →(32,30,40)
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), # →(32,15,20)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),                           # →(32,2,2)
            nn.Flatten(),                                           # →(128,)
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

    def forward(self, x):
        # x: (B, H, W, 3) uint8 from env  →  normalise and permute
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        if x.ndim == 3:
            x = x.unsqueeze(0)
        # (B,H,W,C) → (B,C,H,W)
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.net(x)


# ── Shared Trunk ─────────────────────────────────────────────────────────────
class SharedTrunk(nn.Module):
    """
    Fuses mask encoding + tabular state into a shared representation
    that both actor and critic heads read from.
    """
    def __init__(self):
        super().__init__()
        fused_dim = 32 + 21   # mask_enc + tabular

        self.fc1 = nn.Linear(fused_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(128)

    def forward(self, mask_enc, tabular):
        x = torch.cat([mask_enc, tabular], dim=-1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return x   # (B, 128)


# ── Actor ────────────────────────────────────────────────────────────────────
class Actor(nn.Module):
    """
    Gaussian policy:  trunk → mean + log_std → throttle, steering ∈ [-1,1]
    """
    def __init__(self):
        super().__init__()
        self.mean_head   = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh(),          # bound output to [-1, 1]
        )
        self.log_std = nn.Parameter(torch.zeros(2))   # learnable, shared across states

    def forward(self, trunk_out):
        mean    = self.mean_head(trunk_out)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std     = log_std.exp().expand_as(mean)
        dist    = Normal(mean, std)
        return dist

    def get_action(self, trunk_out, deterministic=False):
        dist   = self.forward(trunk_out)
        action = dist.mean if deterministic else dist.rsample()
        action = action.clamp(-1.0, 1.0)
        log_p  = dist.log_prob(action).sum(-1, keepdim=True)
        return action, log_p, dist.entropy().sum(-1, keepdim=True)


# ── Critic ───────────────────────────────────────────────────────────────────
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, trunk_out):
        return self.value_head(trunk_out)


# ── Full Policy (Actor-Critic) ────────────────────────────────────────────────
class NomaderPolicy(nn.Module):
    """
    Full Actor-Critic network for PPO.
    Call .encode_obs(obs_dict) to get (trunk_out, mask_enc).
    """
    def __init__(self):
        super().__init__()
        self.mask_encoder = MaskEncoder()
        self.trunk        = SharedTrunk()
        self.actor        = Actor()
        self.critic       = Critic()

    def encode_obs(self, obs):
        """
        obs: dict with tensors (batched).
        Returns trunk_out (B,128) for actor/critic heads.
        """
        cam     = obs["cam_frame"]              # (B,H,W,3)
        mask_enc = self.mask_encoder(cam)        # (B,32)

        tabular = torch.cat([
            obs["target_centroid"],             # (B,2)
            obs["target_bbox"],                 # (B,4)
            obs["depth_estimate"],              # (B,1)
            obs["imu"],                         # (B,6)
            obs["wheel_encoders"],              # (B,6)
            obs["prev_action"],                 # (B,2)
        ], dim=-1)                              # (B,21)

        trunk_out = self.trunk(mask_enc, tabular)
        return trunk_out

    def get_action(self, obs, deterministic=False):
        trunk_out = self.encode_obs(obs)
        return self.actor.get_action(trunk_out, deterministic)

    def get_value(self, obs):
        trunk_out = self.encode_obs(obs)
        return self.critic(trunk_out)

    def get_action_and_value(self, obs, action=None):
        trunk_out   = self.encode_obs(obs)
        dist        = self.actor(trunk_out)
        if action is None:
            action  = dist.rsample().clamp(-1.0, 1.0)
        log_p       = dist.log_prob(action).sum(-1, keepdim=True)
        entropy     = dist.entropy().sum(-1, keepdim=True)
        value       = self.critic(trunk_out)
        return action, log_p, entropy, value


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)