"""
AtLoc — Attention-guided Absolute Pose Regression.

Based on: "AtLoc: Attention Guided Camera Localization" (Wang et al., AAAI 2020).

Key idea: a self-attention module on top of CNN feature maps allows the model
to focus on geometrically stable regions (corners, edges, structural features)
while suppressing distractors (textures, reflections, dynamic objects).
"""

import torch
import torch.nn as nn

from .backbone import build_backbone


class SelfAttention(nn.Module):
    """
    Non-local self-attention module for spatial feature maps.
    Computes attention weights over spatial positions to capture
    long-range dependencies in the feature map.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        mid = max(in_channels // 8, 1)

        self.query = nn.Conv2d(in_channels, mid, 1)
        self.key = nn.Conv2d(in_channels, mid, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q = self.query(x).view(B, -1, N).permute(0, 2, 1)  # (B, N, mid)
        k = self.key(x).view(B, -1, N)                       # (B, mid, N)
        attn = torch.bmm(q, k) / (q.shape[-1] ** 0.5)       # (B, N, N)
        attn = torch.softmax(attn, dim=-1)

        v = self.value(x).view(B, C, N)                      # (B, C, N)
        out = torch.bmm(v, attn.permute(0, 2, 1))            # (B, C, N)
        out = out.view(B, C, H, W)

        return self.gamma * out + x, attn


class AtLoc(nn.Module):
    def __init__(self, backbone: str = "resnet34", pretrained: bool = True,
                 unfreeze_last_n: int = 2, dropout: float = 0.3,
                 hidden_dim: int = 512, **kwargs):
        super().__init__()
        self.encoder = build_backbone({
            "backbone": backbone,
            "pretrained": pretrained,
            "unfreeze_last_n": unfreeze_last_n,
        })
        feat_dim = self.encoder.feat_dim

        self.attention = SelfAttention(feat_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        self.fc_xyz = nn.Linear(hidden_dim, 3)
        self.fc_q = nn.Linear(hidden_dim, 4)

        nn.init.xavier_uniform_(self.fc_xyz.weight)
        nn.init.xavier_uniform_(self.fc_q.weight)

    def forward(self, x, scene_id=None):
        feat_map, _ = self.encoder(x)
        att_feat, _ = self.attention(feat_map)
        pooled = self.avgpool(att_feat).flatten(1)

        feat = self.fc(pooled)
        xyz = self.fc_xyz(feat)
        q = self.fc_q(feat)
        q = q / (torch.norm(q, dim=1, keepdim=True) + 1e-8)

        return xyz, q

    def get_attention_maps(self, x):
        """Extract spatial attention weights for visualization."""
        feat_map, _ = self.encoder(x)
        _, attn = self.attention(feat_map)
        B, N, _ = attn.shape
        H = W = int(N ** 0.5)
        attn_map = attn.mean(dim=1).view(B, H, W)
        return attn_map
