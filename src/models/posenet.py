"""
PoseNet+ — improved CNN-based absolute pose regression.

Improvements over original PoseNet / MiniPoseNet:
- Configurable backbone (ResNet34 default instead of ResNet18)
- Partial backbone unfreezing for fine-tuning
- Larger FC head with BatchNorm
- Quaternion normalization in forward pass
"""

import torch
import torch.nn as nn

from .backbone import build_backbone


class PoseNetPlus(nn.Module):
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
        _, pooled = self.encoder(x)
        feat = self.fc(pooled)

        xyz = self.fc_xyz(feat)
        q = self.fc_q(feat)
        q = q / (torch.norm(q, dim=1, keepdim=True) + 1e-8)

        return xyz, q

    def get_attention_maps(self, x):
        """Return feature maps for visualization (no attention in this model)."""
        feat_map, _ = self.encoder(x)
        return feat_map.mean(dim=1)
