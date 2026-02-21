"""
TransPoseNet — Transformer-based Absolute Pose Regression.

Architecture:
1. CNN backbone extracts spatial feature map (B, C, H, W)
2. Feature map is flattened into patch tokens (B, N, C) with positional encoding
3. A learnable [CLS] token is prepended
4. Transformer Encoder processes all tokens
5. [CLS] token output is used for pose regression via two FC heads

This allows the model to capture global spatial dependencies across the
entire image, which is critical for accurate pose estimation.
"""

import math
import torch
import torch.nn as nn

from .backbone import build_backbone


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, num_tokens: int, d_model: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)

    def forward(self, x):
        return x + self.pos_embed[:, :x.shape[1], :]


class TransPoseNet(nn.Module):
    def __init__(self, backbone: str = "resnet34", pretrained: bool = True,
                 unfreeze_last_n: int = 2, dropout: float = 0.1,
                 hidden_dim: int = 512,
                 num_heads: int = 8, num_layers: int = 4,
                 transformer_dim: int = 256,
                 **kwargs):
        super().__init__()

        self.encoder = build_backbone({
            "backbone": backbone,
            "pretrained": pretrained,
            "unfreeze_last_n": unfreeze_last_n,
        })
        feat_dim = self.encoder.feat_dim
        self.transformer_dim = transformer_dim

        self.input_proj = nn.Conv2d(feat_dim, transformer_dim, 1)

        # Max expected spatial positions (7x7 for 224px input with ResNet)
        max_tokens = 7 * 7 + 1
        self.pos_encoding = LearnedPositionalEncoding(max_tokens, transformer_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(transformer_dim)

        self.fc = nn.Sequential(
            nn.Linear(transformer_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

        self.fc_xyz = nn.Linear(hidden_dim, 3)
        self.fc_q = nn.Linear(hidden_dim, 4)

        nn.init.xavier_uniform_(self.fc_xyz.weight)
        nn.init.xavier_uniform_(self.fc_q.weight)

    def forward(self, x, scene_id=None):
        feat_map, _ = self.encoder(x)
        B, C, H, W = feat_map.shape

        tokens = self.input_proj(feat_map)              # (B, transformer_dim, H, W)
        tokens = tokens.flatten(2).permute(0, 2, 1)     # (B, H*W, transformer_dim)

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)         # (B, 1+H*W, transformer_dim)
        tokens = self.pos_encoding(tokens)

        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        cls_out = tokens[:, 0]                            # (B, transformer_dim)

        feat = self.fc(cls_out)
        xyz = self.fc_xyz(feat)
        q = self.fc_q(feat)
        q = q / (torch.norm(q, dim=1, keepdim=True) + 1e-8)

        return xyz, q

    def get_attention_maps(self, x):
        """
        Extract attention weights from the last transformer layer.
        Returns attention from CLS token to spatial tokens, reshaped to (B, H, W).
        """
        feat_map, _ = self.encoder(x)
        B, C, H, W = feat_map.shape

        tokens = self.input_proj(feat_map)
        tokens = tokens.flatten(2).permute(0, 2, 1)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_encoding(tokens)

        # Run through all but last layer
        for layer in self.transformer.layers[:-1]:
            tokens = layer(tokens)

        # Get attention from last layer manually
        last_layer = self.transformer.layers[-1]
        normed = last_layer.norm1(tokens)
        attn_out, attn_weights = last_layer.self_attn(
            normed, normed, normed, need_weights=True, average_attn_weights=True
        )

        # CLS token attention to spatial tokens
        cls_attn = attn_weights[:, 0, 1:]  # (B, H*W)
        return cls_attn.view(B, H, W)
