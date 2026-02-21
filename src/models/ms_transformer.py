"""
MS-Transformer — Multi-Scene Transformer for Absolute Pose Regression.

Based on: "Learning Multi-Scene Absolute Pose Regression with Transformers"
(Shavit et al., ICCV 2021).

Key additions over TransPoseNet:
- Learnable scene embedding token, appended alongside CLS and patch tokens
- Enables the model to learn shared geometric representations across scenes
- Two-stage training: pre-train on multiple scenes, fine-tune on target scene

During fine-tuning on a new scene, only the scene embedding and pose heads
are reset; the backbone + transformer retain pre-trained weights.
"""

import torch
import torch.nn as nn

from .backbone import build_backbone


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, num_tokens: int, d_model: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)

    def forward(self, x):
        return x + self.pos_embed[:, :x.shape[1], :]


class MSTransformer(nn.Module):
    def __init__(self, backbone: str = "resnet34", pretrained: bool = True,
                 unfreeze_last_n: int = 2, dropout: float = 0.1,
                 hidden_dim: int = 512,
                 num_heads: int = 8, num_layers: int = 4,
                 transformer_dim: int = 256,
                 num_scenes: int = 7,
                 **kwargs):
        super().__init__()
        self.num_scenes = num_scenes
        self.transformer_dim = transformer_dim

        self.encoder = build_backbone({
            "backbone": backbone,
            "pretrained": pretrained,
            "unfreeze_last_n": unfreeze_last_n,
        })
        feat_dim = self.encoder.feat_dim

        self.input_proj = nn.Conv2d(feat_dim, transformer_dim, 1)

        max_tokens = 7 * 7 + 2  # +1 CLS, +1 scene token
        self.pos_encoding = LearnedPositionalEncoding(max_tokens, transformer_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim) * 0.02)
        self.scene_embeddings = nn.Embedding(num_scenes, transformer_dim)

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

        tokens = self.input_proj(feat_map)
        tokens = tokens.flatten(2).permute(0, 2, 1)  # (B, H*W, D)

        cls = self.cls_token.expand(B, -1, -1)

        if scene_id is not None:
            if isinstance(scene_id, int):
                scene_id = torch.full((B,), scene_id, dtype=torch.long,
                                      device=x.device)
            scene_tok = self.scene_embeddings(scene_id).unsqueeze(1)  # (B, 1, D)
            tokens = torch.cat([cls, scene_tok, tokens], dim=1)
        else:
            tokens = torch.cat([cls, tokens], dim=1)

        tokens = self.pos_encoding(tokens)
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        cls_out = tokens[:, 0]

        feat = self.fc(cls_out)
        xyz = self.fc_xyz(feat)
        q = self.fc_q(feat)
        q = q / (torch.norm(q, dim=1, keepdim=True) + 1e-8)

        return xyz, q

    def prepare_finetune(self, new_num_scenes: int = 1):
        """
        Prepare model for fine-tuning on a new scene:
        reset scene embeddings and pose heads, keep backbone + transformer.
        """
        self.num_scenes = new_num_scenes
        self.scene_embeddings = nn.Embedding(new_num_scenes, self.transformer_dim)

        nn.init.xavier_uniform_(self.fc_xyz.weight)
        nn.init.zeros_(self.fc_xyz.bias)
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.zeros_(self.fc_q.bias)

        hidden_dim = self.fc[0].in_features
        self.fc = nn.Sequential(
            nn.Linear(self.transformer_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=self.fc[2].p),
        )
        self.fc_xyz = nn.Linear(hidden_dim, 3)
        self.fc_q = nn.Linear(hidden_dim, 4)

    def get_attention_maps(self, x, scene_id=None):
        feat_map, _ = self.encoder(x)
        B, C, H, W = feat_map.shape

        tokens = self.input_proj(feat_map)
        tokens = tokens.flatten(2).permute(0, 2, 1)
        cls = self.cls_token.expand(B, -1, -1)

        offset = 1
        if scene_id is not None:
            if isinstance(scene_id, int):
                scene_id = torch.full((B,), scene_id, dtype=torch.long,
                                      device=x.device)
            scene_tok = self.scene_embeddings(scene_id).unsqueeze(1)
            tokens = torch.cat([cls, scene_tok, tokens], dim=1)
            offset = 2
        else:
            tokens = torch.cat([cls, tokens], dim=1)

        tokens = self.pos_encoding(tokens)

        for layer in self.transformer.layers[:-1]:
            tokens = layer(tokens)

        last_layer = self.transformer.layers[-1]
        normed = last_layer.norm1(tokens)
        _, attn_weights = last_layer.self_attn(
            normed, normed, normed, need_weights=True, average_attn_weights=True
        )

        cls_attn = attn_weights[:, 0, offset:]
        return cls_attn.view(B, H, W)
