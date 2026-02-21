"""
Shared backbone extractors for pose regression models.
Supports ResNet18/34/50, EfficientNet-B0, and DeiT-Small (ViT).
"""

import torch
import torch.nn as nn
from torchvision import models


def _remove_classifier(module_list):
    """Remove the final FC layer from a list of ResNet children, keep up to avgpool."""
    children = list(module_list)
    return nn.Sequential(*children[:-1])


class ResNetBackbone(nn.Module):
    FEAT_DIMS = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}

    def __init__(self, arch: str = "resnet34", pretrained: bool = True,
                 unfreeze_last_n: int = 2):
        super().__init__()
        assert arch in self.FEAT_DIMS, f"Unsupported arch: {arch}"
        self.feat_dim = self.FEAT_DIMS[arch]

        weights_map = {
            "resnet18": models.ResNet18_Weights.DEFAULT,
            "resnet34": models.ResNet34_Weights.DEFAULT,
            "resnet50": models.ResNet50_Weights.DEFAULT,
        }
        weights = weights_map[arch] if pretrained else None
        resnet = getattr(models, arch)(weights=weights)

        children = list(resnet.children())
        self.backbone = nn.Sequential(*children[:-2])
        self.avgpool = children[-2]

        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last N blocks (layer3, layer4, etc.)
        if unfreeze_last_n > 0:
            blocks = [c for c in self.backbone.children()]
            for block in blocks[-unfreeze_last_n:]:
                for param in block.parameters():
                    param.requires_grad = True

    def forward(self, x):
        feat_map = self.backbone(x)
        pooled = self.avgpool(feat_map)
        return feat_map, pooled.flatten(1)


class EfficientNetBackbone(nn.Module):
    def __init__(self, pretrained: bool = True, unfreeze_last_n: int = 2):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        effnet = models.efficientnet_b0(weights=weights)
        self.features = effnet.features
        self.avgpool = effnet.avgpool
        self.feat_dim = 1280

        for param in self.features.parameters():
            param.requires_grad = False

        if unfreeze_last_n > 0:
            blocks = list(self.features.children())
            for block in blocks[-unfreeze_last_n:]:
                for param in block.parameters():
                    param.requires_grad = True

    def forward(self, x):
        feat_map = self.features(x)
        pooled = self.avgpool(feat_map)
        return feat_map, pooled.flatten(1)


def build_backbone(cfg: dict):
    arch = cfg.get("backbone", "resnet34")
    pretrained = cfg.get("pretrained", True)
    unfreeze = cfg.get("unfreeze_last_n", 2)

    if arch.startswith("resnet"):
        return ResNetBackbone(arch, pretrained, unfreeze)
    elif arch.startswith("efficientnet"):
        return EfficientNetBackbone(pretrained, unfreeze)
    else:
        raise ValueError(f"Unknown backbone: {arch}")
