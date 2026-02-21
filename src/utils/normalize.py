"""
Coordinate normalization for multi-scene compatibility.
Centers positions and scales to unit variance per scene.
"""

import numpy as np


def compute_normalization_params(positions: np.ndarray):
    """
    Compute center and scale from an array of positions (N, 3).
    Returns (center, scale) where center is (3,) and scale is scalar.
    """
    center = positions.mean(axis=0)
    centered = positions - center
    scale = np.sqrt((centered ** 2).sum(axis=1).mean()) + 1e-8
    return center, float(scale)


def normalize_poses(positions: np.ndarray, center: np.ndarray, scale: float):
    """Normalize positions by centering and scaling."""
    return (positions - center) / scale


def denormalize_poses(positions: np.ndarray, center: np.ndarray, scale: float):
    """Inverse of normalize_poses."""
    return positions * scale + center
