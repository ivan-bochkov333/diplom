"""
Evaluation metrics for 6DoF camera pose estimation.
"""

import math
import numpy as np
import torch


def position_error(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> float:
    """Euclidean distance between predicted and ground-truth positions."""
    return float(np.linalg.norm(pred_xyz - gt_xyz))


def rotation_error_deg(pred_q: np.ndarray, gt_q: np.ndarray) -> float:
    """
    Angular error in degrees between two unit quaternions.
    Uses the formula: angle = 2 * arccos(|<q1, q2>|)
    """
    pred_q = pred_q / (np.linalg.norm(pred_q) + 1e-8)
    gt_q = gt_q / (np.linalg.norm(gt_q) + 1e-8)
    dot = float(np.clip(np.abs(np.dot(pred_q, gt_q)), 0.0, 1.0))
    angle = 2.0 * math.acos(dot)
    return math.degrees(angle)


def batch_rotation_error_deg(pred_q: torch.Tensor, gt_q: torch.Tensor) -> torch.Tensor:
    """Batch angular error in degrees. Input shape: (B, 4)."""
    pred_q = pred_q / (pred_q.norm(dim=1, keepdim=True) + 1e-8)
    gt_q = gt_q / (gt_q.norm(dim=1, keepdim=True) + 1e-8)
    dot = torch.sum(pred_q * gt_q, dim=1).abs().clamp(0.0, 1.0)
    angles = 2.0 * torch.acos(dot)
    return angles * (180.0 / math.pi)


def compute_metrics(pred_xyz_list, pred_q_list, gt_xyz_list, gt_q_list):
    """
    Compute median and mean position/rotation errors.
    All inputs are lists of numpy arrays of shape (3,) and (4,).
    Returns dict with median_pos, mean_pos, median_rot, mean_rot.
    """
    pos_errors = []
    rot_errors = []

    for p_xyz, g_xyz, p_q, g_q in zip(pred_xyz_list, gt_xyz_list, pred_q_list, gt_q_list):
        pos_errors.append(position_error(p_xyz, g_xyz))
        rot_errors.append(rotation_error_deg(p_q, g_q))

    pos_errors = np.array(pos_errors)
    rot_errors = np.array(rot_errors)

    return {
        "median_pos": float(np.median(pos_errors)),
        "mean_pos": float(np.mean(pos_errors)),
        "median_rot": float(np.median(rot_errors)),
        "mean_rot": float(np.mean(rot_errors)),
        "pos_errors": pos_errors,
        "rot_errors": rot_errors,
    }
