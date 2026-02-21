"""
Learnable-weighted pose loss (Kendall et al., 2017).
L = L_x * exp(-sx) + sx + L_q * exp(-sq) + sq

sx, sq are learnable log-variance parameters that automatically
balance position and orientation losses during training.
"""

import math
import torch
import torch.nn as nn


class PoseLoss(nn.Module):
    def __init__(self, sx_init: float = 0.0, sq_init: float = -3.0,
                 learn_weights: bool = True):
        super().__init__()
        self.sx = nn.Parameter(torch.tensor(sx_init), requires_grad=learn_weights)
        self.sq = nn.Parameter(torch.tensor(sq_init), requires_grad=learn_weights)

    def forward(self, xyz_pred, xyz_gt, q_pred, q_gt):
        loss_pos = nn.functional.mse_loss(xyz_pred, xyz_gt)
        loss_ori = self._quaternion_loss(q_pred, q_gt)

        loss = loss_pos * torch.exp(-self.sx) + self.sx + \
               loss_ori * torch.exp(-self.sq) + self.sq

        return loss, loss_pos.detach(), loss_ori.detach()

    @staticmethod
    def _quaternion_loss(q_pred, q_gt):
        """1 - |<q_pred, q_gt>|, equivalent to angular error."""
        q_pred = q_pred / (q_pred.norm(dim=1, keepdim=True) + 1e-8)
        q_gt = q_gt / (q_gt.norm(dim=1, keepdim=True) + 1e-8)
        dot = torch.sum(q_pred * q_gt, dim=1).abs()
        return (1.0 - dot).mean()

    def get_effective_weights(self):
        """Return current effective weights for logging."""
        return {
            "w_pos": float(torch.exp(-self.sx).item()),
            "w_ori": float(torch.exp(-self.sq).item()),
            "sx": float(self.sx.item()),
            "sq": float(self.sq.item()),
        }
