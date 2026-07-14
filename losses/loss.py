"""
losses/loss.py

V3 Loss — SegLoss + BoundaryLoss

SegLoss  = bce_weight × CE + dice_weight × Dice
BoundaryLoss = BCE（Boundary Head 输出 vs GT 边缘）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# Dice Loss
# ==========================================================

class DiceLoss(nn.Module):
    """Multi-Class Dice Loss"""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        """
        logits: [B, C, H, W]
        target: [B, 1, H, W] long
        """
        num_classes = logits.shape[1]
        target = target.squeeze(1).long()

        pred = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(pred * target_onehot, dims)
        union = torch.sum(pred, dims) + torch.sum(target_onehot, dims)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


# ==========================================================
# Boundary Loss
# ==========================================================

class BoundaryLoss(nn.Module):
    """
    Boundary Head 辅助损失

    用 Laplacian 从 GT mask 提取边缘 →
    与 Boundary Head 输出做 BCEWithLogitsLoss
    """

    def __init__(self):
        super().__init__()

        kernel = torch.tensor(
            [[[-1, -1, -1],
              [-1,  8, -1],
              [-1, -1, -1]]],
            dtype=torch.float32
        )
        self.register_buffer("laplacian", kernel.unsqueeze(0))

    def forward(self, boundary_logits, target):
        """
        boundary_logits: [B, 1, H, W] — Boundary Head 输出
        target:          [B, 1, H, W] — GT mask（值 0/1）
        """
        # 强制 FP32 + 对齐设备
        gt = target.to(dtype=torch.float32)

        # 提取 GT 边缘
        gt_edge = F.conv2d(gt, self.laplacian.to(device=gt.device), padding=1).abs()
        gt_edge = (gt_edge > 0).to(dtype=torch.float32)

        # 对齐尺寸
        if boundary_logits.shape[2:] != gt_edge.shape[2:]:
            gt_edge = F.interpolate(
                gt_edge,
                size=boundary_logits.shape[2:],
                mode="nearest"
            )

        return F.binary_cross_entropy_with_logits(
            boundary_logits.to(dtype=torch.float32), gt_edge
        )


# ==========================================================
# Total Loss
# ==========================================================

class TotalLoss(nn.Module):
    """
    TotalLoss = SegLoss + boundary_weight × BoundaryLoss
    """

    def __init__(self, cfg):
        super().__init__()

        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()

        self.bce_weight = cfg.bce_weight
        self.dice_weight = cfg.dice_weight
        self.boundary_weight = cfg.boundary_weight

    def forward(self, seg, boundary, target):
        """
        seg:      [B, num_classes, H, W]
        boundary: [B, 1, H, W]
        target:   [B, 1, H, W] long
        """
        target_squeezed = target.squeeze(1).long()

        # SegLoss = CE + Dice
        ce_loss = self.ce(seg, target_squeezed)
        dice_loss = self.dice(seg, target)
        seg_loss = self.bce_weight * ce_loss + self.dice_weight * dice_loss

        # BoundaryLoss
        boundary_loss = self.boundary(boundary, target)

        loss = seg_loss + self.boundary_weight * boundary_loss

        return loss
