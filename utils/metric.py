"""
utils/metric.py

语义分割评估指标（边缘专用）
"""

import torch
import numpy as np


def iou_score(pred, target, num_classes=2, eps=1e-6):
    """
    pred: [B, H, W] or [B, 1, H, W]
    target: [B, 1, H, W]
    """

    if pred.dim() == 4:
        pred = pred.squeeze(1)

    target = target.squeeze(1)

    ious = []

    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        iou = (intersection + eps) / (union + eps)
        ious.append(iou.item())

    return np.mean(ious)


def dice_score(pred, target, eps=1e-6):
    """
    binary dice
    """

    if pred.dim() == 4:
        pred = pred.squeeze(1)

    target = target.squeeze(1)

    pred = pred.float()
    target = target.float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2 * intersection + eps) / (union + eps)

    return dice.item()


def boundary_f1(pred, target, kernel_size=3):
    """
    简化边界F1（工业近似版本）
    用于边缘连续性评估
    """

    if pred.dim() == 4:
        pred = pred.squeeze(1)

    target = target.squeeze(1)

    # Sobel-like edge extraction
    kernel = torch.tensor([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=torch.float32, device=pred.device).unsqueeze(0).unsqueeze(0)

    pred_edge = torch.nn.functional.conv2d(
        pred.unsqueeze(1).float(), kernel, padding=1
    ).abs() > 0

    target_edge = torch.nn.functional.conv2d(
        target.unsqueeze(1).float(), kernel, padding=1
    ).abs() > 0

    pred_edge = pred_edge.float()
    target_edge = target_edge.float()

    tp = (pred_edge * target_edge).sum()
    fp = (pred_edge * (1 - target_edge)).sum()
    fn = ((1 - pred_edge) * target_edge).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return f1.item()