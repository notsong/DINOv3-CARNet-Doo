"""
models/boundary_head.py

Boundary Head for DINOv3 Decoder V3

用于晶界边缘辅助监督
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryHead(nn.Module):
    """
    输入: decoder feature
    输出: edge probability map [B,1,H,W]
    """

    def __init__(self, in_channels):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.GroupNorm(8, in_channels // 2),
            nn.GELU(),

            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1, bias=False),
            nn.GroupNorm(8, in_channels // 4),
            nn.GELU(),

            nn.Conv2d(in_channels // 4, 1, 1)
        )

    def forward(self, x):
        return self.head(x)


class BoundaryRefineModule(nn.Module):
    """
    可选：轻量边界增强模块
    （推理可不使用，仅训练辅助）
    """

    def __init__(self, channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),

            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels)
        )

    def forward(self, x):
        return x + self.conv(x)