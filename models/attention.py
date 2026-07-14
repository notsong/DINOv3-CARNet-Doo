"""
models/attention.py

Attention Modules
------------------------------------
包含：

1. SEBlock
2. SpatialAttention
3. CBAM
4. AttentionFusion

用于 DINOv3 Decoder V3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# SE Block
# ==========================================================

class SEBlock(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden = max(channels // reduction, 8)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        w = self.pool(x)
        w = self.fc(w)

        return x * w


# ==========================================================
# Spatial Attention
# ==========================================================

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg = torch.mean(x, dim=1, keepdim=True)

        mx, _ = torch.max(x, dim=1, keepdim=True)

        w = torch.cat([avg, mx], dim=1)

        w = self.conv(w)

        w = self.sigmoid(w)

        return x * w


# ==========================================================
# CBAM
# ==========================================================

class CBAM(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.channel = SEBlock(channels)

        self.spatial = SpatialAttention()

    def forward(self, x):

        x = self.channel(x)

        x = self.spatial(x)

        return x


# ==========================================================
# Learnable Weighted Fusion
# ==========================================================

class AttentionFusion(nn.Module):
    """
    输入:
        f4
        f8
        f16

    输出:
        fused feature

    自动学习三个尺度的重要性
    """

    def __init__(self, channels):

        super().__init__()

        self.cbam4 = CBAM(channels)
        self.cbam8 = CBAM(channels)
        self.cbam16 = CBAM(channels)

        self.weight = nn.Parameter(torch.ones(3))

        self.fuse = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                bias=False
            ),

            nn.GroupNorm(8, channels),

            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                bias=False
            ),

            nn.GroupNorm(8, channels),

            nn.GELU()
        )

    def forward(self, f4, f8, f16):

        f4 = self.cbam4(f4)

        f8 = self.cbam8(f8)

        f16 = self.cbam16(f16)

        w = F.softmax(self.weight, dim=0)

        out = (
            w[0] * f4 +
            w[1] * f8 +
            w[2] * f16
        )

        out = self.fuse(out)

        return out