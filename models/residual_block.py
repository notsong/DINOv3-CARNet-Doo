"""
models/residual_block.py

Residual Decoder Block
"""

import torch
import torch.nn as nn


class ConvGNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    标准Residual Block

    Conv
        ↓
    GN
        ↓
    GELU
        ↓
    Conv
        ↓
    GN
        ↓
      +
      │
    Shortcut
        ↓
    GELU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = ConvGNAct(in_channels, out_channels)

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(8, out_channels)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.GroupNorm(8, out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.GELU()

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)

        out = self.conv2(out)

        out = out + identity

        out = self.act(out)

        return out


class UpResidualBlock(nn.Module):
    """
    Decoder Block

    Upsample
        ↓
    ResidualBlock
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        self.block = ResidualBlock(
            in_channels,
            out_channels
        )

    def forward(self, x):

        x = self.up(x)

        x = self.block(x)

        return x