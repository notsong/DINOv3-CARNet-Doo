"""
models/decoder_v3.py

DINOv3 + Attention Fusion + Residual Decoder + Boundary Head

核心结构：
1. Attention Fusion (f4, f8, f16)
2. Residual Decoder (multi-stage upsampling)
3. Boundary Head (aux supervision)
4. Segmentation Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import AttentionFusion
from models.residual_block import ResidualBlock, UpResidualBlock
from models.boundary_head import BoundaryHead


class DecoderV3(nn.Module):

    def __init__(self, num_classes, feat_dim=768):

        super().__init__()

        # =========================
        # 1. Fusion
        # =========================
        self.fusion = AttentionFusion(feat_dim)

        # =========================
        # 2. Channel Projection
        # =========================
        self.proj = nn.Sequential(
            nn.Conv2d(feat_dim, 512, 1, bias=False),
            nn.GroupNorm(8, 512),
            nn.GELU()
        )

        # =========================
        # 3. Residual Decoder
        # =========================
        self.dec1 = UpResidualBlock(512, 256)
        self.dec2 = UpResidualBlock(256, 128)
        self.dec3 = UpResidualBlock(128, 64)
        self.dec4 = UpResidualBlock(64, 32)

        # =========================
        # 4. Segmentation Head
        # =========================
        self.seg_head = nn.Conv2d(32, num_classes, 1)

        # =========================
        # 5. Boundary Head
        # =========================
        self.boundary_head = BoundaryHead(32)

    def forward(self, feats, output_size=None):

        """
        feats:
            dict:
                f4
                f8
                f16
        """

        f4 = feats["f4"]
        f8 = feats["f8"]
        f16 = feats["f16"]

        # =========================
        # Fusion
        # =========================
        x = self.fusion(f4, f8, f16)

        # =========================
        # Projection
        # =========================
        x = self.proj(x)

        # =========================
        # Decoder
        # =========================
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)

        # =========================
        # Heads
        # =========================
        seg = self.seg_head(x)
        boundary = self.boundary_head(x)

        if output_size is not None:
            seg = F.interpolate(seg, size=output_size, mode="bilinear", align_corners=False)
            boundary = F.interpolate(boundary, size=output_size, mode="bilinear", align_corners=False)

        return seg, boundary