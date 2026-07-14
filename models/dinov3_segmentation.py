"""
models/dinov3_segmentation.py

V3版本：
DINOv3 Encoder + DecoderV3
输出：
    seg logits
    boundary logits
"""

import torch
import torch.nn as nn

from models.dinov3_encoder import DINOv3Encoder
from models.decoder_v3 import DecoderV3


class DINOv3Seg(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.encoder = DINOv3Encoder(
            cfg.backbone_name,
            trainable=not cfg.freeze_backbone
        )

        self.decoder = DecoderV3(
            num_classes=cfg.num_classes,
            feat_dim=768
        )

    def forward(self, x):

        feats = self.encoder(x)

        seg, boundary = self.decoder(
            feats,
            output_size=x.shape[2:]
        )

        return seg, boundary