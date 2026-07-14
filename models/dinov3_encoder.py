"""
models/dinov3_encoder.py

DINOv3 Multi-layer Feature Extractor
输出多尺度 feature maps（取最后3层hidden states）
"""
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


class DINOv3Encoder(nn.Module):
    def __init__(self, model_name: str, trainable=False):
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(
            model_name, local_files_only=True
        )

        self.backbone = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float32,
            trust_remote_code=True,
            local_files_only=True
        )

        self.trainable = trainable
        if not trainable:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def set_trainable(self, trainable: bool):
        """动态切换backbone是否可训练"""
        self.trainable = trainable
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        """
        Returns:
            feats = {"f4": low-level, "f8": mid-level, "f16": high-level}
            所有特征空间分辨率相同（ViT patch=14, image_size/14）
        """
        outputs = self.backbone(
            pixel_values=x,
            output_hidden_states=True
        )

        hs = outputs.hidden_states

        # 取最后3层：low, mid, high
        f_low = hs[-3]
        f_mid = hs[-2]
        f_high = hs[-1]

        def reshape(feat):
            B, N, C = feat.shape
            # DINOv3: [CLS] + patches + 4 register tokens
            # 取 patches 部分（去掉 CLS 和 register tokens）
            feat = feat[:, 1:-4, :]  # drop CLS (idx 0) and 4 register tokens (last 4)
            num_patches = feat.shape[1]
            H = W = int(num_patches ** 0.5)
            feat = feat.reshape(B, H, W, C).permute(0, 3, 1, 2)
            return feat.contiguous()

        return {
            "f4": reshape(f_low),
            "f8": reshape(f_mid),
            "f16": reshape(f_high),
        }
