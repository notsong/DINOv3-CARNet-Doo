"""
utils/visualize.py

可视化工具（训练 / 推理 debug）
"""

import os
import cv2
import numpy as np
import torch


def colorize_mask(mask):
    """
    0=背景
    1=边缘
    """

    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    color[mask == 1] = [255, 255, 255]   # white boundary
    return color


def save_visual(image, mask, save_path):
    """
    image: RGB
    mask: 0/1
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()

    image = (image * 255).astype(np.uint8)

    mask_vis = colorize_mask(mask)

    overlay = cv2.addWeighted(image, 0.7, mask_vis, 0.3, 0)

    cv2.imwrite(save_path + "_img.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(save_path + "_mask.png", mask_vis)
    cv2.imwrite(save_path + "_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))