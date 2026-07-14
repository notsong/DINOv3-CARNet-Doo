"""
export_onnx.py

DINOv3 + DecoderV3 ONNX导出（工业部署用）
"""
import os, sys
import numpy as np

# monkey-patch: NumPy 2.x checkpoint → NumPy 1.x 环境兼容
sys.modules.setdefault("numpy._core", np)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)

import torch

from config import cfg
from models.dinov3_segmentation import DINOv3Seg


def export_onnx(weight_path, onnx_path, device):
    model = DINOv3Seg(cfg).to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(f"Loaded: {weight_path} (epoch={ckpt.get('epoch','?')})")

    dummy_input = torch.randn(1, 3, cfg.image_size, cfg.image_size, device=device)

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    # V3: 双输出 — seg + boundary
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["seg", "boundary"],
        dynamic_axes={
            "input":    {0: "batch_size", 2: "height", 3: "width"},
            "seg":      {0: "batch_size", 2: "height", 3: "width"},
            "boundary": {0: "batch_size", 2: "height", 3: "width"},
        },
    )

    print(f"ONNX exported to: {onnx_path}")


if __name__ == "__main__":
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # 使用 best_iou.pth
    ckpt = os.path.join(cfg.save_dir, "best_iou.pth")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(cfg.save_dir, "best_dice.pth")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(cfg.save_dir, "last.pth")

    export_onnx(weight_path=ckpt, onnx_path=cfg.onnx_path, device=device)
