"""
infer.py

DINOv3 + DecoderV3 极速推理
- Letterbox 无形变 resize（保持宽高比 + 灰条填充）
- 推理后自动去除灰条、还原尺寸
- FP16 推理
- 批量滑窗（单次前向处理所有窗口）
- 大图自动降采样减少窗口数
- 形态学后处理增强连续性
- Boundary Head 输出丢弃，仅用 Seg
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import cv2
import numpy as np
import torch

from config import cfg
from models.dinov3_segmentation import DINOv3Seg


# FP16 预计算常量
MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float16).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float16).view(1, 3, 1, 1)

GRAY = (128, 128, 128)
MARGIN = 16  # 预留边缘，避免 Conv2d zero-padding 伪影


# ==========================================================
# Letterbox 工具
# ==========================================================

def letterbox(image, target_size, color=GRAY):
    """
    无形变 resize：保持宽高比，灰条填充到 target_size×target_size
    预留 MARGIN 边缘，避免 Conv 零填充伪影

    Returns:
        padded: 填充后的图像
        pads:   (top, bottom, left, right)
    """
    h, w = image.shape[:2]
    inner = target_size - 2 * MARGIN
    scale = inner / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=color
    )
    return padded, (pad_top, pad_bottom, pad_left, pad_right)


def unletterbox(prob, pads, output_size):
    """
    去除灰条 + resize 回原始尺寸

    Args:
        prob:       模型输出概率图 [H, W]
        pads:       (top, bottom, left, right)
        output_size: (W, H) 原始尺寸
    """
    pad_top, pad_bottom, pad_left, pad_right = pads
    h, w = prob.shape
    prob_cropped = prob[pad_top:h - pad_bottom, pad_left:w - pad_right]
    prob_resized = cv2.resize(prob_cropped, output_size)
    return prob_resized


# ==========================================================
# 模型加载
# ==========================================================

def load_model(weight_path, device):
    model = DINOv3Seg(cfg).half().to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(f"Loaded: {weight_path} (epoch={ckpt.get('epoch','?')}, "
          f"IoU={ckpt.get('iou',0):.4f}, Dice={ckpt.get('dice',0):.4f})")
    return model


def enhance_connectivity(mask, kernel_size=3):
    if kernel_size <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


# ==========================================================
# 推理
# ==========================================================

def single_infer(model, image, device):
    """单张图直推（≤1024 的小图，letterbox 保持宽高比）"""
    orig_h, orig_w = image.shape[:2]

    # Letterbox → 1024×1024
    inp, pads = letterbox(image, cfg.image_size)
    inp = inp.astype(np.float32) / 255.0
    inp = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).half().to(device)
    inp = (inp - MEAN.to(device)) / STD.to(device)

    with torch.no_grad():
        seg, _ = model(inp)
        prob = torch.softmax(seg, dim=1)[0, 1].float().cpu().numpy()

        # TTA: 水平翻转消除左右不对称
        inp_flip = torch.flip(inp, dims=[3])
        seg_flip, _ = model(inp_flip)
        prob_flip = torch.softmax(seg_flip, dim=1)[0, 1].float().cpu().numpy()
        prob_flip = np.fliplr(prob_flip)
        prob = (prob + prob_flip) / 2.0

    # 去除灰条 + 还原尺寸
    prob = unletterbox(prob, pads, (orig_w, orig_h))
    return prob


def batched_sliding(model, image, device, crop=1024, stride=768):
    """批量滑窗：所有窗口一次前向"""
    h, w = image.shape[:2]
    xs = sorted(set(min(x, w - crop) for x in range(0, w, stride)))
    ys = sorted(set(min(y, h - crop) for y in range(0, h, stride)))

    patches, positions = [], []
    for y1 in ys:
        for x1 in xs:
            patch = image[y1:y1+crop, x1:x1+crop]
            if patch.shape[0] != crop or patch.shape[1] != crop:
                patch = cv2.resize(patch, (crop, crop))
            patches.append(patch.astype(np.float32) / 255.0)
            positions.append((y1, x1))

    batch = torch.from_numpy(np.stack(patches)).permute(0, 3, 1, 2).half().to(device)
    batch = (batch - MEAN.to(device)) / STD.to(device)

    with torch.no_grad():
        seg, _ = model(batch)
        probs = torch.softmax(seg, dim=1)[:, 1].float().cpu().numpy()

        # TTA: 水平翻转消除左右不对称
        batch_flip = torch.flip(batch, dims=[3])
        seg_flip, _ = model(batch_flip)
        probs_flip = torch.softmax(seg_flip, dim=1)[:, 1].float().cpu().numpy()
        probs = (probs + np.fliplr(probs_flip)) / 2.0

    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    for (y1, x1), prob in zip(positions, probs):
        prob_map[y1:y1+crop, x1:x1+crop] += prob
        count_map[y1:y1+crop, x1:x1+crop] += 1

    return prob_map / (count_map + 1e-6)


SCALE = 0.85  # 大图缩放比例（速度/质量平衡）


def infer_single(model, image_path, device):
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # 大图：缩放到 0.85x 再滑窗（窗口数减半，质量几乎无损）
    if max(h, w) > 1200:
        nh, nw = int(h * SCALE), int(w * SCALE)
        small = cv2.resize(image, (nw, nh))
        if max(nh, nw) <= cfg.image_size:
            prob = single_infer(model, small, device)
        else:
            prob = batched_sliding(model, small, device)
        prob = cv2.resize(prob, (w, h))
    elif max(h, w) > cfg.image_size * 1.2:
        prob = batched_sliding(model, image, device)
    else:
        prob = single_infer(model, image, device)

    mask = (prob > cfg.infer_threshold).astype(np.uint8)
    mask = enhance_connectivity(mask, cfg.infer_close_kernel)
    return mask, prob


def save_result(mask, prob, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path + "_mask.png", mask * 255)
    cv2.imwrite(save_path + "_prob.png", (prob * 255).astype(np.uint8))


if __name__ == "__main__":
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.join(cfg.save_dir, "best_dice.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(cfg.save_dir, "best_iou.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(cfg.save_dir, "last.pth")

    model = load_model(ckpt_path, device)
    _ = MEAN.to(device); _ = STD.to(device)  # pin to GPU

    test_dir = cfg.test_image_dir
    if not os.path.exists(test_dir):
        print(f"Test dir not found: {test_dir}")
        exit(1)

    print(f"Image size: {cfg.image_size}, Threshold: {cfg.infer_threshold}, "
          f"Close: {cfg.infer_close_kernel}")

    for name in sorted(os.listdir(test_dir)):
        if not name.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
            continue
        path = os.path.join(test_dir, name)
        print(f"Infer: {name}")
        mask, prob = infer_single(model, path, device)
        save_result(mask, prob,
                    os.path.join(cfg.output_dir, "infer", os.path.splitext(name)[0]))

    print("Done.")
