"""
data/transforms.py

数据增强（Albumentations）
针对晶界分割优化：几何/颜色增强 → Normalize
（Letterbox 在 Dataset 中完成，transforms 仅做增强）
"""
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

GRAY = (128, 128, 128)


# ==========================================================
# Letterbox（在 Dataset 中调用，transforms 之前）
# ==========================================================

def letterbox_resize(image, mask, target_size):
    """
    无形变 resize：保持宽高比，灰条填充到 target_size×target_size。

    Args:
        image:      原图 [H,W,3] uint8
        mask:       标注 [H,W] uint8
        target_size: 目标尺寸

    Returns:
        image: [target_size, target_size, 3]
        mask:  [target_size, target_size]
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    image = cv2.resize(image, (new_w, new_h))
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    image = cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=GRAY
    )
    mask = cv2.copyMakeBorder(
        mask, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=0
    )

    return image, mask


# ==========================================================
# 增强 transforms（输入已是 target_size）
# ==========================================================

def get_train_transform():
    """训练增强：几何 + 颜色增强 → Normalize（不包含 resize）"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        A.RandomRotate90(p=0.5),

        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.90, 1.10),
            rotate=(-15, 15),
            border_mode=0,
            p=0.5
        ),

        # 弹性变形模拟金相图像畸变
        A.ElasticTransform(
            alpha=1,
            sigma=20,
            border_mode=0,
            p=0.2
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5
        ),

        A.GaussNoise(
            std_range=(0.01, 0.05),
            p=0.3
        ),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),

        ToTensorV2()
    ])


def get_val_transform():
    """验证集：仅 Normalize（不包含 resize）"""
    return A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),

        ToTensorV2()
    ])


def get_test_transform():
    """测试集：同验证集"""
    return A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),

        ToTensorV2()
    ])
