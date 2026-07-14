"""
datasets/grain_dataset.py

金相晶界数据集
适配 JLD 数据集：images=.jpg, masks=.png, mask像素0/1
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from data.transforms import letterbox_resize


class GrainDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None, is_train=True,
                 image_size=1024):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.is_train = is_train
        self.image_size = image_size

        self.image_list = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))
        ])

        self.has_mask = mask_dir is not None

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = None

        if self.has_mask:
            # mask 统一为 .png，和原图后缀可能不同
            base_name = os.path.splitext(img_name)[0]
            mask_name = base_name + ".png"
            mask_path = os.path.join(self.mask_dir, mask_name)

            if not os.path.exists(mask_path):
                # fallback: 尝试同后缀
                mask_path = os.path.join(self.mask_dir, img_name)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Mask not found or unreadable: {mask_path}")

            # mask 像素值是 0/1，不需要 >127 二值化
            mask = mask.astype(np.uint8)

        # ── Letterbox：无形变 resize + 灰条填充 ──
        if self.has_mask:
            image, mask = letterbox_resize(image, mask, self.image_size)
        else:
            h, w = image.shape[:2]
            scale = self.image_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
            pad_h = self.image_size - new_h
            pad_w = self.image_size - new_w
            image = cv2.copyMakeBorder(
                image, pad_h // 2, pad_h - pad_h // 2,
                pad_w // 2, pad_w - pad_w // 2,
                cv2.BORDER_CONSTANT, value=(128, 128, 128)
            )

        if self.transform:
            if self.has_mask:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]

        if self.has_mask:
            # mask → [1,H,W] float32（ToTensorV2可能已转tensor）
            if isinstance(mask, torch.Tensor):
                mask = mask.unsqueeze(0).float()
            else:
                mask = torch.from_numpy(mask).unsqueeze(0).float()
            return image, mask, img_name

        return image, img_name


class UnlabeledDataset(Dataset):
    """未标注数据（pseudo label / self-supervised）"""

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        self.image_list = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, img_name
