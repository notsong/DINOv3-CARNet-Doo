"""
train.py
DINOv3 + DecoderV3 — 两阶段训练 + Boundary Head + 日志系统
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json, dataclasses
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import cfg
from data.dataset_loader import SegDataset
from data.transforms import get_train_transform, get_val_transform
from models.dinov3_segmentation import DINOv3Seg
from losses.loss import TotalLoss
from utils.metric import iou_score, dice_score
from utils.seed import set_seed
from utils.scheduler import WarmupCosineScheduler
from utils.logger import TrainLogger


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc="Train")

    for images, masks, _ in loop:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=cfg.amp):
            seg, boundary = model(images)

        loss = criterion(seg, boundary, masks)

        scaler.scale(loss).backward()

        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    iou_total = 0
    dice_total = 0

    for images, masks, _ in tqdm(loader, desc="Val"):
        images = images.to(device)
        masks = masks.to(device).long()

        seg, boundary = model(images)
        loss = criterion(seg, boundary, masks)

        preds = torch.argmax(seg, dim=1, keepdim=True)

        total_loss += loss.item()
        iou_total += iou_score(preds.cpu(), masks.cpu())
        dice_total += dice_score(preds.cpu(), masks.cpu())

    n = len(loader)
    return total_loss / n, iou_total / n, dice_total / n


def main():
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Backbone: {cfg.backbone_name}")
    print(f"Config: image_size={cfg.image_size}, batch={cfg.batch_size}, "
          f"epochs={cfg.epochs}, lr={cfg.learning_rate}")
    print(f"Loss: bce={cfg.bce_weight}, dice={cfg.dice_weight}, "
          f"boundary={cfg.boundary_weight}")
    print(f"Stage1: freeze {cfg.freeze_epochs} epochs → Stage2: unfreeze lr={cfg.unfreeze_lr}")

    # ======================
    # Logger
    # ======================
    logger = TrainLogger(cfg.log_dir)
    logger.save_config(dataclasses.asdict(cfg))

    # ======================
    # Dataset
    # ======================
    train_dataset = SegDataset(
        cfg.train_image_dir, cfg.train_mask_dir,
        transform=get_train_transform(), is_train=True,
        image_size=cfg.image_size
    )
    val_dataset = SegDataset(
        cfg.val_image_dir, cfg.val_mask_dir,
        transform=get_val_transform(), is_train=False,
        image_size=cfg.image_size
    )
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    # ======================
    # Model
    # ======================
    model = DINOv3Seg(cfg).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")

    # ======================
    # Loss, Optimizer, Scheduler
    # ======================
    criterion = TotalLoss(cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=cfg.warmup_epochs,
        total_epochs=cfg.epochs, min_lr=cfg.min_lr
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_iou = 0.0
    best_dice = 0.0
    start_epoch = 0
    stage2_activated = False

    # ======================
    # Resume
    # ======================
    if cfg.resume and os.path.exists(cfg.resume):
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_iou = ckpt.get("iou", 0.0)
        best_dice = ckpt.get("dice", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    os.makedirs(cfg.save_dir, exist_ok=True)

    # ======================
    # Training Loop
    # ======================
    for epoch in range(start_epoch, cfg.epochs):
        current_lr = optimizer.param_groups[0]['lr']

        # ---- 阶段2切换：解冻backbone ----
        if epoch == cfg.freeze_epochs and not stage2_activated:
            stage2_activated = True
            print(f"\n{'='*50}")
            print(f"Stage 2: Unfreezing backbone, lr={cfg.unfreeze_lr}")
            print(f"{'='*50}")

            model.encoder.set_trainable(True)

            # 重建optimizer（更低LR）
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.unfreeze_lr, weight_decay=cfg.weight_decay
            )
            # 重建scheduler（剩余epoch用cosine）
            remaining = cfg.epochs - epoch
            scheduler = WarmupCosineScheduler(
                optimizer, warmup_epochs=0,
                total_epochs=remaining, min_lr=cfg.min_lr
            )

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Params: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")

        print(f"\n{'='*50}")
        print(f"Epoch [{epoch+1}/{cfg.epochs}]  LR: {current_lr:.2e}"
              f"  {'[Stage2 finetune]' if stage2_activated else '[Stage1 frozen]'}")
        print(f"{'='*50}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )
        val_loss, val_iou, val_dice = validate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f}")
        print(f"Val   IoU : {val_iou:.4f}")
        print(f"Val   Dice: {val_dice:.4f}")

        # ---- log ----
        logger.log_epoch(epoch, train_loss, val_loss, val_iou, val_dice, current_lr)

        # ---- save best IoU ----
        if val_iou > best_iou and cfg.save_best_iou:
            best_iou = val_iou
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "iou": val_iou, "dice": val_dice},
                os.path.join(cfg.save_dir, "best_iou.pth")
            )
            print(f"  -> Saved best_iou.pth (IoU={best_iou:.4f})")

        # ---- save best Dice ----
        if val_dice > best_dice and cfg.save_best_dice:
            best_dice = val_dice
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "iou": val_iou, "dice": val_dice},
                os.path.join(cfg.save_dir, "best_dice.pth")
            )
            print(f"  -> Saved best_dice.pth (Dice={best_dice:.4f})")

        # ---- save latest ----
        save_checkpoint(
            {"epoch": epoch, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(),
             "iou": val_iou, "dice": val_dice},
            os.path.join(cfg.save_dir, "last.pth")
        )

    # ======================
    # Plot training curves
    # ======================
    logger.plot()
    print(f"\nTraining done! Best IoU: {best_iou:.4f}, Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
