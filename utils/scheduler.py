"""
utils/scheduler.py

Warmup + Cosine LR Scheduler
适用于 DINOv3 fine-tune
"""

import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch

        if epoch < self.warmup_epochs:
            return [
                base_lr * epoch / max(1, self.warmup_epochs)
                for base_lr in self.base_lrs
            ]

        # cosine decay
        progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)

        return [
            self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]