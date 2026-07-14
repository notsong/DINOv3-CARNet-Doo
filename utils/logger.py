"""
utils/logger.py

训练日志系统：保存配置JSON、每epoch指标CSV、自动绘制折线图
"""
import os, json, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TrainLogger:
    def __init__(self, log_dir="./logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "metrics.csv")
        self.config_path = os.path.join(log_dir, "config.json")
        self._csv_header_written = False
        self.records = []

    def save_config(self, cfg_dict):
        with open(self.config_path, "w") as f:
            json.dump(cfg_dict, f, indent=2, ensure_ascii=False)
        print(f"[Logger] Config saved to {self.config_path}")

    def log_epoch(self, epoch, train_loss, val_loss, val_iou, val_dice, lr):
        record = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_iou": round(val_iou, 6),
            "val_dice": round(val_dice, 6),
            "lr": lr,
        }
        self.records.append(record)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(record)

    def plot(self):
        if not self.records:
            return

        epochs = [r["epoch"] for r in self.records]
        train_loss = [r["train_loss"] for r in self.records]
        val_loss = [r["val_loss"] for r in self.records]
        val_iou = [r["val_iou"] for r in self.records]
        val_dice = [r["val_dice"] for r in self.records]
        lr = [r["lr"] for r in self.records]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss
        axes[0, 0].plot(epochs, train_loss, label="Train Loss", color="#1f77b4")
        axes[0, 0].plot(epochs, val_loss, label="Val Loss", color="#ff7f0e")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training & Validation Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # IoU + Dice
        axes[0, 1].plot(epochs, val_iou, label="Val IoU", color="#2ca02c")
        axes[0, 1].plot(epochs, val_dice, label="Val Dice", color="#d62728")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Score")
        axes[0, 1].set_title("Validation IoU & Dice")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # LR
        axes[1, 0].plot(epochs, lr, color="#9467bd")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Learning Rate")
        axes[1, 0].set_title("Learning Rate Schedule")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

        # Loss + IoU combined (双轴)
        ax1 = axes[1, 1]
        ax1.plot(epochs, val_loss, label="Val Loss", color="#ff7f0e")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Val Loss", color="#ff7f0e")
        ax1.tick_params(axis="y", labelcolor="#ff7f0e")

        ax2 = ax1.twinx()
        ax2.plot(epochs, val_dice, label="Val Dice", color="#d62728")
        ax2.set_ylabel("Val Dice", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")

        ax1.set_title("Loss vs Dice")
        ax1.grid(True, alpha=0.3)

        fig.tight_layout()
        save_path = os.path.join(self.log_dir, "training_curves.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Logger] Curves saved to {save_path}")
