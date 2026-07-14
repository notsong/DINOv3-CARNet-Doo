# CLAUDE.md — DINOv3 + FPN-UNet 金相晶界语义分割

## 项目概述

基于 DINOv3（ViT-B/16）作为特征提取骨干，结合 FPN + UNet 混合解码器的工业级晶界分割方案。专为金相显微图像的晶界（grain boundary）检测设计。

## 技术栈

- **Python 3.10+** | **PyTorch 2.x** | **CUDA** (AMP 混合精度)
- **HuggingFace Transformers** — 加载 DINOv3 预训练权重
- **Albumentations** — 数据增强
- **OpenCV** — 图像读写、形态学后处理
- **Matplotlib** — 训练曲线绘制
- conda 环境: `D:/work/anaconda/envs/unet_pro/python`

## 项目结构

```
├── config.py                  # 全局配置 dataclass（唯一配置入口）
├── train.py                   # 训练入口（两阶段：冻结→解冻微调）
├── infer.py                   # 推理入口（FP16 + 批量滑窗 + 形态学后处理）
├── export_onnx.py             # ONNX 导出
├── data/
│   ├── grain_dataset.py       # GrainDataset（有标注） + UnlabeledDataset
│   └── transforms.py          # Albumentations 训练/验证增强
├── models/
│   ├── dinov3_encoder.py      # DINOv3 多层特征提取（f4/f8/f16）
│   ├── fpn_unet_decoder.py    # FPN 投影 + 4级×2上采样解码器
│   └── dinov3_segmentation.py # 顶层模型 = Encoder + Decoder
├── losses/
│   └── loss.py                # TotalLoss = CE + Dice + Boundary
├── utils/
│   ├── metric.py              # IoU / Dice / Boundary F1
│   ├── scheduler.py           # Warmup + Cosine 学习率调度
│   ├── logger.py              # CSV 日志 + 训练曲线 PNG
│   ├── visualize.py           # 可视化工具
│   └── seed.py                # 随机种子固定
├── checkpoints/               # 模型权重 (best_iou.pth / best_dice.pth / last.pth)
├── logs/                      # config.json + metrics.csv + training_curves.png
├── dataset/                   # 数据集（按 README 结构组织）
└── output/                    # 推理输出
```

## 常用命令

```bash
# 激活环境并训练
D:/work/anaconda/envs/unet_pro/python train.py

# 推理
D:/work/anaconda/envs/unet_pro/python infer.py

# ONNX 导出
D:/work/anaconda/envs/unet_pro/python export_onnx.py

# 快速检查导入
D:/work/anaconda/envs/unet_pro/python -c "from config import cfg; print(cfg)"
```

## 架构关键细节

### Encoder: DINOv3 特征提取
- **Backbone**: `facebook/dinov3-vitb16-pretrain-lvd1689m`
- **patch_size=14**, 输入尺寸 `image_size=784`（56×14，保证整除）
- 取最后 3 层 hidden_states，reshape 时去掉 CLS token (idx 0) 和 4 个 register tokens (last 4)
- 所有层级输出相同空间分辨率 `[B, 768, 56, 56]`（ViT 不像 CNN 有下采样金字塔）

### Decoder: FPN + UNet 混合
- 3 层特征通过 1×1 卷积统一到 256 通道 → **同分辨率相加融合**（非 concat）
- 融合后逐级 2× 上采样: 56→112→224→448→896
- 每级一个 `ConvBlock`（Conv→BN→ReLU→Conv→BN→ReLU）
- 最后 `F.interpolate` 精确到 `output_size`
- 输出头: 1×1 Conv → 2 通道 logits

### 两阶段训练
- **Stage 1 (epoch 0-59)**: 冻结 backbone，仅训练 decoder，lr=1e-4
- **Stage 2 (epoch 60-99)**: 解冻 backbone，全模型微调，lr=1e-5
- Stage 2 切换时重建 optimizer + scheduler（剩余 epochs 走 cosine decay）

### 损失函数
- `TotalLoss = 0.55×CE + 0.30×Dice + 0.15×Boundary`（实际代码中的权重）
- 注意：`config.py` 中的权重参数 (`bce_weight`, `dice_weight`, `boundary_weight`, `connectivity_weight`) 定义但在 `loss.py` 中**未使用** — loss.py 有自己硬编码的权重
- Boundary Loss 用 Laplacian 核 `[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]` 提取边缘后计算 L1

### 推理优化
- **FP16** 全流程（模型 + 归一化常量都用 half）
- **批量滑窗**: 所有窗口一次前向传播（非逐窗口循环）
- 大图自动降采样 0.85× 以减少窗口数
- 形态学闭运算后处理 (`cv2.MORPH_CLOSE`) 增强晶界连续性

## 编码约定

- 配置**只能**通过 `config.py` 的 `Config` dataclass 修改，不散落硬编码
- 模型通过 `cfg` 对象参数化，不直接 import 模块级常量
- 新增 loss 权重需同时更新 `config.py`（定义参数）和 `loss.py`（实际使用）
- 所有 tensor 操作优先使用 `torch.cuda.amp.autocast` 兼容混合精度
- 随机种子通过 `utils/seed.py` 统一设置
- 数据集 mask 统一为 `.png` 格式，像素值 0/1（非 0/255）

## 已知问题 / 待办

1. **config.py 与 loss.py 的权重不一致**: config 定义了 `bce_weight=0.25`, `dice_weight=0.4`, `boundary_weight=0.2`, `connectivity_weight=0.15`，但 `TotalLoss.__init__()` 硬编码了 `ce_weight=0.55, dice_weight=0.30, boundary_weight=0.15`。`TotalLoss` 构造函数未接收 `cfg` 参数，`connectivity_weight` 未在 loss 中实现。修改 loss 时需同步两处。
2. `batch_size=4` + `image_size=784` + ViT-B 对显存要求较高（建议 ≥12GB），AMP 已开启。
3. 权重文件较大（DINOv3 backbone ≈ 350MB），不纳入版本控制。
