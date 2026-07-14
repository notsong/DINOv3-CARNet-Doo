# DINOv3 + FPN-UNet：金相晶界语义分割

基于 **DINOv3（ViT-B/16）** 作为特征提取骨干，结合 **FPN + UNet 混合解码器** 的工业级晶界分割方案。专为金相显微图像的晶界（grain boundary）检测设计，支持训练、推理、ONNX 导出全流程。

---

## 目录

- [项目背景](#项目背景)
- [模型架构](#模型架构)
- [目录结构](#目录结构)
- [环境依赖](#环境依赖)
- [快速开始](#快速开始)
  - [1. 准备数据集](#1-准备数据集)
  - [2. 修改配置](#2-修改配置)
  - [3. 训练](#3-训练)
  - [4. 推理](#4-推理)
  - [5. ONNX 导出](#5-onnx-导出)
- [核心设计](#核心设计)
  - [Encoder：DINOv3 多层特征提取](#encoderdinov3-多层特征提取)
  - [Decoder：FPN + UNet 混合结构](#decoderfpn--unet-混合结构)
  - [损失函数：三合一组合 Loss](#损失函数三合一组合-loss)
  - [学习率调度：Warmup + Cosine](#学习率调度warmup--cosine)
  - [滑窗推理](#滑窗推理)
- [评估指标](#评估指标)
- [配置说明](#配置说明)
- [License](#license)

---

## 项目背景

金相分析中，晶界（grain boundary）的自动提取是材料科学的关键任务。晶界具有以下特点：

- **细薄**：通常仅 1-3 像素宽，属于极端的细粒度分割
- **拓扑连续**：晶界必须形成闭合回路才有物理意义
- **尺度不一**：同一图像中晶粒大小差异可达数十倍

传统图像处理方法（Canny、分水岭等）在噪声大、对比度低的工业场景下鲁棒性不足。本项目采用深度学习方案，利用 DINOv3 自监督预训练的强语义特征，配合边界感知损失，实现高精度晶界分割。

---

## 模型架构

```
输入图像 (768×768×3)
       │
       ▼
┌─────────────────────┐
│  DINOv3 Encoder     │  ← ViT-B/16, pretrained on LVD-1689M
│  (frozen backbone)  │
├─────────────────────┤
│  f4: low-level      │  ← hidden_states[-3] → reshape to [B, C, H/4, W/4]
│  f8: mid-level      │  ← hidden_states[-2] → reshape to [B, C, H/8, W/8]
│  f16: high-level    │  ← hidden_states[-1] → reshape to [B, C, H/16, W/16]
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  FPN Projection     │
│  p16/p8/p4 → 256ch  │  ← 1×1 Conv 统一通道数
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  UNet-style Decoder │
│  p16 → upsample     │
│    + p8 → ConvBlock │  ← 逐级上采样 + 跳跃连接
│    + p4 → ConvBlock │
│    → upsample → head│
└─────────┬───────────┘
          │
          ▼
输出 logits (768×768×2)
```

### 关键参数

| 组件 | 参数 |
|------|------|
| Backbone | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| 输入尺寸 | 768×768 |
| FPN 通道数 | 256 |
| 解码器通道数 | 256→128→64→32 |
| 参数量（骨干冻结） | ~15M（仅解码器可训练） |

---

## 目录结构

```
├── datasets/
│   ├── grain_dataset.py      # 数据集加载（训练/验证/测试/无标注）
│   └── transforms.py         # 数据增强（Albumentations）
│
├── models/
│   ├── dinov3_encoder.py     # DINOv3 多层特征提取器
│   ├── fpn_unet_decoder.py   # FPN + UNet 混合解码器
│   └── dinov3_segmentation.py # 完整模型（Encoder + Decoder）
│
├── losses/
│   ├── dice.py               # Dice Loss（占位，实际实现在 loss.py）
│   ├── boundary_loss.py      # Boundary Loss（占位，实际实现在 loss.py）
│   └── loss.py               # 总损失：0.4×BCE + 0.4×Dice + 0.2×Boundary
│
├── utils/
│   ├── metric.py             # 评估指标：IoU、Dice、Boundary F1
│   ├── visualize.py          # 可视化工具：mask 着色、叠加显示
│   ├── scheduler.py          # Warmup + Cosine 学习率调度器
│   └── seed.py               # 随机种子固定（可复现性）
│
├── train.py                  # 训练入口
├── infer.py                  # 推理入口（单图 + 滑窗）
├── export_onnx.py            # ONNX 模型导出
├── config.py                 # 全局配置（dataclass）
│
├── checkpoints/              # 模型权重保存目录
└── README.md
```

---

## 环境依赖

```
torch >= 2.0
torchvision
transformers          # HuggingFace（加载 DINOv3）
albumentations        # 数据增强
opencv-python         # 图像读写
numpy
tqdm
```

一键安装：

```bash
pip install torch torchvision transformers albumentations opencv-python numpy tqdm
```

---

## 快速开始

### 1. 准备数据集

按以下结构组织数据：

```
dataset/
├── train/
│   ├── images/          # 训练原图 (.png/.jpg/.tif)
│   └── masks/           # 训练标注 (灰度图，晶界=255，背景=0)
├── val/
│   ├── images/          # 验证原图
│   └── masks/           # 验证标注
├── test/
│   └── images/          # 测试原图（无需标注）
└── unlabeled/           # 无标注数据（可选，用于伪标签扩展）
```

> **注意**：mask 文件名必须与对应 image 文件名一致。

### 2. 修改配置

编辑 [config.py](config.py)，调整以下关键参数：

```python
dataset_root: str = "./dataset"          # 数据集根目录
image_size: int = 768                    # 输入尺寸
batch_size: int = 4                      # 批次大小（显存不足可减小）
epochs: int = 80                         # 训练轮数
freeze_backbone: bool = True             # 是否冻结 DINOv3
```

### 3. 训练

```bash
python train.py
```

训练过程输出示例：

```
Epoch [1/80]
Train: 100%|██████████| 50/50 [02:30<00:00, loss=0.342]
Val: 100%|██████████| 10/10 [00:20<00:00]
Train Loss: 0.3512
Val Loss: 0.2891
Val IoU: 0.7823
Val Dice: 0.8534
Saved best model
```

- 每个 epoch 自动保存 `best.pth`（最佳 IoU）和 `last.pth`（最新）
- 支持断点续训：设置 `cfg.resume = "./checkpoints/last.pth"`

### 4. 推理

```bash
python infer.py
```

推理结果保存至 `./output/infer/`，每个输入生成：
- `*_mask.png`：二值分割结果
- `*_prob.png`：概率热力图

对于超过 768×768 的大图，自动启用 **滑窗推理**（`crop_size=768, stride=512`），通过重叠加权平均消除拼接伪影。

### 5. ONNX 导出

```bash
python export_onnx.py
```

导出为 `./onnx/grain_boundary.onnx`，支持动态 batch size 和输入尺寸，可直接部署到 Triton Inference Server 或其他 ONNX Runtime 环境。

---

## 核心设计

### Encoder：DINOv3 多层特征提取

不同于常见的仅取最后一层特征，本项目从 DINOv3 的 hidden states 中提取 **三个尺度** 的特征：

| 层级 | 来源 | 下采样倍率 | 语义 |
|------|------|-----------|------|
| f4 | hidden_states[-3] | 4× | 低级纹理（晶界边缘细节） |
| f8 | hidden_states[-2] | 8× | 中级结构（晶粒形状） |
| f16 | hidden_states[-1] | 16× | 高级语义（晶粒区域） |

特征从 `[B, N, C]` (patch sequence) reshape 为 `[B, C, H, W]` 空间格式，丢弃 CLS token。默认冻结骨干网络，仅训练解码器，避免在小数据集上过拟合。

### Decoder：FPN + UNet 混合结构

1. **FPN 阶段**：三个 1×1 卷积将不同尺度的特征统一投影到 256 通道
2. **UNet 阶段**：从最高语义层级（f16）开始，逐级 2× 上采样并与上一级特征相加（跳跃连接），经过双层卷积块（Conv→BN→ReLU→Conv→BN→ReLU）精炼
3. **输出头**：1×1 卷积输出 `num_classes=2` 通道 logits

### 损失函数：三合一组合 Loss

针对晶界分割的特殊性，采用三种损失的加权组合：

```
Total Loss = 0.4 × BCE + 0.4 × Dice + 0.2 × Boundary
```

| 损失 | 作用 | 实现细节 |
|------|------|---------|
| **BCE Loss** | 像素级二分类监督 | `BCEWithLogitsLoss`，对 logits 的晶界通道 (channel=1) 计算 |
| **Dice Loss** | 缓解类别不平衡（晶界像素占比极小） | Smooth Dice，加入 Laplace 平滑（eps=1） |
| **Boundary Loss** | 强化晶界边缘连续性 | 用 Laplacian 核 `[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]` 提取预测与真值的边缘，计算 L1 Loss |

> **为什么需要 Boundary Loss？** 晶界仅占图像像素的 1-3%，BCE + Dice 的组合倾向于给出模糊的边界预测。Boundary Loss 显式监督边缘梯度，迫使模型输出锐利、连续的晶界线。

### 学习率调度：Warmup + Cosine

```
Epoch 0-4:   linear warmup (0 → lr)
Epoch 5-79:  cosine decay (lr → min_lr)
```

- Warmup 阶段避免训练初期梯度不稳定导致骨干特征被破坏
- Cosine 衰减在训练后期平滑收敛，避免验证 loss 震荡
- 默认 `lr=1e-4`, `min_lr=1e-6`

### 滑窗推理

对于超过 `image_size` 的大图（如 2048×2048 的全景显微扫描），自动采用滑窗方式：

1. 以 `crop_size=768, stride=512` 在图像上滑动
2. 每个窗口独立推理，得到概率图
3. 重叠区域累加概率并除以计数矩阵（加权平均融合）
4. 边界窗口自动对齐到图像边缘，避免越界

---

## 评估指标

| 指标 | 说明 | 代码位置 |
|------|------|---------|
| **IoU (Jaccard)** | 逐类 IoU 的平均值（背景 + 晶界），评估整体分割质量 | [utils/metric.py](utils/metric.py) |
| **Dice (F1)** | 晶界类的二值 Dice 系数，对细粒度目标更敏感 | [utils/metric.py](utils/metric.py) |
| **Boundary F1** | 用 Laplacian 核提取边缘后计算 F1，评估晶界连续性 | [utils/metric.py](utils/metric.py) |

---

## 配置说明

完整配置见 [config.py](config.py)，以下是核心配置项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `dataset_root` | `"./dataset"` | 数据集根目录 |
| `backbone_name` | `"facebook/dinov3-vitb16-pretrain-lvd1689m"` | DINOv3 模型名称（HuggingFace） |
| `image_size` | `768` | 训练/推理输入尺寸 |
| `batch_size` | `4` | 批次大小 |
| `epochs` | `80` | 总训练轮数 |
| `learning_rate` | `1e-4` | 初始学习率 |
| `warmup_epochs` | `5` | Warmup 轮数 |
| `min_lr` | `1e-6` | 最小学习率 |
| `dice_weight` | `0.4` | Dice Loss 权重 |
| `bce_weight` | `0.4` | BCE Loss 权重 |
| `boundary_weight` | `0.2` | Boundary Loss 权重 |
| `freeze_backbone` | `True` | 是否冻结 DINOv3 骨干 |
| `amp` | `True` | 混合精度训练 |
| `crop_size` | `768` | 滑窗推理窗口大小 |
| `stride` | `512` | 滑窗步长 |
| `onnx_path` | `"./onnx/grain_boundary.onnx"` | ONNX 导出路径 |

---

## License

本项目仅供学习和研究使用。
