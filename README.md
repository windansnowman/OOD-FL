# 基于 OOD 偏差漂移的联邦学习后门防御

👋 欢迎来到本仓库！这里是我在**本科阶段的第二项研究工作**的代码实现，专注于联邦学习（Federated Learning, FL）环境下的后门攻击检测与防御。

## 📖 简介 (Introduction)

本项目提出了一种全新的联邦学习后门防御框架。我们在研究中揭示了一个关键现象：后门植入会导致模型在**分布内数据（In-Distribution, ID）**和**分布外数据（Out-of-Distribution, OOD）**之间产生可测量的**偏差漂移（Bias Shift）**。具体而言，受后门感染的模型会在 OOD 数据上对目标类别表现出过度的自信，同时抑制 ID 偏差。

基于这一核心发现，本框架实现了两阶段的强力防御：
1. **精准检测恶意更新**：通过计算客户端局部模型在 ID 数据与 OOD 数据上软标签（Soft-label）偏差的差异得分，准确过滤掉恶意客户端的更新。
2. **后门消除与净化**：在全局聚合阶段，通过自适应剪枝（Adaptive Pruning）技术，定位并剔除对后门至关重要的神经元，从而在不损害模型主任务性能的前提下彻底消除后门。

在高度 Non-IID（非独立同分布）的数据划分设置下，本框架在 CIFAR-10 和 Tiny-ImageNet 数据集上均进行了广泛评估。实验结果表明，本方法在真阳性率（TPR）、假阳性率（FPR）以及攻击成功率（ASR）的降低幅度上，显著优于现有的 SOTA 防御方法（如 FLAME, FDCR, Indicator, AlignIns 等）。

## ✨ 核心特性 (Features)

- 🚀 **新颖的检测范式**：首次利用 ID 与 OOD 数据之间的偏差漂移作为联邦学习后门的检测指标。
- 🛡️ **双重防御机制**：结合了“恶意客户端过滤（差异得分检测）”与“全局模型自适应剪枝”的联合防御。
- 📦 **丰富的基线支持**：
  - **主流攻击库**：内置 A3FL, Chameleon, DarkFed, DBA, Neurotoxin, PFedBA, PGD, WaNet 等多种前沿后门攻击的实现。
  - **SOTA 防御库**：集成了 AlignIns, DeepSight, FedDMC, FLAME, FoolsGold, Indicator, MultiKrum 等最新防御策略。
- 🧩 **高度模块化**：Client、Server、Dataloader 与 Models 解耦，通过 YAML 文件一键配置复杂的联邦攻防实验。

## 📂 代码结构 (Directory Structure)

本项目的代码结构设计清晰，易于扩展和二次开发：

```text
.
├── main.py                   # 实验启动主入口
├── dataloader/               # 数据集加载与处理模块 (General FL, Watermark FL 等)
├── dataset/                  # 存放实验所需数据集及 OOD/压缩 数据
├── models/                   # 模型架构定义 (CNN, ResNet, VGG, ViT-B 等)
├── participants/             # 联邦学习节点抽象与实现
│   ├── clients/              # 客户端逻辑 (涵盖良性客户端及各类后门攻击客户端)
│   └── servers/              # 服务端逻辑 (涵盖我们的防御方法及其他基线聚合策略)
└── utils/                    # 工具函数、损失函数与核心配置文件
    ├── losses.py             # 攻防相关的自定义损失函数
    └── yamls/                # 丰富的实验配置文件，按防御策略及攻击类型分类归档
```

## 🛠️ 快速开始 (Quick Start)

1. 环境依赖  
建议使用 Python 3.8+ 及 PyTorch。安装必要的依赖项：

```bash
pip install -r requirements.txt
```

（请确保您的环境中已正确配置 PyTorch, torchvision, pyyaml, numpy 等常用科学计算包）

2. 运行实验  
实验的所有超参数和策略均由 `utils/yamls/` 目录下的配置文件控制。例如，要运行我们的防御框架对抗 DBA（Distributed Backdoor Attack）攻击，可以直接通过命令行指定对应的 YAML 配置文件：

```bash
python main.py --config utils/yamls/ours/params_dba_ours.yaml
```

若需评估其他基线防御（例如使用 FLAME 防御 A3FL 攻击），只需替换配置文件路径即可：

```bash
python main.py --config utils/yamls/flame/params_a3fl_Flame.yaml
```

## 📊 实验结果 (Results Highlights)

在 CIFAR-10 和 Tiny-ImageNet 数据集上，本方法在以下维度均取得了领先的防御效果：

- 高 TPR 与低 FPR：在高度 Non-IID 场景下，依然能够精准区分正常客户端与恶意客户端，避免误杀良性节点。
- 卓越的 ASR 抑制：面对 Neurotoxin、Chameleon 等高级自适应后门攻击，能够在保持主任务准确率（Main Task Accuracy）稳定不降的同时，将攻击成功率（Attack Success Rate）降至极低水平。

