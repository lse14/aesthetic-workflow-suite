# Aesthetic Workflow Suite

---

## 中文说明

### 项目简介

这是一个面向真实生产流程的美学评分工作流仓库，覆盖：

1. 标注（`labeling_ui`）
2. 训练（`training_ui`）
3. 批量推理与单图推理（`infer_ui`）
4. 便携批处理与分拣（`batch`）

核心目标是把“数据生产 -> 模型训练 -> 推理落地”串成一条可持续迭代的链路。

### 仓库结构

```text
apps/
  labeling_ui/      # 标注 UI
  training_ui/      # 训练 UI
  infer_ui/         # 批量推理 + 单图推理 UI
  batch/            # 便携批处理脚本（推理/分拣）
  README.md
  requirements.txt
  .gitignore
```

### 模型架构

训练与推理使用融合多任务头，结构如下：

1. 特征提取器 A：`JTP-3` 视觉特征
2. 特征提取器 B：`Waifu/CLIP` 视觉特征（可带 waifu 分支信息）
3. 特征融合：`concat([feat_jtp3, feat_waifu], dim=-1)`
4. 任务头：共享 trunk + 多头输出
   1. 回归头（4 个）：`aesthetic`、`composition`、`color`、`sexual`
   2. 分类头（1 个）：`in_domain`（用于 special 样本判定）

推理时：

- 回归输出映射到 `1~5` 分区间
- 分类头通过阈值 `special_threshold` 计算 `special_tag`

### 快速开始

推荐 Python 3.10+（Windows）。

```bat
cd /d d:\vscode\vibecode\apps
py -3 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

启动服务：

```bat
cd /d d:\vscode\vibecode\apps\labeling_ui
py -3 run.py --config config.yaml
```

```bat
cd /d d:\vscode\vibecode\apps\training_ui
py -3 run.py --config config.yaml
```

```bat
cd /d d:\vscode\vibecode\apps\infer_ui
py -3 run.py --config config.yaml
```

默认端口：

- `labeling_ui`: `9100`
- `training_ui`: `9300`
- `infer_ui`: `9400`

### 推理设备说明

- `auto`：自动选择
- `cpu`：强制 CPU
- `gpu`：强制 GPU，自动检测 CUDA 与 torch；缺失时自动安装 torch 运行时

### 安全与隐私

仓库默认忽略：

- 标注数据集
- 模型权重
- 输出目录
- 缓存和虚拟环境

请将私密信息（token/key）放入环境变量，不要写入配置文件。

### 开发方式

本项目部分功能迭代采用 **OpenAI Codex** 进行 vibe coding 协作开发（需求驱动、小步快改、快速回归验证）。

### 致谢

- `FastAPI`, `Uvicorn` 提供 Web 服务基础
- `PyTorch`, `Transformers`, `open_clip`, `timm`, `safetensors` 提供模型与推理能力
- 感谢开源社区以及数据标注、训练、验证参与者

---

## English

### Overview

This repository provides an end-to-end aesthetic scoring workflow:

1. Labeling (`labeling_ui`)
2. Training (`training_ui`)
3. Batch + single-image inference (`infer_ui`)
4. Portable batch processing and sorting (`batch`)

The goal is to keep the full pipeline reproducible and practical for iterative model development.

### Repository Layout

```text
apps/
  labeling_ui/      # Labeling UI
  training_ui/      # Training UI
  infer_ui/         # Batch + single-image inference UI
  batch/            # Portable batch scripts (infer/sort)
  README.md
  requirements.txt
  .gitignore
```

### Model Architecture

The training/inference stack uses a fused multi-task head:

1. Feature extractor A: `JTP-3` visual features
2. Feature extractor B: `Waifu/CLIP` visual features (optionally with waifu branch info)
3. Fusion: `concat([feat_jtp3, feat_waifu], dim=-1)`
4. Shared trunk + multi-head outputs:
   1. Four regression heads: `aesthetic`, `composition`, `color`, `sexual`
   2. One classification head: `in_domain` (used for special-sample tagging)

At inference time:

- Regression outputs are mapped to `1~5`
- Classification probability + `special_threshold` determines `special_tag`

### Quick Start

Python 3.10+ (Windows recommended).

```bat
cd /d d:\vscode\vibecode\apps
py -3 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Start services:

```bat
cd /d d:\vscode\vibecode\apps\labeling_ui
py -3 run.py --config config.yaml
```

```bat
cd /d d:\vscode\vibecode\apps\training_ui
py -3 run.py --config config.yaml
```

```bat
cd /d d:\vscode\vibecode\apps\infer_ui
py -3 run.py --config config.yaml
```

Default ports:

- `labeling_ui`: `9100`
- `training_ui`: `9300`
- `infer_ui`: `9400`

### Inference Device Modes

- `auto`: automatic selection
- `cpu`: force CPU
- `gpu`: force GPU, auto-check CUDA/torch, auto-install torch runtime when missing

### Security & Privacy

By default, this repo ignores:

- datasets
- model weights
- runtime outputs
- caches and virtual environments

Keep private credentials in environment variables. Do not hardcode keys/tokens in config files.

### Development Note

Parts of this project were iterated with **OpenAI Codex** in a vibe-coding workflow (requirement-driven, small patches, quick validation loops).

### Acknowledgements

- `FastAPI` and `Uvicorn` for web serving
- `PyTorch`, `Transformers`, `open_clip`, `timm`, and `safetensors` for model and inference stack
- Thanks to contributors involved in labeling, training, and evaluation
