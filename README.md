### 简介

一个美学评分工作流，包括：

1. 标注（`labeling_ui`）
2. 训练（`training_ui`）
3. 批量推理与单图推理（`infer_ui`）
4. 便携批处理与分拣（`batch`）

模型发布页：[lse14/lse14-scorer](https://huggingface.co/lse14/lse14-scorer)

训练数据配比
- `0.2` Danbooru 图像
- `0.4` e621 图像
- `0.4` 本地图像

### 仓库结构

```text
apps/
  labeling_ui/      # 标注 UI
  training_ui/      # 训练 UI
  infer_ui/         # 批量推理 + 单图推理 UI
  batch/            # 便携批处理脚本（推理/分拣）
  models/           # 训练好的模型发布目录（双模型位）
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

### 快速开始（推荐）

推荐使用仓库根目录启动器：

```bat
cd /d <repo_root>
start_all.bat
```

`start_all.bat` 用于统一入口启动 `labeling_ui / training_ui / infer_ui / batch`。

也可直接启动单个模块：

```bat
cd /d <repo_root>\labeling_ui
start.bat
```

```bat
cd /d <repo_root>\training_ui
start.bat
```

```bat
cd /d <repo_root>\infer_ui
start.bat
```

默认端口：

- `labeling_ui`: `9100`
- `training_ui`: `9300`
- `infer_ui`: `9400`

### 嵌入式运行时（免系统 Python）

以下启动脚本均采用嵌入式 Python 运行（不回退系统 Python）：

- `labeling_ui/start.bat`
- `training_ui/start.bat`
- `infer_ui/start.bat`
- `batch/run_portable_infer.bat`

运行时查找顺序：

1. `<app>/runtime/python/python.exe`
2. `../runtime/python/python.exe`

若上述路径均不存在，启动脚本会自动下载并解压 embeddable Python 至 `runtime/python/`，随后安装依赖并启动服务。

### 推理设备说明

- `auto`：自动选择
- `cpu`：强制 CPU
- `gpu`：强制 GPU，自动检测 CUDA 与 torch；缺失时自动安装 torch 运行时

### 安全与隐私

请将私密信息（token/key）放入环境变量，不要写入配置文件。

### 开发方式

本项目部分功能迭代采用 **OpenAI Codex** 协作开发。

### 致谢

- Web 与服务框架：`FastAPI`, `Uvicorn`
  - FastAPI: https://github.com/fastapi/fastapi
  - Uvicorn: https://github.com/encode/uvicorn
- 深度学习与模型生态：`PyTorch`, `Transformers`, `OpenCLIP`, `timm`, `safetensors`
  - PyTorch: https://pytorch.org/
  - Transformers: https://github.com/huggingface/transformers
  - OpenCLIP: https://github.com/mlfoundations/open_clip
  - timm: https://github.com/huggingface/pytorch-image-models
  - safetensors: https://github.com/huggingface/safetensors
- 相关模型来源（按本项目配置/流程使用）
  - JTP-3: https://huggingface.co/RedRocket/JTP-3
  - Waifu Scorer v3: https://huggingface.co/Eugeoter/waifu-scorer-v3

许可与使用说明：

- 本仓库代码仅覆盖本项目自身实现；第三方模型与依赖库遵循其各自许可证与使用条款。
- 使用、分发、商用前请自行核验对应模型与依赖的 license / ToS / 权利边界。
- 如上游项目要求引用（citation/attribution），请按其官方说明进行署名。

---

### Overview

This repository provides an end-to-end aesthetic scoring workflow:

1. Labeling (`labeling_ui`)
2. Training (`training_ui`)
3. Batch + single-image inference (`infer_ui`)
4. Portable batch processing and sorting (`batch`)

Model release page: [lse14/lse14-scorer](https://huggingface.co/lse14/lse14-scorer)

Training data ratio:
- `0.2` Danbooru 
- `0.4` e621 
- `0.4` local


### Repository Layout

```text
apps/
  labeling_ui/      # Labeling UI
  training_ui/      # Training UI
  infer_ui/         # Batch + single-image inference UI
  batch/            # Portable batch scripts (infer/sort)
  models/           # Published trained models (two slots)
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

### Quick Start (Recommended)

Use the root launcher as the default entrypoint:

```bat
cd /d <repo_root>
start_all.bat
```

`start_all.bat` provides a unified launcher for `labeling_ui / training_ui / infer_ui / batch`.

You can also start each module directly:

```bat
cd /d <repo_root>\labeling_ui
start.bat
```

```bat
cd /d <repo_root>\training_ui
start.bat
```

```bat
cd /d <repo_root>\infer_ui
start.bat
```

Default ports:

- `labeling_ui`: `9100`
- `training_ui`: `9300`
- `infer_ui`: `9400`

### Embedded Runtime (No System Python Required)

The following launchers run on embedded Python only (no system Python fallback):

- `labeling_ui/start.bat`
- `training_ui/start.bat`
- `infer_ui/start.bat`
- `batch/run_portable_infer.bat`

Runtime resolution order:

1. `<app>/runtime/python/python.exe`
2. `../runtime/python/python.exe`

If no runtime is found, launchers automatically download and extract embeddable Python into `runtime/python/`, then install dependencies and start services.

### Inference Device Modes

- `auto`: automatic selection
- `cpu`: force CPU
- `gpu`: force GPU, auto-check CUDA/torch, auto-install torch runtime when missing

### Security & Privacy

Keep private credentials in environment variables. Do not hardcode keys/tokens in config files.

### Development Note

Parts of this project were iterated with **OpenAI Codex** in a vibe-coding workflow (requirement-driven, small patches, quick validation loops).

### Acknowledgements

- Web/runtime foundations: `FastAPI`, `Uvicorn`
  - FastAPI: https://github.com/fastapi/fastapi
  - Uvicorn: https://github.com/encode/uvicorn
- ML/model ecosystem: `PyTorch`, `Transformers`, `OpenCLIP`, `timm`, `safetensors`
  - PyTorch: https://pytorch.org/
  - Transformers: https://github.com/huggingface/transformers
  - OpenCLIP: https://github.com/mlfoundations/open_clip
  - timm: https://github.com/huggingface/pytorch-image-models
  - safetensors: https://github.com/huggingface/safetensors
- Model sources used by this workflow
  - JTP-3: https://huggingface.co/RedRocket/JTP-3
  - Waifu Scorer v3: https://huggingface.co/Eugeoter/waifu-scorer-v3

License and usage notice:

- This repository license applies only to this project's own code.
- Third-party libraries/models keep their own licenses and terms.
- Please verify license/ToS/commercial-use constraints before redistribution or production use.
- If upstream projects require citation/attribution, follow their official instructions.
