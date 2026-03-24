# Aesthetic Workflow Suite

一套完整的美学打分数据工作流，覆盖：

1. 标注（`labeling_ui`）
2. 训练（`training_ui`）
3. 批处理推理（`infer_ui`）
4. 单图推理（`infer_ui` WebUI）

---

## Repository Structure

```text
apps/
  labeling_ui/      # 标注 UI
  training_ui/      # 训练 UI
  infer_ui/         # 批量/单图推理 UI 与脚本
  README.md
  requirements.txt
  .gitignore
```

---

## Environment Setup

建议 Python 3.10+（Windows 环境）。

```bat
cd /d d:\vscode\vibecode\apps
py -3 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Quick Start

### 1) 标注 UI

默认端口：`9100`

```bat
cd /d d:\vscode\vibecode\apps\labeling_ui
py -3 run.py --config config.yaml
```

### 2) 训练 UI

默认端口：`9300`

```bat
cd /d d:\vscode\vibecode\apps\training_ui
py -3 run.py --config config.yaml
```

### 3) 推理 UI（批处理 + 单图）

默认端口：`9400`

```bat
cd /d d:\vscode\vibecode\apps\infer_ui
py -3 run.py --config config.yaml
```

在推理 UI 中：

- `device=auto`：自动选择
- `device=cpu`：强制 CPU
- `device=gpu`：自动检测 CUDA 与 torch，缺失时自动安装（torch/torchvision）

---

## Workflow

1. 在 `labeling_ui` 完成数据标注，产出 `labels.db` 与图片数据集。
2. 在 `training_ui` 指定标注库与图片根目录，训练得到 checkpoint（推荐 `.safetensors`）。
3. 在 `infer_ui` 填写 checkpoint 与输入目录，执行批处理推理。
4. 在 `infer_ui` 单图页面按路径或上传图片进行单图推理与结果解构。

---

## Notes

- 当前 `.gitignore` 默认忽略大体积数据和模型：
  - `labeling_ui/dataset/`
  - `training_ui/outputs/`
  - `infer_ui/outputs/`
  - `infer_ui/_models/`
  - `*.safetensors`, `*.pt`, `*.pth`, `*.bin`
- 若需要版本化模型文件，建议使用 Git LFS。

---

## Publish to GitHub

```bat
cd /d d:\vscode\vibecode\apps
git init
git add .
git commit -m "init: integrate labeling training inference workflow"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```
