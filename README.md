# Aesthetic Workflow Suite

End-to-end aesthetic scoring workflow for real projects:  
**Labeling -> Training -> Batch Inference -> Single-Image Inference -> Batch Post-Process**

美学打分全流程工具链（可直接落地使用）：  
**标注 -> 训练 -> 批处理推理 -> 单图推理 -> 批量后处理**

---

## Why This Repo / 项目价值

- One repo, full workflow  
  一个仓库覆盖完整流程，不用拼接多套工具
- Web UI for each stage  
  每个阶段都有独立 Web UI
- Practical inference runtime  
  推理支持 CPU/GPU 选择，GPU 模式可自动准备 torch 运行时
- Safe-by-default publishing setup  
  默认忽略数据集、模型和输出，适合公开协作

---

## Components / 组件

```text
apps/
  labeling_ui/      # 数据标注 UI
  training_ui/      # 训练 UI
  infer_ui/         # 批量推理 + 单图推理 UI
  batch/            # 批处理脚本（便携推理/结果整理）
  README.md
  requirements.txt
  .gitignore
```

---

## Quick Start (60s) / 60 秒启动

Python 3.10+ (Windows recommended).

```bat
cd /d d:\vscode\vibecode\apps
py -3 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Start each UI:

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

Default ports / 默认端口:
- `labeling_ui`: `9100`
- `training_ui`: `9300`
- `infer_ui`: `9400`

---

## Typical Workflow / 典型流程

1. Label data in `labeling_ui` and generate annotation DB  
   在 `labeling_ui` 完成标注，生成标注库
2. Train model in `training_ui`, export checkpoint (`.safetensors` recommended)  
   在 `training_ui` 训练模型，导出 checkpoint（推荐 `.safetensors`）
3. Run batch inference in `infer_ui`  
   在 `infer_ui` 进行批处理推理
4. Use single-image page for quick inspection (path/upload)  
   用单图页面进行路径/上传推理与快速检查
5. (Optional) Use `batch/` scripts for portable inference and score-based image sorting  
   （可选）使用 `batch/` 脚本做便携推理与按分数整理图片

---

## Inference Device Modes / 推理设备模式

- `auto`: automatic selection / 自动选择
- `cpu`: force CPU / 强制 CPU
- `gpu`: require GPU, check CUDA, auto-install torch runtime when needed  
  强制 GPU，检测 CUDA，必要时自动安装 torch 运行时

---

## Security & Privacy / 安全与隐私

- Dataset, model weights, outputs, caches are ignored by default  
  数据集、模型权重、输出目录、缓存默认忽略
- Keep private keys/tokens in environment variables  
  私有密钥和 token 请放环境变量，不要写入配置文件
- Use Git LFS if you decide to version large model files  
  如需管理大模型文件，请使用 Git LFS

---

## License

Add your preferred license before making the repository public for broader reuse.  
如果要更广泛开源复用，建议在公开前补充许可证。
