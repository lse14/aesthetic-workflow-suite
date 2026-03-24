Infer UI 打包分发说明（仅打包 infer_ui）

1) 发送方：
- 将整个 infer_ui 文件夹压缩为 zip。
- 模型文件（.safetensors）单独发送。

2) 接收方：
- 解压 zip 后进入 infer_ui 目录，双击 start.bat。
- 首次运行会创建 .venv 并安装依赖。
- 打开页面后，在 checkpoint 填入你发送的模型路径。
- input_dir 可留空（默认优先 infer_ui\data\infer_images）。
- 点击“启动推理”即可。

3) 本地缓存位置：
- infer_ui\_models（HF/JTP3 相关缓存都在这里）。

4) 一键下载 JTP3 基座（推荐先执行）：
- 双击 `prefetch_jtp3.bat`
- 默认下载到：`infer_ui\_models\repos\RedRocket__JTP-3`
- 下载完成后，再运行 `start.bat`。
