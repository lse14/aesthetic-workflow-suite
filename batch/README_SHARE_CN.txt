批量推理分拣（单文件夹可转发版）
================================

目标
----
只压缩并发送 `batch` 文件夹即可。
模型文件（`.safetensors`）可单独发送，接收方在界面里自行选择路径。

接收方使用步骤
--------------
1. 解压你发送的 `batch` 文件夹。
2. 双击运行 `run_portable_infer.bat`。
3. 首次运行会自动创建 `.venv` 并安装依赖（需要联网）。
4. 在弹出的界面中：
   - 选择模型路径（你单独发送的 `.safetensors`）
   - 选择图片文件夹
   - 选择维度（单选）
   - 选择是否遍历子文件夹
5. 点击“开始处理”，完成后会弹窗提示输出目录和统计信息。

waifu 权重说明
-------------
- 若 checkpoint 内保存的是你本机绝对路径，程序会自动尝试这些位置：
  - `batch\_models\waifu-scorer-v3\model.safetensors`
  - `%FUSION_MODEL_CACHE_ROOT%\waifu-scorer-v3\model.safetensors`
  - checkpoint 同级目录下的 `waifu-scorer-v3\model.safetensors`

输出位置
--------
默认输出在：
`batch\outputs\batch_sort_<dimension>\`

注意
----
- 如对方机器无 CUDA，会自动用 CPU。
- `HF_TOKEN` 警告是可选优化，不影响功能。
- 如希望尽量离线使用，可将模型缓存目录 `_models` 一并发送（可选）。
