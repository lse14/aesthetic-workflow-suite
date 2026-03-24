# Models

该目录用于存放可发布的训练模型。

建议结构：

```text
models/
  model_a/
    MODEL_CARD.md
    <your_model>.safetensors
  model_b/
    MODEL_CARD.md
    <your_model>.safetensors
```

标准训练数据配比（本仓库约定）：

- `0.2` Danbooru 图像
- `0.4` e621 图像
- `0.4` 本地图像

建议使用 Git LFS 管理大模型文件。

