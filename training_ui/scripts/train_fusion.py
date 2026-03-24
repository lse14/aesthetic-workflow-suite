import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_CONFIG_PATH = ROOT / "configs" / "fusion_1k_baseline.yaml"
TARGETS = ("aesthetic", "composition", "color", "sexual")
ALLOWED_MODEL_FORMATS = ("safetensors", "pt", "pth", "ckpt")

DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "annotations": "data/annotations.jsonl",
        "image_root": None,
        "train_split": "train",
        "val_split": "val",
    },
    "models": {
        "jtp3_model_id": "RedRocket/JTP-3",
        "jtp3_fallback_model_id": None,
        "hf_token_env": "HF_TOKEN",
        "waifu_clip_model_name": "ViT-L-14",
        "waifu_clip_pretrained": "openai",
        "waifu_v3_head_path": None,
        "freeze_extractors": True,
        "include_waifu_score": True,
    },
    "model_head": {
        "hidden_dims": [1024, 256],
        "dropout": 0.2,
    },
    "training": {
        "seed": 42,
        "device": "cuda",
        "batch_size": 8,
        "num_workers": 4,
        "val_ratio": None,
        "epochs": 10,
        "lr": 0.0003,
        "weight_decay": 0.0001,
        "loss": "mse",
        "cls_loss_weight": 1.0,
        "cls_pos_weight": None,
        "output_dir": "outputs/fusion_auto",
    },
}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool):
    from torch.utils.data import DataLoader
    from fusion_scorer.data import collate_pil_batch

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pil_batch,
        pin_memory=True,
    )


def fmt_or_dash(v: float) -> str:
    return "-" if not math.isfinite(float(v)) else f"{float(v):.4f}"


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_base_config(config_path: Path | None) -> tuple[dict[str, Any], Path | None]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    loaded_from: Path | None = None

    if DEFAULT_CONFIG_PATH.exists():
        tpl = load_config(DEFAULT_CONFIG_PATH) or {}
        if isinstance(tpl, dict):
            cfg = _deep_merge(cfg, tpl)
            loaded_from = DEFAULT_CONFIG_PATH.resolve()

    if config_path is not None:
        user_cfg = load_config(config_path) or {}
        if not isinstance(user_cfg, dict):
            raise ValueError("配置文件顶层必须是 mapping。")
        cfg = _deep_merge(cfg, user_cfg)
        loaded_from = config_path.resolve()
    return cfg, loaded_from


def _normalize_split(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"none", "null"}:
        return None
    return s


def _resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    p = Path(str(value))
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def _apply_simple_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    cfg.setdefault("data", {})
    cfg.setdefault("training", {})

    if args.annotations is not None:
        cfg["data"]["annotations"] = args.annotations
    if args.image_root is not None:
        cfg["data"]["image_root"] = args.image_root
    if args.train_split is not None:
        cfg["data"]["train_split"] = args.train_split
    if args.val_split is not None:
        cfg["data"]["val_split"] = args.val_split
    if args.device is not None:
        cfg["training"]["device"] = args.device
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        cfg["training"]["num_workers"] = int(args.num_workers)
    if args.val_ratio is not None:
        cfg["training"]["val_ratio"] = float(args.val_ratio)
    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if args.lr is not None:
        cfg["training"]["lr"] = float(args.lr)
    if args.weight_decay is not None:
        cfg["training"]["weight_decay"] = float(args.weight_decay)
    if args.seed is not None:
        cfg["training"]["seed"] = int(args.seed)
    if args.output_dir is not None:
        cfg["training"]["output_dir"] = args.output_dir
    if args.model_name is not None:
        cfg["training"]["model_name"] = args.model_name
    if args.model_format is not None:
        cfg["training"]["model_format"] = args.model_format
    if args.loss is not None:
        cfg["training"]["loss"] = args.loss
    if args.cls_loss_weight is not None:
        cfg["training"]["cls_loss_weight"] = float(args.cls_loss_weight)
    if args.cls_pos_weight is not None:
        cfg["training"]["cls_pos_weight"] = float(args.cls_pos_weight)


def _build_config(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    cfg_path = args.config.resolve() if args.config is not None else None
    cfg, loaded_from = _load_base_config(cfg_path)

    # 纯简化模式（不传 config）下，默认不过滤 split，避免只给训练集路径时直接报错。
    if loaded_from is None and args.train_split is None and args.val_split is None:
        cfg.setdefault("data", {})
        cfg["data"]["train_split"] = None
        cfg["data"]["val_split"] = None

    _apply_simple_overrides(cfg, args)

    data = cfg.setdefault("data", {})
    models = cfg.setdefault("models", {})
    training = cfg.setdefault("training", {})

    data["train_split"] = _normalize_split(data.get("train_split"))
    data["val_split"] = _normalize_split(data.get("val_split"))

    config_base = loaded_from.parent if loaded_from is not None else ROOT
    ann = _resolve_path(config_base, data.get("annotations"))
    if ann is None:
        raise ValueError("请提供训练数据标注文件：--annotations xxx.jsonl/csv/db")
    data["annotations"] = str(ann)

    img_root = _resolve_path(config_base, data.get("image_root"))
    data["image_root"] = str(img_root) if img_root is not None else None

    out_dir = _resolve_path(config_base, training.get("output_dir"))
    if out_dir is None:
        out_dir = (ROOT / "outputs" / "fusion_auto").resolve()
    training["output_dir"] = str(out_dir)

    waifu_head = _resolve_path(config_base, models.get("waifu_v3_head_path"))
    models["waifu_v3_head_path"] = str(waifu_head) if waifu_head is not None else None

    val_ratio_raw = training.get("val_ratio")
    if val_ratio_raw in (None, "", "null"):
        training["val_ratio"] = None
    else:
        training["val_ratio"] = float(val_ratio_raw)

    return cfg, config_base


def _print_train_args(
    cfg: dict[str, Any],
    *,
    eval_split: str | None,
    eval_batch_size: int,
    split_meta: dict[str, Any],
    target_dims: list[str],
) -> None:
    d = cfg["data"]
    t = cfg["training"]
    print("\n========== 训练参数 ==========")
    print(f"标注文件: {d['annotations']}")
    print(f"图片根目录: {d.get('image_root') or '(按标注中的绝对路径/相对路径解析)'}")
    print(f"训练 split: {d.get('train_split')}")
    print(f"评估 split: {eval_split}")
    print(
        "验证集策略: "
        f"{split_meta.get('strategy')} "
        f"(train_n={split_meta.get('train_size')}, val_n={split_meta.get('val_size')}, "
        f"val_ratio={split_meta.get('val_ratio')})"
    )
    print(f"设备: {t.get('device')}")
    print(f"批量大小: {t.get('batch_size')}  (评估批量: {eval_batch_size})")
    print(f"训练轮数: {t.get('epochs')}")
    print(f"学习率: {t.get('lr')}")
    print(f"权重衰减: {t.get('weight_decay')}")
    print(f"随机种子: {t.get('seed')}")
    print(f"损失函数: {t.get('loss')}")
    print(f"训练维度: {', '.join(target_dims)}")
    print(f"输出目录: {t.get('output_dir')}")


def _print_eval_result(report: dict[str, Any], target_dims: list[str]) -> None:
    overall = report.get("overall", {})
    cls = overall.get("classification", {})
    print("\n========== 自动评估结果 ==========")
    print(f"评估样本数(回归): {overall.get('n')}")
    print(
        "总体回归指标: "
        f"MAE={fmt_or_dash(overall.get('overall_mae', float('nan')))}  "
        f"RMSE={fmt_or_dash(overall.get('overall_rmse', float('nan')))}  "
        f"均值维度MAE={fmt_or_dash(overall.get('mean_dim_mae', float('nan')))}"
    )
    if report.get("has_cls_head", False):
        print(
            "分类指标(in_domain): "
            f"Acc={fmt_or_dash(cls.get('accuracy', float('nan')))}  "
            f"P={fmt_or_dash(cls.get('precision', float('nan')))}  "
            f"R={fmt_or_dash(cls.get('recall', float('nan')))}  "
            f"F1={fmt_or_dash(cls.get('f1', float('nan')))}"
        )
    else:
        print("分类指标: 当前 checkpoint 不包含分类头，已跳过。")

    per_dim = overall.get("per_dim", {})
    for name in target_dims:
        item = per_dim.get(name) or {}
        print(
            f"- {name}: "
            f"MAE={fmt_or_dash(item.get('mae', float('nan')))}  "
            f"RMSE={fmt_or_dash(item.get('rmse', float('nan')))}  "
            f"Spearman={fmt_or_dash(item.get('spearman', float('nan')))}"
        )


def _build_cn_summary(
    *,
    cfg: dict[str, Any],
    best_path: Path,
    history_path: Path,
    raw_eval_path: Path,
    report: dict[str, Any],
    eval_split: str | None,
    eval_batch_size: int,
    split_meta: dict[str, Any],
    eval_annotations: Path,
    target_dims: list[str],
) -> dict[str, Any]:
    overall = report.get("overall", {})
    cls = overall.get("classification", {})
    return {
        "说明": {
            "用途": "本文件是训练完成后的中文摘要，便于快速查看训练参数与评估指标。",
            "指标解读": {
                "回归_mae_rmse": "越小越好。",
                "spearman": "越接近 1 越好，表示排序一致性更高。",
                "分类_acc_f1": "越大越好。",
            },
        },
        "训练参数": {
            "annotations": cfg["data"].get("annotations"),
            "image_root": cfg["data"].get("image_root"),
            "train_split": cfg["data"].get("train_split"),
            "eval_split": eval_split,
            "val_ratio": cfg["training"].get("val_ratio"),
            "split_strategy": split_meta.get("strategy"),
            "train_size": split_meta.get("train_size"),
            "val_size": split_meta.get("val_size"),
            "device": cfg["training"].get("device"),
            "batch_size": cfg["training"].get("batch_size"),
            "eval_batch_size": eval_batch_size,
            "epochs": cfg["training"].get("epochs"),
            "lr": cfg["training"].get("lr"),
            "weight_decay": cfg["training"].get("weight_decay"),
            "seed": cfg["training"].get("seed"),
            "loss": cfg["training"].get("loss"),
            "target_dims": list(target_dims),
            "cls_loss_weight": cfg["training"].get("cls_loss_weight"),
            "cls_pos_weight": cfg["training"].get("cls_pos_weight"),
            "output_dir": cfg["training"].get("output_dir"),
        },
        "评估结果": {
            "样本数_回归": overall.get("n"),
            "总体_mae": overall.get("overall_mae"),
            "总体_rmse": overall.get("overall_rmse"),
            "均值维度_mae": overall.get("mean_dim_mae"),
            "均值维度_rmse": overall.get("mean_dim_rmse"),
            "分类_是否可用": bool(report.get("has_cls_head", False)),
            "分类_accuracy": cls.get("accuracy"),
            "分类_precision": cls.get("precision"),
            "分类_recall": cls.get("recall"),
            "分类_f1": cls.get("f1"),
            "分维度": overall.get("per_dim", {}),
        },
        "输出文件": {
            "best_checkpoint": str(best_path),
            "history_json": str(history_path),
            "eval_annotations": str(eval_annotations),
            "eval_report_json": str(raw_eval_path),
        },
    }


def _parse_val_ratio(v: Any) -> float | None:
    if v in (None, "", "null"):
        return None
    fv = float(v)
    if not (0.0 < fv < 1.0):
        raise ValueError("val_ratio 必须在 (0,1) 区间，例如 0.1 或 0.15")
    return fv


def _parse_target_dims(raw: Any) -> list[str]:
    if raw in (None, "", "null"):
        return list(TARGETS)
    if isinstance(raw, (list, tuple)):
        vals = [str(x).strip().lower() for x in raw if str(x).strip()]
    else:
        vals = [x.strip().lower() for x in str(raw).split(",") if x.strip()]
    if not vals:
        return list(TARGETS)
    invalid = [x for x in vals if x not in TARGETS]
    if invalid:
        raise ValueError(f"target_dims 含非法维度: {invalid}，可选: {list(TARGETS)}")
    # 去重并保持顺序
    out = []
    seen = set()
    for x in vals:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _normalize_model_format(raw: Any) -> str:
    fmt = str(raw or "").strip().lower().lstrip(".")
    if not fmt:
        return "safetensors"
    if fmt not in ALLOWED_MODEL_FORMATS:
        raise ValueError(f"model_format 不合法: {fmt}，可选: {list(ALLOWED_MODEL_FORMATS)}")
    return fmt


def _sanitize_model_name(raw: Any) -> str:
    name = str(raw or "").strip()
    if not name:
        return "best"
    # Prevent path traversal and keep filenames cross-platform friendly.
    name = Path(name).name
    if Path(name).suffix:
        name = Path(name).stem
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return name or "best"


def _build_model_filename(name_raw: Any, fmt_raw: Any) -> str:
    name = _sanitize_model_name(name_raw)
    fmt = _normalize_model_format(fmt_raw)
    return f"{name}.{fmt}"


def _build_train_val_datasets(
    *,
    annotations_path: Path,
    image_root: Path | None,
    train_split: str | None,
    val_split: str | None,
    val_ratio: float | None,
    seed: int,
    torch_mod,
    RatingDatasetCls,
) -> tuple[Any, Any, dict[str, Any], list[int] | None, Any]:
    base_ds = RatingDatasetCls(
        annotation_file=annotations_path,
        image_root=image_root,
        split=None,
    )
    has_split_field = bool(getattr(base_ds, "has_split_field", False))
    requested_split = bool(train_split is not None or val_split is not None)

    if has_split_field and requested_split:
        train_ds = RatingDatasetCls(
            annotation_file=annotations_path,
            image_root=image_root,
            split=train_split,
        )
        val_ds = RatingDatasetCls(
            annotation_file=annotations_path,
            image_root=image_root,
            split=val_split,
        )
        meta = {
            "strategy": "split_field",
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "val_ratio": None,
            "has_split_field": True,
        }
        return train_ds, val_ds, meta, None, base_ds

    if val_ratio is not None:
        n = len(base_ds)
        if n < 2:
            raise ValueError("样本数不足，无法按比例切分验证集（至少需要 2 条）。")
        val_n = int(round(n * float(val_ratio)))
        val_n = max(1, min(n - 1, val_n))
        generator = torch_mod.Generator().manual_seed(int(seed))
        perm = torch_mod.randperm(n, generator=generator).tolist()
        val_indices = perm[:val_n]
        train_indices = perm[val_n:]
        train_ds = torch_mod.utils.data.Subset(base_ds, train_indices)
        val_ds = torch_mod.utils.data.Subset(base_ds, val_indices)
        meta = {
            "strategy": "random_ratio",
            "train_size": len(train_indices),
            "val_size": len(val_indices),
            "val_ratio": float(val_n) / float(n),
            "has_split_field": bool(has_split_field),
        }
        return train_ds, val_ds, meta, val_indices, base_ds

    train_ds = RatingDatasetCls(
        annotation_file=annotations_path,
        image_root=image_root,
        split=train_split,
    )
    val_ds = RatingDatasetCls(
        annotation_file=annotations_path,
        image_root=image_root,
        split=val_split,
    )
    same_dataset = len(train_ds) == len(val_ds)
    meta = {
        "strategy": "shared_dataset" if same_dataset else "split_fallback",
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "val_ratio": None,
        "has_split_field": bool(has_split_field),
    }
    return train_ds, val_ds, meta, None, base_ds


def _write_eval_subset_annotations(base_ds, indices: list[int], out_path: Path) -> Path:
    rows = []
    for i in indices:
        rec = base_ds.records[int(i)]
        targets = rec["targets"].tolist()
        row = {
            "id": rec.get("id"),
            "image_path": str(rec["image_path"]),
            "aesthetic": float(targets[0]),
            "composition": float(targets[1]),
            "color": float(targets[2]),
            "sexual": float(targets[3]),
            "in_domain": int(round(float(rec["cls_target"].item()))),
            "exclude_from_score_train": 0
            if float(rec["score_mask"].sum().item()) > 0.5
            else 1,
            "exclude_from_cls_train": 0 if float(rec["cls_mask"].item()) > 0.5 else 1,
            "source": "random_val_subset",
        }
        rows.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_path


def _normalize_waifu_head_path(cfg: dict[str, Any], config_base: Path) -> None:
    models = cfg.setdefault("models", {})
    raw = models.get("waifu_v3_head_path")
    if not raw:
        return

    p = Path(str(raw))
    if not p.is_absolute():
        p = (config_base / p).resolve()

    if not p.exists():
        fallback = (ROOT.parent / "model" / "_models" / "waifu-scorer-v3" / "model.safetensors").resolve()
        if fallback.exists():
            p = fallback

    models["waifu_v3_head_path"] = str(p)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="训练+自动评估一体化脚本（简化参数，支持中文结果）。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="可选。高级模式配置文件；不传时走简化模式。",
    )
    parser.add_argument("--annotations", type=str, default=None, help="训练标注文件(.jsonl/.csv/.db)")
    parser.add_argument("--image-root", type=str, default=None, help="图片根目录，可不填")
    parser.add_argument("--train-split", type=str, default=None, help="训练 split；不填表示不过滤")
    parser.add_argument("--val-split", type=str, default=None, help="验证 split；不填表示不过滤")
    parser.add_argument("--val-ratio", type=float, default=None, help="无 split 时随机划分验证集比例，如 0.15")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=None, help="batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=None, help="权重衰减")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu")
    parser.add_argument("--output-dir", type=str, default=None, help="模型与报告输出目录")
    parser.add_argument("--model-name", type=str, default=None, help="输出模型名（不含扩展名）")
    parser.add_argument(
        "--model-format",
        type=str,
        default=None,
        choices=list(ALLOWED_MODEL_FORMATS),
        help="输出模型格式",
    )
    parser.add_argument("--loss", choices=["mse", "smooth_l1"], default=None, help="回归损失")
    parser.add_argument("--cls-loss-weight", type=float, default=None, help="分类损失权重")
    parser.add_argument("--cls-pos-weight", type=float, default=None, help="分类正样本权重")
    parser.add_argument("--eval-split", type=str, default=None, help="评估 split；默认使用 val_split")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="评估 batch size")
    parser.add_argument("--target-dims", type=str, default=None, help="训练维度，多选逗号分隔，如 aesthetic,color")
    parser.add_argument("--skip-eval", action="store_true", help="只训练，不做自动评估")
    args = parser.parse_args()

    import torch

    from fusion_scorer.data import RatingDataset
    from fusion_scorer.evaluation import run_evaluation
    from fusion_scorer.extractors import JTP3FeatureExtractor, WaifuV3ClipFeatureExtractor
    from fusion_scorer.model import FusionMultiTaskHead
    from fusion_scorer.train_utils import run_epoch, save_checkpoint, set_seed

    cfg, config_base = _build_config(args)
    _normalize_waifu_head_path(cfg, config_base)
    set_seed(int(cfg["training"]["seed"]))

    annotations_path = Path(str(cfg["data"]["annotations"])).resolve()
    if not annotations_path.exists():
        raise FileNotFoundError(f"标注文件不存在: {annotations_path}")
    image_root = None
    if cfg["data"].get("image_root"):
        image_root = Path(str(cfg["data"]["image_root"])).resolve()
        if not image_root.exists():
            raise FileNotFoundError(f"image_root 不存在: {image_root}")

    train_split = _normalize_split(cfg["data"].get("train_split"))
    val_split = _normalize_split(cfg["data"].get("val_split"))
    eval_split = _normalize_split(args.eval_split) if args.eval_split is not None else val_split
    val_ratio = _parse_val_ratio(cfg["training"].get("val_ratio"))
    target_dims = _parse_target_dims(args.target_dims if args.target_dims is not None else cfg["training"].get("target_dims"))
    cfg["training"]["target_dims"] = list(target_dims)
    model_filename = _build_model_filename(
        args.model_name if args.model_name is not None else cfg["training"].get("model_name"),
        args.model_format if args.model_format is not None else cfg["training"].get("model_format"),
    )
    cfg["training"]["model_name"] = Path(model_filename).stem
    cfg["training"]["model_format"] = Path(model_filename).suffix.lstrip(".").lower()

    device = str(cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[警告] 请求使用 CUDA，但当前不可用。已自动切换为 CPU。")
        device = "cpu"
    cfg["training"]["device"] = device
    cfg["training"]["val_ratio"] = val_ratio
    target_dims_set = set(target_dims)
    target_mask = torch.tensor(
        [1.0 if t in target_dims_set else 0.0 for t in TARGETS],
        dtype=torch.float32,
        device=device,
    )

    train_ds, val_ds, split_meta, random_val_indices, base_ds = _build_train_val_datasets(
        annotations_path=annotations_path,
        image_root=image_root,
        train_split=train_split,
        val_split=val_split,
        val_ratio=val_ratio,
        seed=int(cfg["training"]["seed"]),
        torch_mod=torch,
        RatingDatasetCls=RatingDataset,
    )

    train_loader = make_loader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"].get("num_workers", 4)),
        shuffle=True,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"].get("num_workers", 4)),
        shuffle=False,
    )

    eval_batch_size = int(args.eval_batch_size or cfg["training"]["batch_size"])
    _print_train_args(
        cfg,
        eval_split=eval_split,
        eval_batch_size=eval_batch_size,
        split_meta=split_meta,
        target_dims=target_dims,
    )
    if split_meta.get("strategy") == "shared_dataset":
        print("[提示] 当前 train/val 数据相同。建议设置 split，或配置 val_ratio 做随机验证切分。")

    jtp = JTP3FeatureExtractor(
        model_id=cfg["models"]["jtp3_model_id"],
        device=device,
        hf_token_env=cfg["models"].get("hf_token_env", "HF_TOKEN"),
        freeze=bool(cfg["models"].get("freeze_extractors", True)),
        fallback_model_id=cfg["models"].get(
            "jtp3_fallback_model_id",
            None,
        ),
    )
    print(
        f"[JTP] backend={getattr(jtp, 'backend', 'unknown')} "
        f"loaded_model={getattr(jtp, 'loaded_model_id', cfg['models']['jtp3_model_id'])}"
    )
    waifu = WaifuV3ClipFeatureExtractor(
        clip_model_name=cfg["models"].get("waifu_clip_model_name", "ViT-L-14"),
        clip_pretrained=cfg["models"].get("waifu_clip_pretrained", "openai"),
        waifu_head_path=cfg["models"].get("waifu_v3_head_path"),
        device=device,
        freeze=bool(cfg["models"].get("freeze_extractors", True)),
        include_waifu_score=bool(cfg["models"].get("include_waifu_score", True)),
    )
    if not (getattr(jtp, "freeze", True) and getattr(waifu, "freeze", True)):
        raise NotImplementedError(
            "This version trains fusion head only. Set models.freeze_extractors=true."
        )

    images0, _, _, _, _, _ = next(iter(train_loader))
    with torch.no_grad():
        dim_jtp = int(jtp(images0[:2]).shape[-1])
        dim_waifu = int(waifu(images0[:2]).shape[-1])
    input_dim = dim_jtp + dim_waifu

    fusion_head = FusionMultiTaskHead(
        input_dim=input_dim,
        hidden_dims=cfg["model_head"]["hidden_dims"],
        dropout=float(cfg["model_head"].get("dropout", 0.2)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        fusion_head.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )

    out_dir = ROOT / cfg["training"].get("output_dir", "outputs/fusion")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / model_filename
    history_path = out_dir / "history.json"

    history = {"train": [], "val": []}
    best_val_objective = float("inf")
    epochs = int(cfg["training"]["epochs"])
    loss_name = cfg["training"].get("loss", "mse").lower()
    cls_loss_weight = float(cfg["training"].get("cls_loss_weight", 1.0))
    cls_pos_weight_raw = cfg["training"].get("cls_pos_weight")
    cls_pos_weight = None if cls_pos_weight_raw in (None, "", "null") else float(cls_pos_weight_raw)

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            train=True,
            loader=train_loader,
            jtp_extractor=jtp,
            waifu_extractor=waifu,
            fusion_head=fusion_head,
            optimizer=optimizer,
            device=device,
            loss_name=loss_name,
            cls_loss_weight=cls_loss_weight,
            cls_pos_weight=cls_pos_weight,
            target_mask=target_mask,
        )
        val_metrics = run_epoch(
            train=False,
            loader=val_loader,
            jtp_extractor=jtp,
            waifu_extractor=waifu,
            fusion_head=fusion_head,
            optimizer=optimizer,
            device=device,
            loss_name=loss_name,
            cls_loss_weight=cls_loss_weight,
            cls_pos_weight=cls_pos_weight,
            target_mask=target_mask,
        )

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)
        history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

        train_dim = ", ".join(
            f"{name}:{fmt_or_dash(float(train_metrics['per_dim_mae'][i]))}" for i, name in enumerate(target_dims)
        )
        val_dim = ", ".join(
            f"{name}:{fmt_or_dash(float(val_metrics['per_dim_mae'][i]))}" for i, name in enumerate(target_dims)
        )
        print(
            f"[第 {epoch}/{epochs} 轮] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_mae={fmt_or_dash(train_metrics['mae'])} "
            f"train_cls_acc={fmt_or_dash(train_metrics['cls_acc'])} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_mae={fmt_or_dash(val_metrics['mae'])} "
            f"val_cls_acc={fmt_or_dash(val_metrics['cls_acc'])}"
        )
        print(
            "  "
            f"train 回归loss={train_metrics['score_loss']:.4f} "
            f"cls_loss={train_metrics['cls_loss']:.4f} "
            f"score_n={train_metrics['score_n']} cls_n={train_metrics['cls_n']}"
        )
        print(
            "  "
            f"val   回归loss={val_metrics['score_loss']:.4f} "
            f"cls_loss={val_metrics['cls_loss']:.4f} "
            f"score_n={val_metrics['score_n']} cls_n={val_metrics['cls_n']}"
        )
        print(f"  train 各维度MAE: [{train_dim}]")
        print(f"  val   各维度MAE: [{val_dim}]")

        val_objective = float(val_metrics["loss"])
        if val_objective < best_val_objective:
            best_val_objective = val_objective
            save_checkpoint(
                best_path,
                fusion_head=fusion_head,
                input_dim=input_dim,
                hidden_dims=cfg["model_head"]["hidden_dims"],
                dropout=float(cfg["model_head"].get("dropout", 0.2)),
                config=cfg,
                epoch=epoch,
                val_mae=float(val_metrics["mae"]),
                val_loss=float(val_metrics["loss"]),
                val_cls_acc=float(val_metrics["cls_acc"]),
                cls_loss_weight=cls_loss_weight,
            )
            print(
                f"  已保存最佳模型: {best_path} "
                f"(val_loss={best_val_objective:.4f})"
            )

    print(f"\n训练完成。最佳 val_loss={best_val_objective:.4f}")
    print(f"最佳模型: {best_path}")
    print(f"训练历史: {history_path}")

    if args.skip_eval:
        print("已按参数跳过自动评估（--skip-eval）。")
        return

    eval_annotations = annotations_path
    eval_image_root = image_root
    eval_split_for_run = eval_split
    if split_meta.get("strategy") == "random_ratio" and random_val_indices:
        eval_annotations = _write_eval_subset_annotations(
            base_ds,
            random_val_indices,
            out_dir / "random_val_subset.jsonl",
        )
        eval_image_root = None
        eval_split_for_run = None
        print(f"[提示] 已导出随机验证子集用于自动评估: {eval_annotations}")

    try:
        report = run_evaluation(
            checkpoint=best_path,
            annotations=eval_annotations,
            image_root=eval_image_root,
            split=eval_split_for_run,
            batch_size=eval_batch_size,
            device=device,
            target_dims=target_dims,
        )
    except RuntimeError as e:
        # 常见场景：配置了 val split，但数据里没有该 split，回退到不过滤评估。
        if "No rows after split filter." in str(e) and eval_split_for_run is not None:
            print(
                f"[提示] split='{eval_split_for_run}' 没有可评估样本，自动回退为不过滤 split 评估。"
            )
            eval_split_for_run = None
            report = run_evaluation(
                checkpoint=best_path,
                annotations=eval_annotations,
                image_root=eval_image_root,
                split=eval_split_for_run,
                batch_size=eval_batch_size,
                device=device,
                target_dims=target_dims,
            )
        else:
            raise

    raw_eval_path = out_dir / "eval_report.json"
    raw_eval_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    cn_summary = _build_cn_summary(
        cfg=cfg,
        best_path=best_path,
        history_path=history_path,
        raw_eval_path=raw_eval_path,
        report=report,
        eval_split=eval_split_for_run,
        eval_batch_size=eval_batch_size,
        split_meta=split_meta,
        eval_annotations=eval_annotations,
        target_dims=target_dims,
    )
    cn_eval_path = out_dir / "train_eval_report_cn.json"
    cn_eval_path.write_text(json.dumps(cn_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    _print_eval_result(report, target_dims)
    print("\n评估报告(原始):", raw_eval_path)
    print("评估报告(中文):", cn_eval_path)


if __name__ == "__main__":
    main()
