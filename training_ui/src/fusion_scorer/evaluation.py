import csv
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

TARGETS = ("aesthetic", "composition", "color", "sexual")
TARGET_ALIASES = {
    "aesthetic": ("aesthetic", "ann_aesthetic"),
    "composition": ("composition", "ann_composition"),
    "color": ("color", "ann_color"),
    "sexual": ("sexual", "ann_sexual"),
}


def load_records(annotation_file: Path) -> list[dict]:
    if annotation_file.suffix.lower() == ".jsonl":
        rows = []
        with annotation_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if annotation_file.suffix.lower() == ".csv":
        with annotation_file.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    if annotation_file.suffix.lower() == ".db":
        conn = sqlite3.connect(str(annotation_file))
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            has_samples = cur.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='samples'"
            ).fetchone()
            has_annotations = cur.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='annotations'"
            ).fetchone()
            if not has_samples or not has_annotations:
                raise ValueError("SQLite schema mismatch: missing samples/annotations tables")
            rows = cur.execute(
                """
                SELECT
                    s.id AS id,
                    s.source AS source,
                    s.source_post_id AS source_post_id,
                    s.source_page_url AS source_page_url,
                    s.local_path AS local_path,
                    a.aesthetic AS aesthetic,
                    a.composition AS composition,
                    a.color AS color,
                    a.sexual AS sexual,
                    a.in_domain AS in_domain,
                    a.content_type AS content_type,
                    a.exclude_from_train AS exclude_from_train,
                    a.exclude_from_score_train AS exclude_from_score_train,
                    a.exclude_from_cls_train AS exclude_from_cls_train,
                    a.exclude_reason AS exclude_reason
                FROM samples s
                JOIN annotations a ON a.sample_id = s.id
                WHERE a.status = 'labeled'
                ORDER BY s.id ASC
                """
            ).fetchall()
            out: list[dict] = []
            for r in rows:
                item = {k: r[k] for k in r.keys()}
                item["image_path"] = item.get("local_path")
                out.append(item)
            return out
        finally:
            conn.close()
    raise ValueError("annotation file must be .jsonl, .csv or .db")


def _parse_binary(raw, *, default: int = 0) -> int:
    if raw in (None, ""):
        return int(default)
    try:
        iv = int(raw)
    except Exception:
        return int(default)
    return 1 if iv == 1 else 0


def _pick(rec: dict, keys: tuple[str, ...], default=None):
    for k in keys:
        if k in rec and rec.get(k) not in (None, ""):
            return rec.get(k)
    return default


def _normalize_split(raw) -> str | None:
    if raw in (None, ""):
        return None
    s = str(raw).strip()
    return s or None


def _parse_float(raw, default: float | None = None) -> float | None:
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _resolve_image_path(image_path: Path, image_root: Path | None) -> Path:
    if image_path.is_absolute() or image_root is None:
        return image_path
    candidate = image_root / image_path
    if candidate.exists():
        return candidate

    iparts = image_path.parts
    rparts = image_root.parts
    max_k = min(len(iparts), len(rparts))
    for k in range(max_k, 0, -1):
        left = tuple(p.lower() for p in iparts[:k])
        right = tuple(p.lower() for p in rparts[-k:])
        if left == right:
            suffix = Path(*iparts[k:]) if k < len(iparts) else Path()
            return image_root / suffix
    return candidate


def rankdata(vals):
    idx = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i + 1
        while j < len(vals) and vals[idx[j]] == vals[idx[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[idx[k]] = avg_rank
        i = j
    return ranks


def corr_pearson(x, y):
    n = len(x)
    if n < 2:
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx <= 0 or vy <= 0:
        return float("nan")
    c = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return c / math.sqrt(vx * vy)


def corr_spearman(x, y):
    return corr_pearson(rankdata(x), rankdata(y))


def summarize_regression(pred_rows: list[list[float]], tgt_rows: list[list[float]]) -> dict:
    n = len(pred_rows)
    if n == 0:
        return {
            "n": 0,
            "overall_mae": float("nan"),
            "overall_rmse": float("nan"),
            "mean_dim_mae": float("nan"),
            "mean_dim_rmse": float("nan"),
            "per_dim": {},
        }
    per_dim = {}
    maes = []
    rmses = []
    for di, name in enumerate(TARGETS):
        p = [float(r[di]) for r in pred_rows]
        t = [float(r[di]) for r in tgt_rows]
        abs_err = [abs(a - b) for a, b in zip(p, t)]
        sq_err = [(a - b) ** 2 for a, b in zip(p, t)]
        mae = sum(abs_err) / n
        rmse = math.sqrt(sum(sq_err) / n)
        rho = corr_spearman(p, t)
        per_dim[name] = {
            "mae": mae,
            "rmse": rmse,
            "spearman": rho,
        }
        maes.append(mae)
        rmses.append(rmse)

    dim_n = len(TARGETS)
    all_abs = [abs(pred_rows[i][j] - tgt_rows[i][j]) for i in range(n) for j in range(dim_n)]
    all_sq = [(pred_rows[i][j] - tgt_rows[i][j]) ** 2 for i in range(n) for j in range(dim_n)]
    return {
        "n": n,
        "overall_mae": sum(all_abs) / len(all_abs),
        "overall_rmse": math.sqrt(sum(all_sq) / len(all_sq)),
        "mean_dim_mae": sum(maes) / len(maes),
        "mean_dim_rmse": sum(rmses) / len(rmses),
        "per_dim": per_dim,
    }


def summarize_classification(probs: list[float], targets: list[int]) -> dict:
    n = len(probs)
    if n == 0:
        return {
            "n": 0,
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "pos_rate": float("nan"),
            "pred_pos_rate": float("nan"),
        }
    preds = [1 if float(p) >= 0.5 else 0 for p in probs]
    tp = sum(1 for p, t in zip(preds, targets) if p == 1 and t == 1)
    tn = sum(1 for p, t in zip(preds, targets) if p == 0 and t == 0)
    fp = sum(1 for p, t in zip(preds, targets) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(preds, targets) if p == 0 and t == 1)

    accuracy = (tp + tn) / n
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if (precision + recall) <= 0 else (2 * precision * recall) / (precision + recall)
    pos_rate = sum(targets) / n
    pred_pos_rate = sum(preds) / n
    return {
        "n": n,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pos_rate": float(pos_rate),
        "pred_pos_rate": float(pred_pos_rate),
    }


def _load_checkpoint(path: Path) -> dict:
    import torch

    if path.suffix.lower() != ".safetensors":
        return torch.load(path, map_location="cpu")

    from safetensors import safe_open
    from safetensors.torch import load_file as load_safetensors_file

    state = load_safetensors_file(str(path), device="cpu")
    with safe_open(str(path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}

    config_json = metadata.get("config_json")
    if not config_json:
        raise ValueError(f"safetensors checkpoint missing required metadata: config_json ({path})")

    hidden_dims_raw = metadata.get("hidden_dims_json")
    if hidden_dims_raw:
        hidden_dims = json.loads(hidden_dims_raw)
    else:
        hidden_dims = json.loads(metadata.get("hidden_dims", "[]"))

    def _as_float(name: str, default: float = float("nan")) -> float:
        raw = metadata.get(name)
        if raw in (None, ""):
            return float(default)
        try:
            return float(raw)
        except Exception:
            return float(default)

    def _as_int(name: str, default: int = 0) -> int:
        raw = metadata.get(name)
        if raw in (None, ""):
            return int(default)
        try:
            return int(raw)
        except Exception:
            return int(default)

    return {
        "epoch": _as_int("epoch", 0),
        "val_mae": _as_float("val_mae"),
        "val_loss": _as_float("val_loss"),
        "val_cls_acc": _as_float("val_cls_acc"),
        "cls_loss_weight": _as_float("cls_loss_weight"),
        "head_type": metadata.get("format", "fusion_multitask_v1"),
        "input_dim": _as_int("input_dim", 0),
        "hidden_dims": list(hidden_dims),
        "dropout": _as_float("dropout", 0.2),
        "fusion_head": state,
        "config": json.loads(config_json),
    }


def run_evaluation(
    *,
    checkpoint: Path,
    annotations: Path,
    image_root: Path | None = None,
    split: str = "val",
    batch_size: int = 8,
    device: str | None = None,
    target_dims: list[str] | tuple[str, ...] | None = None,
) -> dict:
    import torch
    from PIL import Image

    from fusion_scorer.extractors import JTP3FeatureExtractor, WaifuV3ClipFeatureExtractor
    from fusion_scorer.model import FusionMultiTaskHead

    ckpt = _load_checkpoint(Path(checkpoint))
    cfg = ckpt["config"]
    runtime_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if target_dims is None:
        selected_dims = list(TARGETS)
    else:
        selected_dims = [str(x).strip().lower() for x in target_dims if str(x).strip()]
        selected_dims = [x for x in selected_dims if x in TARGETS]
        if not selected_dims:
            selected_dims = list(TARGETS)
    selected_idx = [TARGETS.index(x) for x in selected_dims]

    rows = load_records(annotations)
    split_keys = ("split", "subset", "dataset_split")
    has_any_split_key = any(_normalize_split(_pick(r, split_keys)) is not None for r in rows)
    use_split = bool(split) and has_any_split_key
    filtered = []
    for r in rows:
        if use_split and _normalize_split(_pick(r, split_keys)) != split:
            continue
        p = Path(str(_pick(r, ("image_path", "path", "local_path", "relative_path"))))
        p = _resolve_image_path(p, image_root)

        score_targets: list[float] = []
        score_dim_mask: list[int] = []
        for name in TARGETS:
            raw_val = _pick(r, TARGET_ALIASES[name], default=None)
            fv = _parse_float(raw_val, default=None)
            if fv is None:
                score_targets.append(0.0)
                score_dim_mask.append(0)
            else:
                score_targets.append(float(fv))
                score_dim_mask.append(1)

        legacy_exclude = _parse_binary(
            _pick(r, ("exclude_from_train", "ann_exclude_from_train")),
            default=0,
        )
        exclude_score = _parse_binary(
            _pick(r, ("exclude_from_score_train", "ann_exclude_from_score_train")),
            default=legacy_exclude,
        )
        exclude_cls = _parse_binary(
            _pick(r, ("exclude_from_cls_train", "ann_exclude_from_cls_train")),
            default=legacy_exclude,
        )
        if exclude_score == 1:
            score_dim_mask = [0 for _ in TARGETS]
        use_score = bool(any(score_dim_mask[idx] == 1 for idx in selected_idx))
        in_domain_raw = _pick(r, ("in_domain", "ann_in_domain", "in_domain_pred"), default=None)
        if in_domain_raw in (None, "") and r.get("special_tag") not in (None, ""):
            in_domain_raw = 0 if _parse_binary(r.get("special_tag"), default=0) == 1 else 1
        use_cls = bool(exclude_cls == 0 and in_domain_raw not in (None, ""))

        if not use_score and not use_cls:
            continue
        filtered.append(
            {
                "image_path": p,
                "source": str(r.get("source", "unknown")),
                "score_targets": score_targets,
                "score_dim_mask": score_dim_mask,
                "cls_target": _parse_binary(in_domain_raw, default=1),
                "use_score": use_score,
                "use_cls": use_cls,
            }
        )
    if not filtered:
        raise RuntimeError("No rows after split filter.")

    jtp = JTP3FeatureExtractor(
        model_id=cfg["models"]["jtp3_model_id"],
        device=runtime_device,
        hf_token_env=cfg["models"].get("hf_token_env", "HF_TOKEN"),
        freeze=True,
        fallback_model_id=cfg["models"].get(
            "jtp3_fallback_model_id",
            None,
        ),
    )
    waifu = WaifuV3ClipFeatureExtractor(
        clip_model_name=cfg["models"].get("waifu_clip_model_name", "ViT-L-14"),
        clip_pretrained=cfg["models"].get("waifu_clip_pretrained", "openai"),
        waifu_head_path=cfg["models"].get("waifu_v3_head_path"),
        device=runtime_device,
        freeze=True,
        include_waifu_score=bool(cfg["models"].get("include_waifu_score", True)),
    )
    head = FusionMultiTaskHead(
        input_dim=int(ckpt["input_dim"]),
        hidden_dims=ckpt["hidden_dims"],
        dropout=float(ckpt["dropout"]),
    ).to(runtime_device)
    state = dict(ckpt["fusion_head"])
    if any(k.startswith("heads.") for k in state.keys()) and not any(
        k.startswith("reg_heads.") for k in state.keys()
    ):
        mapped = {}
        for k, v in state.items():
            if k.startswith("heads."):
                mapped["reg_heads." + k[len("heads.") :]] = v
            else:
                mapped[k] = v
        state = mapped
    has_cls_head = any(k.startswith("cls_head.") for k in state.keys())
    head.load_state_dict(state, strict=False)
    head.eval()

    reg_dim_pred: dict[str, list[float]] = {name: [] for name in selected_dims}
    reg_dim_tgt: dict[str, list[float]] = {name: [] for name in selected_dims}
    cls_probs: list[float] = []
    cls_targs: list[int] = []

    reg_by_source: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    cls_by_source: dict[str, list[tuple[float, int]]] = defaultdict(list)

    for i in range(0, len(filtered), batch_size):
        batch = filtered[i : i + batch_size]
        images = [Image.open(x["image_path"]).convert("RGB") for x in batch]
        with torch.no_grad():
            f1 = jtp(images)
            f2 = waifu(images)
            reg_pred, cls_logit = head(torch.cat([f1, f2], dim=-1))
            reg_pred = reg_pred.cpu().tolist()
            cls_prob = torch.sigmoid(cls_logit).cpu().tolist()

        for b, pred, prob in zip(batch, reg_pred, cls_prob):
            if b["use_score"]:
                pp = [float(x) for x in pred]
                tt = [float(x) for x in b["score_targets"]]
                dm = b.get("score_dim_mask", [1] * len(TARGETS))
                for name, di in zip(selected_dims, selected_idx):
                    if di < len(dm) and int(dm[di]) == 1:
                        reg_dim_pred[name].append(pp[di])
                        reg_dim_tgt[name].append(tt[di])
                        reg_by_source[b["source"]][name].append((pp[di], tt[di]))
            if b["use_cls"] and has_cls_head:
                p = float(prob)
                t = int(b["cls_target"])
                cls_probs.append(p)
                cls_targs.append(t)
                cls_by_source[b["source"]].append((p, t))

    # Build overall regression summary from selected dimensions only.
    per_dim = {}
    all_abs = []
    all_sq = []
    maes = []
    rmses = []
    for name in selected_dims:
        p = reg_dim_pred.get(name, [])
        t = reg_dim_tgt.get(name, [])
        n = len(p)
        if n <= 0:
            per_dim[name] = {"mae": float("nan"), "rmse": float("nan"), "spearman": float("nan")}
            continue
        abs_err = [abs(a - b) for a, b in zip(p, t)]
        sq_err = [(a - b) ** 2 for a, b in zip(p, t)]
        mae = sum(abs_err) / n
        rmse = math.sqrt(sum(sq_err) / n)
        rho = corr_spearman(p, t)
        per_dim[name] = {"mae": float(mae), "rmse": float(rmse), "spearman": float(rho)}
        maes.append(mae)
        rmses.append(rmse)
        all_abs.extend(abs_err)
        all_sq.extend(sq_err)
    if all_abs:
        overall_reg = {
            "n": len(all_abs),
            "overall_mae": float(sum(all_abs) / len(all_abs)),
            "overall_rmse": float(math.sqrt(sum(all_sq) / len(all_sq))),
            "mean_dim_mae": float(sum(maes) / max(len(maes), 1)),
            "mean_dim_rmse": float(sum(rmses) / max(len(rmses), 1)),
            "per_dim": per_dim,
        }
    else:
        overall_reg = {
            "n": 0,
            "overall_mae": float("nan"),
            "overall_rmse": float("nan"),
            "mean_dim_mae": float("nan"),
            "mean_dim_rmse": float("nan"),
            "per_dim": per_dim,
        }
    overall_cls = summarize_classification(cls_probs, cls_targs)

    sources = sorted(set(list(reg_by_source.keys()) + list(cls_by_source.keys())))
    by_source: dict[str, dict] = {}
    for s in sources:
        reg_dim_pairs = reg_by_source.get(s, {})
        cls_pairs = cls_by_source.get(s, [])
        per_src_dim = {}
        src_all_abs = []
        src_all_sq = []
        src_maes = []
        src_rmses = []
        for name in selected_dims:
            pairs = reg_dim_pairs.get(name, [])
            if not pairs:
                per_src_dim[name] = {"mae": float("nan"), "rmse": float("nan"), "spearman": float("nan")}
                continue
            rp = [x[0] for x in pairs]
            rt = [x[1] for x in pairs]
            abs_err = [abs(a - b) for a, b in zip(rp, rt)]
            sq_err = [(a - b) ** 2 for a, b in zip(rp, rt)]
            mae = sum(abs_err) / len(abs_err)
            rmse = math.sqrt(sum(sq_err) / len(sq_err))
            rho = corr_spearman(rp, rt)
            per_src_dim[name] = {"mae": float(mae), "rmse": float(rmse), "spearman": float(rho)}
            src_maes.append(mae)
            src_rmses.append(rmse)
            src_all_abs.extend(abs_err)
            src_all_sq.extend(sq_err)
        cp = [x[0] for x in cls_pairs]
        ct = [x[1] for x in cls_pairs]
        if src_all_abs:
            src_reg = {
                "n": len(src_all_abs),
                "overall_mae": float(sum(src_all_abs) / len(src_all_abs)),
                "overall_rmse": float(math.sqrt(sum(src_all_sq) / len(src_all_sq))),
                "mean_dim_mae": float(sum(src_maes) / max(len(src_maes), 1)),
                "mean_dim_rmse": float(sum(src_rmses) / max(len(src_rmses), 1)),
                "per_dim": per_src_dim,
            }
        else:
            src_reg = {
                "n": 0,
                "overall_mae": float("nan"),
                "overall_rmse": float("nan"),
                "mean_dim_mae": float("nan"),
                "mean_dim_rmse": float("nan"),
                "per_dim": per_src_dim,
            }
        src_cls = summarize_classification(cp, ct)
        by_source[s] = {
            **src_reg,
            "classification": src_cls,
        }

    report = {
        "checkpoint": str(checkpoint),
        "annotations": str(annotations),
        "split": split,
        "target_dims": selected_dims,
        "device": runtime_device,
        "has_cls_head": bool(has_cls_head),
        "overall": {
            **overall_reg,
            "classification": overall_cls,
        },
        "classification": overall_cls,
        "by_source": by_source,
    }
    return report
