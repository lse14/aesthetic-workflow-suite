import argparse
import copy
import csv
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers.utils import logging as hf_transformers_logging

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
hf_transformers_logging.set_verbosity_error()

TARGETS = ("aesthetic", "composition", "color", "sexual")
DEFAULTS: dict[str, Any] = {
    "inference": {
        "checkpoint": "",
        "input_dir": "data/infer_images",
        "output_dir": "outputs/infer_run",
        "recursive": True,
        "image_extensions": [".jpg", ".jpeg", ".png", ".webp", ".bmp"],
        "batch_size": 8,
        "device": None,
        "special_threshold": 0.5,
        "save_jsonl": True,
        "save_csv": True,
        "jsonl_name": "predictions.jsonl",
        "csv_name": "predictions.csv",
        "organize": {
            "enabled": True,
            "root_dir": "outputs/infer_run/organized",
            "mode": "copy",
            "include_special_group": True,
            "dimensions": ["aesthetic", "composition", "color", "sexual"],
            "bucket_strategy": "nearest_int",
        },
    }
}


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _apply_overrides(cfg: dict[str, Any], overrides: dict[str, object]) -> None:
    for raw_key, value in overrides.items():
        key = str(raw_key)
        if "." not in key:
            cfg["inference"][key] = value
            continue
        parts = key.split(".")
        cur: dict[str, Any] = cfg["inference"]
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = value


def _resolve_path(base_dir: Path, path_like: str | None) -> Path | None:
    if path_like is None:
        return None
    raw = str(path_like).strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def _guess_input_dir(checkpoint: Path, preferred: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if preferred is not None:
        candidates.append(preferred)
    candidates.extend(
        [
            (ROOT / "data" / "infer_images").resolve(),
            (ROOT / "images").resolve(),
            (checkpoint.parent / "images").resolve(),
        ]
    )
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    return None


def _guess_output_dir(checkpoint: Path) -> Path:
    return (checkpoint.parent / "infer_run").resolve()


def load_config(config_path: Path, overrides: dict[str, object] | None = None) -> tuple[dict[str, Any], Path]:
    config_path = config_path.resolve()
    config_dir = config_path.parent
    cfg = copy.deepcopy(DEFAULTS)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, user_cfg)
    else:
        logging.warning("Config file not found, use defaults: %s", config_path)

    if "inference" not in cfg or not isinstance(cfg["inference"], dict):
        raise ValueError("config must contain 'inference' mapping")
    if overrides:
        _apply_overrides(cfg, overrides)

    inf = cfg["inference"]
    resolved_ckpt = _resolve_path(config_dir, str(inf.get("checkpoint", "")))
    resolved_input = _resolve_path(config_dir, str(inf.get("input_dir", "")))
    resolved_output = _resolve_path(config_dir, str(inf.get("output_dir", "")))
    inf["checkpoint"] = str(resolved_ckpt) if resolved_ckpt is not None else ""
    inf["input_dir"] = str(resolved_input) if resolved_input is not None else ""
    inf["output_dir"] = str(resolved_output) if resolved_output is not None else ""
    org = inf.setdefault("organize", {})
    org_root = org.get("root_dir")
    if org_root:
        org["root_dir"] = str(_resolve_path(config_dir, str(org_root)))
    else:
        org["root_dir"] = str(Path(inf["output_dir"]) / "organized")
    return cfg, config_path


def _collect_images(input_dir: Path, recursive: bool, exts: list[str]) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    ext_set = {x.lower() if x.startswith(".") else f".{x.lower()}" for x in exts}
    pattern = "**/*" if recursive else "*"
    out = []
    for p in input_dir.glob(pattern):
        if p.is_file() and p.suffix.lower() in ext_set:
            out.append(p.resolve())
    out.sort()
    return out


def _batched(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _score_bucket(value: float, strategy: str) -> int:
    s = strategy.strip().lower()
    if s == "floor":
        bucket = int(value // 1)
    elif s == "ceil":
        bucket = int(-(-value // 1))
    else:
        bucket = int(round(value))
    return max(1, min(5, bucket))


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 1
    while True:
        cand = parent / f"{stem}__{idx}{suffix}"
        if not cand.exists():
            return cand
        idx += 1


def _place_file(src: Path, dst: Path, mode: str) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst = _next_available_path(dst)
    mode = mode.lower()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"Unsupported organize.mode: {mode}")
    return dst


def _load_checkpoint(path: Path) -> dict[str, Any]:
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
    hidden_dims = json.loads(hidden_dims_raw) if hidden_dims_raw else json.loads(metadata.get("hidden_dims", "[]"))

    return {
        "input_dim": int(metadata.get("input_dim", 0)),
        "hidden_dims": list(hidden_dims),
        "dropout": float(metadata.get("dropout", 0.2)),
        "fusion_head": state,
        "config": json.loads(config_json),
    }


def _resolve_waifu_head_path(raw_path: object, checkpoint: Path) -> str | None:
    def _norm(v: object) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        if not s or s.lower() in {"none", "null", "off", "false", "0"}:
            return None
        return s

    preferred = _norm(raw_path)
    env_override = _norm(os.getenv("FUSION_WAIFU_V3_HEAD_PATH"))

    candidates: list[Path] = []

    def _add(path_like: str | None) -> None:
        if not path_like:
            return
        p = Path(path_like).expanduser()
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append((ROOT / p).resolve())
            candidates.append((checkpoint.parent / p).resolve())

    cache_root = _norm(os.getenv("FUSION_MODEL_CACHE_ROOT"))
    default_local = (
        (Path(cache_root).expanduser() / "waifu-scorer-v3" / "model.safetensors").resolve()
        if cache_root
        else (ROOT / "_models" / "waifu-scorer-v3" / "model.safetensors").resolve()
    )
    if cache_root:
        candidates.append((Path(cache_root).expanduser() / "waifu-scorer-v3" / "model.safetensors").resolve())
    candidates.append((ROOT / "_models" / "waifu-scorer-v3" / "model.safetensors").resolve())
    candidates.append((ROOT.parent / "model" / "_models" / "waifu-scorer-v3" / "model.safetensors").resolve())
    candidates.append((checkpoint.parent / "waifu-scorer-v3" / "model.safetensors").resolve())
    candidates.append((checkpoint.parent / "_models" / "waifu-scorer-v3" / "model.safetensors").resolve())
    # Keep overrides last to avoid leaking training-machine absolute path as final error path.
    _add(env_override)
    _add(preferred)

    seen: set[str] = set()
    for c in candidates:
        k = str(c).lower()
        if k in seen:
            continue
        seen.add(k)
        if c.exists() and c.is_file():
            if preferred and str(c).lower() != str(Path(preferred).expanduser()).lower():
                logging.warning("waifu_v3_head_path not found in checkpoint config, auto-resolved to: %s", c)
            return str(c)

    if preferred:
        logging.warning(
            "waifu_v3_head_path from checkpoint/env is unavailable (%s). "
            "Will use local default path for diagnostics: %s",
            preferred,
            default_local,
        )
    return str(default_local)


_ABS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _resolve_model_ref(raw_value: object, *, default_value: str, checkpoint: Path, allow_none: bool = False) -> str | None:
    raw = "" if raw_value is None else str(raw_value).strip()
    if not raw:
        return None if allow_none else default_value
    low = raw.lower()
    if allow_none and low in {"none", "null", "off", "false", "0"}:
        return None

    is_path_like = (
        raw.startswith(".")
        or raw.startswith("/")
        or raw.startswith("\\")
        or ("\\" in raw)
        or bool(_ABS_PATH_RE.match(raw))
    )
    if not is_path_like:
        return raw

    p = Path(raw).expanduser()
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((ROOT / p).resolve())
        candidates.append((checkpoint.parent / p).resolve())

    for c in candidates:
        if c.exists():
            return str(c)

    logging.warning(
        "Ignoring unavailable local model path from checkpoint: %s ; fallback to %s",
        raw,
        "none" if (allow_none and default_value == "none") else default_value,
    )
    if allow_none and default_value == "none":
        return None
    return default_value


def _load_runtime(checkpoint: Path, device_override: str | None):
    from fusion_scorer.extractors import JTP3FeatureExtractor, WaifuV3ClipFeatureExtractor
    from fusion_scorer.model import FusionMultiTaskHead

    ckpt = _load_checkpoint(checkpoint)
    cfg = ckpt.get("config") or {}
    runtime_device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")
    models = cfg.get("models") or {}
    expected_input_dim = int(ckpt["input_dim"])

    configured_model_id = _resolve_model_ref(
        os.getenv("FUSION_JTP3_MODEL_ID") or models.get("jtp3_model_id", "RedRocket/JTP-3"),
        default_value="RedRocket/JTP-3",
        checkpoint=checkpoint,
        allow_none=False,
    )
    configured_fallback = (
        _resolve_model_ref(
            os.getenv("FUSION_JTP3_FALLBACK_MODEL_ID")
            if os.getenv("FUSION_JTP3_FALLBACK_MODEL_ID") is not None
            else models.get("jtp3_fallback_model_id", "google/siglip2-so400m-patch16-naflex"),
            default_value="google/siglip2-so400m-patch16-naflex",
            checkpoint=checkpoint,
            allow_none=True,
        )
    )
    hf_token_env = models.get("hf_token_env", "HF_TOKEN")
    resolved_waifu_head = _resolve_waifu_head_path(models.get("waifu_v3_head_path"), checkpoint)

    waifu = WaifuV3ClipFeatureExtractor(
        clip_model_name=models.get("waifu_clip_model_name", "ViT-L-14"),
        clip_pretrained=models.get("waifu_clip_pretrained", "openai"),
        waifu_head_path=resolved_waifu_head,
        device=runtime_device,
        freeze=True,
        include_waifu_score=bool(models.get("include_waifu_score", True)),
    )

    probe_images = [Image.new("RGB", (224, 224), (0, 0, 0))]
    with torch.no_grad():
        waifu_dim = int(waifu(probe_images).shape[-1])

    def _build_jtp(model_id: str, fallback_model_id: str | None):
        return JTP3FeatureExtractor(
            model_id=model_id,
            device=runtime_device,
            hf_token_env=hf_token_env,
            freeze=True,
            fallback_model_id=fallback_model_id,
        )

    def _probe_jtp_dim(jtp_extractor) -> int:
        with torch.no_grad():
            return int(jtp_extractor(probe_images).shape[-1])

    tried: list[str] = []
    jtp = _build_jtp(configured_model_id, configured_fallback)
    jtp_dim = _probe_jtp_dim(jtp)
    fused_dim = jtp_dim + waifu_dim
    tried.append(
        f"{configured_model_id} => loaded={getattr(jtp, 'loaded_model_id', configured_model_id)} "
        f"(fallback={configured_fallback}) -> fused_dim={fused_dim}"
    )

    if fused_dim != expected_input_dim:
        logging.warning(
            "Feature dim mismatch for checkpoint=%s: expected=%s, got=%s. Auto-trying fallback model ids.",
            checkpoint,
            expected_input_dim,
            fused_dim,
        )
        candidates: list[tuple[str, str | None]] = []
        if configured_fallback and str(configured_fallback).strip() and str(configured_fallback) != configured_model_id:
            candidates.append((str(configured_fallback), None))
        default_fallback = "google/siglip2-so400m-patch16-naflex"
        if default_fallback not in {configured_model_id, str(configured_fallback)}:
            candidates.append((default_fallback, None))

        matched = False
        for cand_model_id, cand_fallback in candidates:
            try:
                cand_jtp = _build_jtp(cand_model_id, cand_fallback)
                cand_jtp_dim = _probe_jtp_dim(cand_jtp)
                cand_fused_dim = cand_jtp_dim + waifu_dim
                tried.append(
                    f"{cand_model_id} => loaded={getattr(cand_jtp, 'loaded_model_id', cand_model_id)} "
                    f"(fallback={cand_fallback}) -> fused_dim={cand_fused_dim}"
                )
                if cand_fused_dim == expected_input_dim:
                    logging.warning(
                        "Auto-switched JTP extractor to '%s' for dimension compatibility (fused_dim=%s).",
                        cand_model_id,
                        cand_fused_dim,
                    )
                    jtp = cand_jtp
                    jtp_dim = cand_jtp_dim
                    fused_dim = cand_fused_dim
                    matched = True
                    break
            except Exception as e:
                tried.append(f"{cand_model_id} (fallback={cand_fallback}) -> error={e}")
                continue

        if not matched:
            raise RuntimeError(
                "Checkpoint/extractor feature dimension mismatch. "
                f"checkpoint_input_dim={expected_input_dim}, current_fused_dim={fused_dim}, waifu_dim={waifu_dim}, jtp_dim={jtp_dim}. "
                f"Tried: {' | '.join(tried)}"
            )

    head = FusionMultiTaskHead(
        input_dim=expected_input_dim,
        hidden_dims=list(ckpt["hidden_dims"]),
        dropout=float(ckpt["dropout"]),
    ).to(runtime_device)

    state = dict(ckpt["fusion_head"])
    if any(k.startswith("heads.") for k in state.keys()) and not any(
        k.startswith("reg_heads.") for k in state.keys()
    ):
        mapped: dict[str, Any] = {}
        for k, v in state.items():
            if k.startswith("heads."):
                mapped["reg_heads." + k[len("heads.") :]] = v
            else:
                mapped[k] = v
        state = mapped
    has_cls_head = any(k.startswith("cls_head.") for k in state.keys())
    head.load_state_dict(state, strict=False)
    head.eval()

    return {
        "checkpoint": checkpoint,
        "config": cfg,
        "device": runtime_device,
        "jtp": jtp,
        "waifu": waifu,
        "head": head,
        "has_cls_head": bool(has_cls_head),
    }


def _infer_records(
    image_paths: list[Path],
    *,
    input_dir: Path,
    runtime: dict[str, Any],
    batch_size: int,
    special_threshold: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    jtp = runtime["jtp"]
    waifu = runtime["waifu"]
    head = runtime["head"]
    has_cls_head = bool(runtime["has_cls_head"])

    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    for batch in tqdm(_batched(image_paths, batch_size), total=total_batches, desc="infer", unit="batch"):
        valid_paths: list[Path] = []
        images: list[Image.Image] = []
        for p in batch:
            try:
                with Image.open(p) as img:
                    images.append(img.convert("RGB"))
                valid_paths.append(p)
            except Exception as e:
                records.append(
                    {
                        "image_path": str(p),
                        "relative_path": str(p.relative_to(input_dir)) if p.is_relative_to(input_dir) else p.name,
                        "aesthetic": None,
                        "composition": None,
                        "color": None,
                        "sexual": None,
                        "in_domain_prob": None,
                        "in_domain_pred": None,
                        "special_tag": None,
                        "special_reason": "",
                        "error": f"image_load_failed: {e}",
                    }
                )
        if not valid_paths:
            continue

        with torch.no_grad():
            f1 = jtp(images)
            f2 = waifu(images)
            reg_pred, cls_logit = head(torch.cat([f1, f2], dim=-1))
            reg_list = reg_pred.cpu().tolist()
            cls_probs = torch.sigmoid(cls_logit).cpu().tolist() if has_cls_head else [None] * len(valid_paths)

        for p, reg_row, cls_prob in zip(valid_paths, reg_list, cls_probs):
            scores = [float(x) for x in reg_row]
            if cls_prob is None:
                in_domain_prob = None
                in_domain_pred = 1
                special_tag = 0
                special_reason = "no_cls_head"
            else:
                in_domain_prob = float(cls_prob)
                in_domain_pred = 1 if in_domain_prob >= special_threshold else 0
                special_tag = 0 if in_domain_pred == 1 else 1
                special_reason = "prob_below_threshold" if special_tag == 1 else ""

            rel = str(p.relative_to(input_dir)) if p.is_relative_to(input_dir) else p.name
            records.append(
                {
                    "image_path": str(p),
                    "relative_path": rel,
                    "aesthetic": scores[0],
                    "composition": scores[1],
                    "color": scores[2],
                    "sexual": scores[3],
                    "in_domain_prob": in_domain_prob,
                    "in_domain_pred": in_domain_pred,
                    "special_tag": special_tag,
                    "special_reason": special_reason,
                    "error": "",
                }
            )
    return records


def _write_outputs(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    save_jsonl: bool,
    save_csv: bool,
    jsonl_name: str,
    csv_name: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    if save_jsonl:
        p = output_dir / jsonl_name
        with p.open("w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out["jsonl"] = p

    if save_csv:
        p = output_dir / csv_name
        fieldnames = [
            "image_path",
            "relative_path",
            "aesthetic",
            "composition",
            "color",
            "sexual",
            "in_domain_prob",
            "in_domain_pred",
            "special_tag",
            "special_reason",
            "error",
        ]
        with p.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in records:
                w.writerow(row)
        out["csv"] = p
    return out


def _organize_images(
    records: list[dict[str, Any]],
    *,
    input_dir: Path,
    organize_cfg: dict[str, Any],
) -> dict[str, int]:
    enabled = bool(organize_cfg.get("enabled", False))
    if not enabled:
        return {"organized": 0, "failed": 0}

    root_dir = Path(str(organize_cfg.get("root_dir", ""))).resolve()
    mode = str(organize_cfg.get("mode", "copy")).strip().lower()
    include_special_group = bool(organize_cfg.get("include_special_group", True))
    dimensions = [x for x in organize_cfg.get("dimensions", list(TARGETS)) if x in TARGETS]
    strategy = str(organize_cfg.get("bucket_strategy", "nearest_int"))

    if not dimensions:
        raise ValueError("organize.dimensions must contain at least one valid dimension")
    if mode == "move" and len(dimensions) > 1:
        logging.warning("organize.mode=move with multiple dimensions is not safe. Fallback to copy.")
        mode = "copy"

    ok = 0
    failed = 0
    for row in tqdm(records, desc="organize", unit="img"):
        if row.get("error"):
            continue
        src = Path(str(row["image_path"]))
        if not src.exists():
            failed += 1
            continue

        cls_group = "special" if int(row.get("special_tag") or 0) == 1 else "in_domain"
        for dim in dimensions:
            score = row.get(dim)
            if score is None:
                continue
            bucket = _score_bucket(float(score), strategy)
            base = root_dir / cls_group if include_special_group else root_dir
            dst_dir = base / dim / f"score_{bucket}"
            try:
                rel = src.relative_to(input_dir) if src.is_relative_to(input_dir) else Path(src.name)
                dst = dst_dir / rel
                _place_file(src, dst, mode)
                ok += 1
            except Exception:
                failed += 1
    return {"organized": ok, "failed": failed}


def run_from_config(config_path: Path, overrides: dict[str, object] | None = None) -> dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )
    cfg, config_path = load_config(config_path, overrides=overrides)
    inf = cfg["inference"]

    checkpoint = Path(str(inf["checkpoint"])).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    input_raw = str(inf.get("input_dir", "") or "").strip()
    preferred_input = Path(input_raw).resolve() if input_raw else None
    input_dir = _guess_input_dir(checkpoint, preferred=preferred_input)
    if input_dir is None:
        raise FileNotFoundError(
            "input_dir not found. Please set inference.input_dir or pass --input-dir."
        )
    if preferred_input is not None and preferred_input != input_dir and not preferred_input.exists():
        logging.warning("Configured input_dir not found: %s; fallback to %s", preferred_input, input_dir)

    output_raw = str(inf.get("output_dir", "") or "").strip()
    output_dir = Path(output_raw).resolve() if output_raw else _guess_output_dir(checkpoint)

    recursive = bool(inf.get("recursive", True))
    exts = list(inf.get("image_extensions", [".jpg", ".jpeg", ".png", ".webp", ".bmp"]))
    batch_size = int(inf.get("batch_size", 8))
    special_threshold = float(inf.get("special_threshold", 0.5))

    images = _collect_images(input_dir, recursive=recursive, exts=exts)
    if not images:
        raise RuntimeError(f"No images found in input_dir={input_dir}")

    runtime = _load_runtime(checkpoint, inf.get("device"))
    logging.info("config=%s", config_path)
    logging.info("checkpoint=%s", checkpoint)
    logging.info("device=%s", runtime["device"])
    logging.info("images=%d", len(images))
    logging.info("has_cls_head=%s", runtime["has_cls_head"])
    logging.info("special_threshold=%.4f", special_threshold)

    records = _infer_records(
        images,
        input_dir=input_dir,
        runtime=runtime,
        batch_size=batch_size,
        special_threshold=special_threshold,
    )
    out_files = _write_outputs(
        records,
        output_dir=output_dir,
        save_jsonl=bool(inf.get("save_jsonl", True)),
        save_csv=bool(inf.get("save_csv", True)),
        jsonl_name=str(inf.get("jsonl_name", "predictions.jsonl")),
        csv_name=str(inf.get("csv_name", "predictions.csv")),
    )
    organize_stats = _organize_images(
        records,
        input_dir=input_dir,
        organize_cfg=dict(inf.get("organize", {})),
    )

    total = len(records)
    infer_ok = sum(1 for r in records if not r.get("error"))
    special_n = sum(1 for r in records if int(r.get("special_tag") or 0) == 1)
    summary = {
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_records": total,
        "inferred_records": infer_ok,
        "special_records": special_n,
        "has_cls_head": bool(runtime["has_cls_head"]),
        "special_threshold": special_threshold,
        "output_files": {k: str(v) for k, v in out_files.items()},
        "organize": organize_stats,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("summary=%s", summary_path)
    for k, v in out_files.items():
        logging.info("%s=%s", k, v)
    logging.info("organize: %s", organize_stats)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch inference + auto-tagging + folder organization."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--special-threshold", type=float, default=None)
    parser.add_argument("--organize", choices=["auto", "on", "off"], default="auto")
    args = parser.parse_args()

    overrides: dict[str, object] = {}
    if args.checkpoint is not None:
        overrides["checkpoint"] = args.checkpoint
    if args.input_dir is not None:
        overrides["input_dir"] = args.input_dir
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.device is not None:
        overrides["device"] = args.device
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.special_threshold is not None:
        overrides["special_threshold"] = args.special_threshold
    if args.organize == "on":
        overrides["organize.enabled"] = True
    elif args.organize == "off":
        overrides["organize.enabled"] = False

    run_from_config(args.config, overrides=overrides)


if __name__ == "__main__":
    main()
