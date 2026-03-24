import io
import importlib
import json
import logging
import csv
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field
from transformers.utils import logging as hf_transformers_logging


APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
TARGETS = ("aesthetic", "composition", "color", "sexual")
ENV_CHECKPOINT_KEY = "INFER_UI_CHECKPOINT"
DEFAULT_MODEL_CACHE_ROOT = (ROOT / "_models").resolve()
LEGACY_MODEL_CACHE_ROOT = (ROOT.parent / "model" / "_models").resolve()
hf_transformers_logging.set_verbosity_error()
hf_transformers_logging.disable_progress_bar()


class InferStartRequest(BaseModel):
    config_path: str | None = None
    checkpoint: str | None = None
    input_dir: str | None = None
    output_dir: str | None = None
    device: str | None = None
    batch_size: int | None = Field(default=None, ge=1, le=2048)
    special_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    organize: str | None = None  # auto/on/off


class SinglePathInferRequest(BaseModel):
    image_path: str
    config_path: str | None = None
    checkpoint: str | None = None
    device: str | None = None
    special_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


def _log_runtime(log_fn, message: str) -> None:
    if log_fn is None:
        return
    try:
        log_fn(message)
    except Exception:
        pass


def _try_import_torch(force_reload: bool = False):
    try:
        if force_reload:
            for name in list(sys.modules.keys()):
                if name == "torch" or name.startswith("torch."):
                    sys.modules.pop(name, None)
            importlib.invalidate_caches()
        return importlib.import_module("torch"), None
    except Exception as e:
        return None, str(e)


def _detect_cuda_version() -> str | None:
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return None
    try:
        p = subprocess.run(
            [nvsmi],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            check=False,
        )
    except Exception:
        return None
    text = f"{p.stdout}\n{p.stderr}"
    m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", text)
    return m.group(1) if m else None


def _choose_torch_index(cuda_version: str | None) -> str:
    if not cuda_version:
        return "cpu"
    try:
        major, minor = (int(x) for x in cuda_version.split(".", 1))
    except Exception:
        return "cu121"
    if major >= 12:
        return "cu124" if minor >= 4 else "cu121"
    if major == 11 and minor >= 8:
        return "cu118"
    return "cu121"


def _install_torch(index_tag: str, log_fn=None) -> None:
    index_url = f"https://download.pytorch.org/whl/{index_tag}"
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "torch",
        "torchvision",
        "--index-url",
        index_url,
    ]
    _log_runtime(log_fn, f"[infer_ui] 正在安装 PyTorch ({index_tag}) ...")
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if p.returncode != 0:
        tail = "\n".join((p.stdout or "").splitlines()[-10:] + (p.stderr or "").splitlines()[-20:])
        raise RuntimeError(f"安装 torch 失败 (index={index_tag})。\n{tail}")
    _log_runtime(log_fn, f"[infer_ui] PyTorch 安装完成 ({index_tag})")


def _ensure_torch_runtime(prefer_gpu: bool | None, log_fn=None):
    # prefer_gpu: True=必须GPU, False=CPU即可, None=自动
    cuda_version = _detect_cuda_version() if prefer_gpu is not False else None
    torch_mod, import_err = _try_import_torch(force_reload=False)

    if torch_mod is None:
        index_tag = _choose_torch_index(cuda_version if prefer_gpu is not False else None)
        _install_torch(index_tag, log_fn=log_fn)
        torch_mod, import_err = _try_import_torch(force_reload=True)
        if torch_mod is None:
            raise RuntimeError(f"安装后仍无法导入 torch: {import_err}")

    if prefer_gpu is True:
        if cuda_version is None:
            raise RuntimeError("选择 GPU 推理，但未检测到 CUDA（nvidia-smi 不可用）。")
        has_cuda_build = bool(getattr(getattr(torch_mod, "version", None), "cuda", None))
        cuda_available = bool(torch_mod.cuda.is_available())
        if (not has_cuda_build) or (not cuda_available):
            index_tag = _choose_torch_index(cuda_version)
            _install_torch(index_tag, log_fn=log_fn)
            torch_mod, import_err = _try_import_torch(force_reload=True)
            if torch_mod is None:
                raise RuntimeError(f"重装 CUDA 版 torch 后导入失败: {import_err}")
            if not torch_mod.cuda.is_available():
                raise RuntimeError(
                    "已安装 CUDA 版 torch，但 torch.cuda 仍不可用。"
                    "请检查 NVIDIA 驱动/CUDA 运行环境。"
                )
    elif prefer_gpu is None and cuda_version is not None:
        has_cuda_build = bool(getattr(getattr(torch_mod, "version", None), "cuda", None))
        if not has_cuda_build:
            index_tag = _choose_torch_index(cuda_version)
            _install_torch(index_tag, log_fn=log_fn)
            torch_mod, import_err = _try_import_torch(force_reload=True)
            if torch_mod is None:
                raise RuntimeError(f"自动安装 CUDA 版 torch 后导入失败: {import_err}")

    return torch_mod


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_model_cache_env(env: dict[str, str]) -> None:
    cache_raw = env.get("FUSION_MODEL_CACHE_ROOT")
    if cache_raw:
        cache_root = Path(cache_raw).resolve()
    else:
        local_rr = DEFAULT_MODEL_CACHE_ROOT / "repos" / "RedRocket__JTP-3" / "model.py"
        legacy_rr = LEGACY_MODEL_CACHE_ROOT / "repos" / "RedRocket__JTP-3" / "model.py"
        if local_rr.exists():
            cache_root = DEFAULT_MODEL_CACHE_ROOT
        elif legacy_rr.exists():
            cache_root = LEGACY_MODEL_CACHE_ROOT
        else:
            cache_root = DEFAULT_MODEL_CACHE_ROOT
    hf_home = Path(env.get("HF_HOME") or (cache_root / "hf_home")).resolve()
    hf_hub_cache = Path(env.get("HF_HUB_CACHE") or (hf_home / "hub")).resolve()

    cache_root.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    hf_hub_cache.mkdir(parents=True, exist_ok=True)

    env.setdefault("FUSION_MODEL_CACHE_ROOT", str(cache_root))
    env.setdefault("HF_HOME", str(hf_home))
    env.setdefault("HF_HUB_CACHE", str(hf_hub_cache))
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    env.setdefault("FUSION_JTP3_MODEL_ID", "RedRocket/JTP-3")
    env.setdefault("FUSION_JTP3_FALLBACK_MODEL_ID", "none")


def _norm_opt_str(v: str | None) -> str | None:
    if v is None:
        return None
    x = str(v).strip()
    return x if x else None


def _safe_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def _safe_int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except Exception:
        return None


def _score_bucket(v: float | None) -> int | None:
    if v is None:
        return None
    return max(1, min(5, int(round(float(v)))))


def _apply_env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    env_ckpt = _norm_opt_str(os.getenv(ENV_CHECKPOINT_KEY))
    if env_ckpt:
        cfg.setdefault("inference", {})
        cfg["inference"]["checkpoint"] = env_ckpt
    return cfg


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _apply_env_overrides(
            {
            "server": {"host": "127.0.0.1", "port": 9400},
            "webui": {"default_config": "config.yaml"},
            "inference": {
                "checkpoint": "",
                "input_dir": "data/infer_images",
                "output_dir": "outputs/infer_run",
                "batch_size": 8,
                "device": None,
                "special_threshold": 0.5,
                "organize": {"enabled": True},
            },
        }
        )

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("server", {})
    cfg["server"].setdefault("host", "127.0.0.1")
    cfg["server"].setdefault("port", 9400)

    cfg.setdefault("webui", {})
    cfg["webui"].setdefault("default_config", "config.yaml")

    cfg.setdefault("inference", {})
    inf = cfg["inference"]
    inf.setdefault("checkpoint", "")
    inf.setdefault("input_dir", "data/infer_images")
    inf.setdefault("output_dir", "outputs/infer_run")
    inf.setdefault("batch_size", 8)
    inf.setdefault("device", None)
    inf.setdefault("special_threshold", 0.5)
    inf.setdefault("organize", {})
    inf["organize"].setdefault("enabled", True)
    return _apply_env_overrides(cfg)


def create_app(config_path: Path) -> FastAPI:
    _ensure_model_cache_env(os.environ)
    cfg = load_config(config_path)
    app = FastAPI(title="独立批量推理 UI", version="0.2.0")
    log = logging.getLogger("infer_ui")
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
    static_file = APP_DIR / "static" / "index.html"
    uploads_dir = ROOT / "outputs" / "_single_uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    lock = threading.Lock()
    proc: subprocess.Popen[str] | None = None
    current_task: dict[str, Any] | None = None
    logs: list[dict[str, Any]] = []
    next_log_id = 1
    max_logs = 8000
    runtime_cache: dict[str, Any] = {"key": None, "runtime": None}
    records_cache: dict[str, dict[str, Any]] = {}
    dialog_lock = threading.Lock()

    def append_log(line: str) -> None:
        nonlocal next_log_id
        line = line.rstrip("\r\n")
        if not line:
            return
        with lock:
            logs.append({"id": next_log_id, "ts": _now(), "line": line})
            next_log_id += 1
            if len(logs) > max_logs:
                logs[:] = logs[-max_logs:]

    def require_localhost(req: Request) -> None:
        host = (req.client.host if req.client else "").strip().lower()
        if host in {"127.0.0.1", "::1", "localhost"}:
            return
        raise HTTPException(
            status_code=403,
            detail="dialog endpoints are only available from localhost",
        )

    def _normalize_device_mode(raw: str | None) -> str | None:
        d = _norm_opt_str(raw)
        if d is None:
            return None
        dl = d.lower()
        if dl in {"", "auto"}:
            return None
        if dl in {"gpu", "cuda"}:
            return "gpu"
        if dl.startswith("cuda:"):
            return d
        if dl == "cpu":
            return "cpu"
        return d

    def _prepare_device(device_raw: str | None) -> str | None:
        mode = _normalize_device_mode(device_raw)
        mode_l = mode.lower() if isinstance(mode, str) else None
        if mode is None:
            _ensure_torch_runtime(prefer_gpu=None, log_fn=append_log)
            return None
        if mode_l == "cpu":
            _ensure_torch_runtime(prefer_gpu=False, log_fn=append_log)
            return "cpu"
        if mode_l == "gpu":
            _ensure_torch_runtime(prefer_gpu=True, log_fn=append_log)
            return "cuda"
        if mode_l and mode_l.startswith("cuda"):
            _ensure_torch_runtime(prefer_gpu=True, log_fn=append_log)
            return mode
        _ensure_torch_runtime(prefer_gpu=False, log_fn=append_log)
        return mode

    def resolve_cfg_path(raw: str | None) -> Path:
        selected = _norm_opt_str(raw) or str(cfg["webui"].get("default_config", "config.yaml"))
        p = Path(selected)
        if not p.is_absolute():
            p = ROOT / p
        return p.resolve()

    def resolve_path(path_like: str | None, base_dir: Path | None = None) -> Path | None:
        x = _norm_opt_str(path_like)
        if not x:
            return None
        p = Path(x)
        if not p.is_absolute():
            p = (base_dir or ROOT) / p
        return p.resolve()

    def _guess_input_dir(preferred_raw: str | None, checkpoint_raw: str | None) -> str | None:
        preferred = resolve_path(preferred_raw)
        if preferred and preferred.exists() and preferred.is_dir():
            return str(preferred)
        ckpt = resolve_path(checkpoint_raw)
        candidates = [
            resolve_path("data/infer_images"),
            resolve_path("images"),
            (ckpt.parent / "images").resolve() if ckpt else None,
        ]
        for p in candidates:
            if p and p.exists() and p.is_dir():
                return str(p)
        return str(preferred) if preferred else None

    def _guess_output_dir(preferred_raw: str | None, checkpoint_raw: str | None) -> str:
        preferred = resolve_path(preferred_raw)
        if preferred:
            return str(preferred)
        ckpt = resolve_path(checkpoint_raw)
        if ckpt:
            return str((ckpt.parent / "infer_run").resolve())
        return str((ROOT / "outputs" / "infer_run").resolve())

    def build_params(req: InferStartRequest, runtime_cfg: dict[str, Any]) -> dict[str, Any]:
        inf = runtime_cfg.get("inference", {})
        org = inf.get("organize", {}) if isinstance(inf.get("organize"), dict) else {}
        default_org = "on" if bool(org.get("enabled", True)) else "off"

        organize = _norm_opt_str(req.organize) or "auto"
        organize = organize.lower()
        if organize not in {"auto", "on", "off"}:
            raise HTTPException(status_code=400, detail="organize must be auto/on/off")

        checkpoint = _norm_opt_str(req.checkpoint) or str(inf.get("checkpoint", ""))
        input_dir = _guess_input_dir(
            _norm_opt_str(req.input_dir) or str(inf.get("input_dir", "")),
            checkpoint,
        )
        output_dir = _guess_output_dir(
            _norm_opt_str(req.output_dir) or str(inf.get("output_dir", "")),
            checkpoint,
        )

        return {
            "checkpoint": checkpoint,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "device": _norm_opt_str(req.device) or inf.get("device"),
            "batch_size": int(req.batch_size if req.batch_size is not None else inf.get("batch_size", 8)),
            "special_threshold": float(
                req.special_threshold
                if req.special_threshold is not None
                else inf.get("special_threshold", 0.5)
            ),
            "organize": organize if organize != "auto" else default_org,
        }

    def _load_infer_cfg(config_file: Path, overrides: dict[str, object] | None = None) -> dict[str, Any]:
        from scripts.batch_infer import load_config as load_infer_config

        infer_cfg, _ = load_infer_config(config_file, overrides=overrides)
        return infer_cfg

    def _resolve_task_runtime(
        *,
        config_raw: str | None,
        checkpoint_raw: str | None,
        device_raw: str | None,
        threshold_raw: float | None,
    ) -> dict[str, Any]:
        cfg_path = resolve_cfg_path(config_raw)
        if not cfg_path.exists():
            raise HTTPException(status_code=400, detail=f"config not found: {cfg_path}")
        # Prepare runtime device before importing batch_infer (which imports torch at module import time).
        base_cfg = load_config(cfg_path)
        base_inf = base_cfg.get("inference", {})
        requested_device = _norm_opt_str(device_raw) or _norm_opt_str(base_inf.get("device"))
        try:
            device = _prepare_device(requested_device)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"device prepare failed: {e}") from e
        overrides: dict[str, object] | None = None
        if _norm_opt_str(checkpoint_raw):
            overrides = {"checkpoint": str(checkpoint_raw)}
        infer_cfg = _load_infer_cfg(cfg_path, overrides=overrides)
        inf = infer_cfg.get("inference", {})
        checkpoint = resolve_path(str(inf.get("checkpoint", "")))
        if checkpoint is None or not checkpoint.exists():
            raise HTTPException(status_code=400, detail=f"checkpoint not found: {checkpoint}")
        special_threshold = float(
            threshold_raw if threshold_raw is not None else inf.get("special_threshold", 0.5)
        )
        return {
            "config_path": cfg_path,
            "checkpoint": checkpoint,
            "device": device,
            "special_threshold": special_threshold,
            "inference_cfg": infer_cfg,
        }

    def _build_runtime_cached(checkpoint: Path, device: str | None) -> dict[str, Any]:
        from scripts.batch_infer import _load_runtime

        key = f"{checkpoint.resolve()}::{_norm_opt_str(device) or 'auto'}"
        with lock:
            if runtime_cache.get("key") == key and runtime_cache.get("runtime") is not None:
                return runtime_cache["runtime"]
        runtime = _load_runtime(checkpoint, device)
        with lock:
            runtime_cache["key"] = key
            runtime_cache["runtime"] = runtime
        append_log(f"[infer_ui] 模型缓存更新: {key}")
        return runtime

    def _resolve_runtime_or_http(
        *,
        config_raw: str | None,
        checkpoint_raw: str | None,
        device_raw: str | None,
        threshold_raw: float | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            runtime_params = _resolve_task_runtime(
                config_raw=config_raw,
                checkpoint_raw=checkpoint_raw,
                device_raw=device_raw,
                threshold_raw=threshold_raw,
            )
            runtime = _build_runtime_cached(runtime_params["checkpoint"], runtime_params["device"])
            return runtime_params, runtime
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"runtime init failed: {e}") from e

    def _format_record(raw: dict[str, Any]) -> dict[str, Any]:
        row = dict(raw)
        for k in TARGETS:
            row[k] = _safe_float(row.get(k))
        row["in_domain_prob"] = _safe_float(row.get("in_domain_prob"))
        row["in_domain_pred"] = _safe_int(row.get("in_domain_pred"))
        row["special_tag"] = _safe_int(row.get("special_tag"))
        row["special_reason"] = str(row.get("special_reason") or "")
        row["error"] = str(row.get("error") or "")
        return row

    def _decompose_record(row: dict[str, Any]) -> dict[str, Any]:
        out = _format_record(row)
        score_heads = []
        for k, name in [
            ("aesthetic", "美学"),
            ("composition", "构图"),
            ("color", "色彩"),
            ("sexual", "色情"),
        ]:
            score = out.get(k)
            score_heads.append(
                {
                    "key": k,
                    "name": name,
                    "score": score,
                    "bucket": _score_bucket(score),
                }
            )
        cls_head = {
            "in_domain_prob": out.get("in_domain_prob"),
            "in_domain_pred": out.get("in_domain_pred"),
            "special_tag": out.get("special_tag"),
            "special_reason": out.get("special_reason") or "",
        }
        out["score_heads"] = score_heads
        out["cls_head"] = cls_head
        return out

    def _infer_images(
        images: list[Image.Image],
        runtime: dict[str, Any],
        special_threshold: float,
    ) -> list[dict[str, Any]]:
        if not images:
            return []
        torch_mod, import_err = _try_import_torch(force_reload=False)
        if torch_mod is None:
            raise RuntimeError(f"torch unavailable: {import_err}")
        jtp = runtime["jtp"]
        waifu = runtime["waifu"]
        head = runtime["head"]
        has_cls_head = bool(runtime.get("has_cls_head"))
        with torch_mod.no_grad():
            f1 = jtp(images)
            f2 = waifu(images)
            reg_pred, cls_logit = head(torch_mod.cat([f1, f2], dim=-1))
            reg_list = reg_pred.cpu().tolist()
            cls_probs = (
                torch_mod.sigmoid(cls_logit).cpu().tolist() if has_cls_head else [None] * len(images)
            )

        out: list[dict[str, Any]] = []
        for reg_row, cls_prob in zip(reg_list, cls_probs):
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
            out.append(
                {
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
        return out

    def _read_summary(output_dir: Path) -> dict[str, Any] | None:
        p = output_dir / "summary.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _pick_predictions_file(output_dir: Path, summary: dict[str, Any] | None) -> Path | None:
        if summary and isinstance(summary.get("output_files"), dict):
            out_files = summary["output_files"]
            jsonl_file = resolve_path(out_files.get("jsonl"))
            csv_file = resolve_path(out_files.get("csv"))
            if jsonl_file and jsonl_file.exists():
                return jsonl_file
            if csv_file and csv_file.exists():
                return csv_file
        for name in ("predictions.jsonl", "predictions.csv"):
            p = output_dir / name
            if p.exists():
                return p
        return None

    def _load_records_cached(pred_file: Path) -> list[dict[str, Any]]:
        key = str(pred_file.resolve())
        st = pred_file.stat()
        with lock:
            cached = records_cache.get(key)
            if cached and cached.get("mtime_ns") == st.st_mtime_ns and cached.get("size") == st.st_size:
                return list(cached["records"])

        rows: list[dict[str, Any]] = []
        if pred_file.suffix.lower() == ".jsonl":
            with pred_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(_format_record(json.loads(line)))
                    except Exception:
                        continue
        else:
            with pred_file.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(_format_record(row))

        with lock:
            records_cache[key] = {
                "mtime_ns": st.st_mtime_ns,
                "size": st.st_size,
                "records": list(rows),
            }
        return rows

    def _resolve_output_dir(raw: str | None) -> Path:
        selected = _norm_opt_str(raw)
        if not selected:
            with lock:
                if current_task and current_task.get("params"):
                    selected = _norm_opt_str(current_task["params"].get("output_dir"))
        if not selected:
            selected = str(cfg.get("inference", {}).get("output_dir", "outputs/infer_run"))
        p = Path(selected)
        if not p.is_absolute():
            p = ROOT / p
        return p.resolve()

    def stream_output(p: subprocess.Popen[str]) -> None:
        if p.stdout is None:
            return
        for line in p.stdout:
            append_log(line)

    def wait_proc(p: subprocess.Popen[str], task_id: str) -> None:
        nonlocal proc
        rc = p.wait()
        with lock:
            task = current_task if current_task and current_task.get("task_id") == task_id else None
            if task is not None:
                if task.get("status") == "stopping":
                    task["status"] = "stopped"
                else:
                    task["status"] = "done" if rc == 0 else "failed"
                task["return_code"] = int(rc)
                task["finished_at"] = _now()
                out_dir = resolve_path(_norm_opt_str(task.get("resolved_output_dir")))
                if rc == 0 and out_dir:
                    summary_path = out_dir / "summary.json"
                    if summary_path.exists():
                        task["summary_path"] = str(summary_path.resolve())
                        try:
                            task["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
                        except Exception:
                            task["summary"] = None
            proc = None

    @app.get("/api/health")
    def health():
        return {"ok": True, "mode": "infer"}

    @app.get("/api/config")
    def api_config():
        return cfg

    @app.get("/api/image")
    def api_image(path: str = Query(..., min_length=1)):
        p = resolve_path(path)
        if p is None or (not p.exists()) or (not p.is_file()):
            raise HTTPException(status_code=404, detail=f"image not found: {path}")
        if p.suffix.lower() not in IMAGE_EXTS:
            raise HTTPException(status_code=400, detail="unsupported image extension")
        return FileResponse(str(p))

    @app.get("/api/infer/status")
    def infer_status():
        with lock:
            task = dict(current_task) if current_task else None
            return {
                "task": task,
                "running": proc is not None and proc.poll() is None,
                "log_count": len(logs),
                "last_log_id": logs[-1]["id"] if logs else 0,
            }

    @app.get("/api/infer/logs")
    def infer_logs(
        since_id: int = Query(default=0, ge=0),
        limit: int = Query(default=300, ge=1, le=2000),
    ):
        with lock:
            items = [x for x in logs if int(x["id"]) > since_id][:limit]
            last_id = logs[-1]["id"] if logs else 0
        return {"items": items, "last_id": last_id}

    @app.get("/api/dialog/pick")
    def dialog_pick(
        request: Request,
        kind: str = Query(..., description="checkpoint|image|input_dir|output_dir"),
        current: str | None = Query(default=None),
    ):
        require_localhost(request)
        k = str(kind).strip().lower()
        if k not in {"checkpoint", "image", "input_dir", "output_dir"}:
            raise HTTPException(status_code=400, detail="invalid kind")

        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"tkinter unavailable: {e}") from e

        initial_dir = str(ROOT)
        cur = _norm_opt_str(current)
        if cur:
            try:
                cp = Path(cur).expanduser()
                if cp.exists():
                    initial_dir = str((cp.parent if cp.is_file() else cp).resolve())
            except Exception:
                pass

        with dialog_lock:
            root = tk.Tk()
            root.withdraw()
            try:
                root.attributes("-topmost", True)
            except Exception:
                pass
            root.update()
            path = ""
            try:
                if k in {"input_dir", "output_dir"}:
                    path = filedialog.askdirectory(
                        title="Select Folder",
                        initialdir=initial_dir,
                    ) or ""
                elif k == "checkpoint":
                    path = filedialog.askopenfilename(
                        title="Select Model Checkpoint",
                        initialdir=initial_dir,
                        filetypes=[
                            ("SafeTensors", "*.safetensors"),
                            ("PyTorch", "*.pt *.pth *.bin"),
                            ("All Files", "*.*"),
                        ],
                    ) or ""
                else:
                    path = filedialog.askopenfilename(
                        title="Select Image",
                        initialdir=initial_dir,
                        filetypes=[
                            ("Images", "*.jpg *.jpeg *.png *.webp *.bmp *.gif"),
                            ("All Files", "*.*"),
                        ],
                    ) or ""
            finally:
                root.destroy()

        return {"ok": True, "kind": k, "path": path or None}

    def _query_result_records(
        *,
        output_dir: str | None,
        q: str,
        special_filter: str,
        sort_by: str,
        sort_order: str,
    ) -> tuple[Path, dict[str, Any] | None, Path | None, list[dict[str, Any]], list[dict[str, Any]]]:
        out_dir = _resolve_output_dir(output_dir)
        summary = _read_summary(out_dir)
        pred_file = _pick_predictions_file(out_dir, summary)
        records: list[dict[str, Any]] = []
        if pred_file and pred_file.exists():
            records = _load_records_cached(pred_file)

        keyword = q.strip().lower()
        items = records
        if keyword:
            items = [
                x
                for x in items
                if keyword in str(x.get("relative_path") or "").lower()
                or keyword in str(x.get("image_path") or "").lower()
            ]

        sf = special_filter.strip().lower()
        if sf == "special":
            items = [x for x in items if int(x.get("special_tag") or 0) == 1]
        elif sf == "in_domain":
            items = [x for x in items if int(x.get("special_tag") or 0) == 0]

        allowed_sort = {
            "aesthetic",
            "composition",
            "color",
            "sexual",
            "in_domain_prob",
            "special_tag",
        }
        if sort_by in allowed_sort:
            reverse = str(sort_order).lower() != "asc"
            items = sorted(
                items,
                key=lambda r: (r.get(sort_by) is None, r.get(sort_by)),
                reverse=reverse,
            )

        return out_dir, summary, pred_file, records, items

    @app.get("/api/infer/results")
    def infer_results(
        output_dir: str | None = Query(default=None),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=24, ge=1, le=200),
        q: str = Query(default=""),
        special_filter: str = Query(default="all"),  # all/special/in_domain
        sort_by: str = Query(default=""),
        sort_order: str = Query(default="desc"),  # asc/desc
    ):
        out_dir, summary, pred_file, records, items = _query_result_records(
            output_dir=output_dir,
            q=q,
            special_filter=special_filter,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        threshold = _safe_float((summary or {}).get("special_threshold"))

        total = len(items)
        pages = (total + page_size - 1) // page_size if total else 0
        if pages > 0 and page > pages:
            page = pages
        start = (page - 1) * page_size
        end = start + page_size
        paged = items[start:end]

        out_items = []
        for row in paged:
            one = _decompose_record(row)
            p = resolve_path(one.get("image_path"))
            one["image_url"] = f"/api/image?path={quote(str(p))}" if p else ""
            one["special_threshold"] = threshold
            out_items.append(one)

        special_count = sum(1 for r in records if int(r.get("special_tag") or 0) == 1)
        return {
            "output_dir": str(out_dir),
            "summary": summary,
            "special_threshold": threshold,
            "prediction_file": str(pred_file) if pred_file else None,
            "total": total,
            "page": page,
            "pages": pages,
            "page_size": page_size,
            "special_count": special_count,
            "records_count": len(records),
            "items": out_items,
        }

    @app.get("/api/infer/results/export")
    def infer_results_export(
        output_dir: str | None = Query(default=None),
        q: str = Query(default=""),
        special_filter: str = Query(default="all"),  # all/special/in_domain
        sort_by: str = Query(default=""),
        sort_order: str = Query(default="desc"),  # asc/desc
    ):
        out_dir, summary, pred_file, records, items = _query_result_records(
            output_dir=output_dir,
            q=q,
            special_filter=special_filter,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        threshold = _safe_float((summary or {}).get("special_threshold"))

        csv_buf = io.StringIO()
        fieldnames = [
            "relative_path",
            "image_path",
            "aesthetic",
            "composition",
            "color",
            "sexual",
            "aesthetic_bucket",
            "composition_bucket",
            "color_bucket",
            "sexual_bucket",
            "in_domain_prob",
            "in_domain_pred",
            "special_tag",
            "special_reason",
            "special_threshold",
            "error",
        ]
        writer = csv.DictWriter(csv_buf, fieldnames=fieldnames)
        writer.writeheader()
        for row in items:
            writer.writerow(
                {
                    "relative_path": row.get("relative_path") or "",
                    "image_path": row.get("image_path") or "",
                    "aesthetic": row.get("aesthetic"),
                    "composition": row.get("composition"),
                    "color": row.get("color"),
                    "sexual": row.get("sexual"),
                    "aesthetic_bucket": _score_bucket(_safe_float(row.get("aesthetic"))),
                    "composition_bucket": _score_bucket(_safe_float(row.get("composition"))),
                    "color_bucket": _score_bucket(_safe_float(row.get("color"))),
                    "sexual_bucket": _score_bucket(_safe_float(row.get("sexual"))),
                    "in_domain_prob": row.get("in_domain_prob"),
                    "in_domain_pred": row.get("in_domain_pred"),
                    "special_tag": row.get("special_tag"),
                    "special_reason": row.get("special_reason") or "",
                    "special_threshold": threshold,
                    "error": row.get("error") or "",
                }
            )

        payload = ("\ufeff" + csv_buf.getvalue()).encode("utf-8")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"infer_results_filtered_{stamp}.csv"
        headers = {
            "Content-Disposition": f'attachment; filename="{fname}"',
            "X-Output-Dir": str(out_dir),
            "X-Prediction-File": str(pred_file) if pred_file else "",
            "X-Records-Count": str(len(records)),
        }
        return StreamingResponse(
            iter([payload]),
            media_type="text/csv; charset=utf-8",
            headers=headers,
        )

    @app.post("/api/single/infer/path")
    def single_infer_path(req: SinglePathInferRequest):
        img_path = resolve_path(req.image_path)
        if img_path is None or (not img_path.exists()) or (not img_path.is_file()):
            raise HTTPException(status_code=400, detail=f"image not found: {req.image_path}")
        if img_path.suffix.lower() not in IMAGE_EXTS:
            raise HTTPException(status_code=400, detail="unsupported image extension")

        runtime_params, runtime = _resolve_runtime_or_http(
            config_raw=req.config_path,
            checkpoint_raw=req.checkpoint,
            device_raw=req.device,
            threshold_raw=req.special_threshold,
        )
        try:
            with Image.open(img_path) as img:
                rgb = img.convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"image load failed: {e}") from e

        try:
            rec = _infer_images([rgb], runtime, runtime_params["special_threshold"])[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"inference failed: {e}") from e
        rec["image_path"] = str(img_path)
        rec["relative_path"] = img_path.name
        out = _decompose_record(rec)
        out["image_url"] = f"/api/image?path={quote(str(img_path))}"
        out["meta"] = {
            "config_path": str(runtime_params["config_path"]),
            "checkpoint": str(runtime_params["checkpoint"]),
            "device": runtime.get("device"),
            "special_threshold": runtime_params["special_threshold"],
            "has_cls_head": bool(runtime.get("has_cls_head")),
        }
        return out

    @app.post("/api/single/infer/upload")
    async def single_infer_upload(
        file: UploadFile = File(...),
        config_path: str | None = Form(default=None),
        checkpoint: str | None = Form(default=None),
        device: str | None = Form(default=None),
        special_threshold: float | None = Form(default=None),
    ):
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="empty upload file")
        suffix = Path(file.filename or "upload.png").suffix.lower()
        if suffix not in IMAGE_EXTS:
            suffix = ".png"
        try:
            rgb = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"image decode failed: {e}") from e

        runtime_params, runtime = _resolve_runtime_or_http(
            config_raw=config_path,
            checkpoint_raw=checkpoint,
            device_raw=device,
            threshold_raw=special_threshold,
        )

        try:
            rec = _infer_images([rgb], runtime, runtime_params["special_threshold"])[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"inference failed: {e}") from e
        save_path = uploads_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{suffix}"
        try:
            save_path.write_bytes(raw)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"save upload failed: {e}") from e
        rec["image_path"] = str(save_path)
        rec["relative_path"] = file.filename or save_path.name
        out = _decompose_record(rec)
        out["image_url"] = f"/api/image?path={quote(str(save_path))}"
        out["meta"] = {
            "config_path": str(runtime_params["config_path"]),
            "checkpoint": str(runtime_params["checkpoint"]),
            "device": runtime.get("device"),
            "special_threshold": runtime_params["special_threshold"],
            "has_cls_head": bool(runtime.get("has_cls_head")),
        }
        return out

    @app.post("/api/infer/start")
    def infer_start(req: InferStartRequest):
        nonlocal proc, current_task, logs, next_log_id
        cfg_path = resolve_cfg_path(req.config_path)
        if not cfg_path.exists():
            raise HTTPException(status_code=400, detail=f"config not found: {cfg_path}")

        runtime_cfg = load_config(cfg_path)
        params = build_params(req, runtime_cfg)
        try:
            params["device"] = _prepare_device(params.get("device"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"device prepare failed: {e}") from e
        if not params["checkpoint"]:
            raise HTTPException(status_code=400, detail="checkpoint is empty")
        if not params["input_dir"]:
            raise HTTPException(status_code=400, detail="input_dir is empty")
        if not params["output_dir"]:
            raise HTTPException(status_code=400, detail="output_dir is empty")

        overrides: dict[str, object] = {
            "checkpoint": params["checkpoint"],
            "input_dir": params["input_dir"],
            "output_dir": params["output_dir"],
            "batch_size": params["batch_size"],
            "special_threshold": params["special_threshold"],
        }
        if _norm_opt_str(params.get("device")):
            overrides["device"] = str(params["device"])
        if str(params.get("organize")) == "on":
            overrides["organize.enabled"] = True
        elif str(params.get("organize")) == "off":
            overrides["organize.enabled"] = False

        infer_cfg = _load_infer_cfg(cfg_path, overrides=overrides)
        resolved_out_dir = resolve_path(str(infer_cfg.get("inference", {}).get("output_dir", "")))

        with lock:
            if proc is not None and proc.poll() is None:
                raise HTTPException(status_code=409, detail="推理任务正在运行中。")
            logs = []
            next_log_id = 1
            task_id = uuid.uuid4().hex[:12]
            current_task = {
                "task_id": task_id,
                "status": "running",
                "created_at": _now(),
                "started_at": _now(),
                "finished_at": None,
                "config_path": str(cfg_path),
                "params": params,
                "resolved_output_dir": str(resolved_out_dir) if resolved_out_dir else None,
                "pid": None,
                "return_code": None,
                "summary_path": None,
                "summary": None,
            }

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "batch_infer.py"),
            "--config",
            str(cfg_path),
            "--checkpoint",
            str(params["checkpoint"]),
            "--input-dir",
            str(params["input_dir"]),
            "--output-dir",
            str(params["output_dir"]),
            "--batch-size",
            str(params["batch_size"]),
            "--special-threshold",
            str(params["special_threshold"]),
            "--organize",
            str(params["organize"]),
        ]
        if params.get("device"):
            cmd.extend(["--device", str(params["device"])])

        try:
            env = dict(os.environ)
            p = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except Exception as e:
            with lock:
                if current_task:
                    current_task["status"] = "failed"
                    current_task["finished_at"] = _now()
                    current_task["return_code"] = -1
            raise HTTPException(status_code=500, detail=f"启动推理失败: {e}") from e

        with lock:
            proc = p
            if current_task:
                current_task["pid"] = p.pid
        append_log(f"[infer_ui] 启动推理: config={cfg_path}")
        append_log(f"[infer_ui] 命令: {' '.join(cmd)}")
        threading.Thread(target=stream_output, args=(p,), daemon=True).start()
        threading.Thread(target=wait_proc, args=(p, task_id), daemon=True).start()
        return {"ok": True, "task_id": task_id}

    @app.post("/api/infer/stop")
    def infer_stop():
        nonlocal proc
        with lock:
            if proc is None or proc.poll() is not None:
                raise HTTPException(status_code=400, detail="当前没有运行中的推理任务。")
            if current_task:
                current_task["status"] = "stopping"
            p = proc
        p.terminate()
        append_log("[infer_ui] 收到停止请求，正在终止推理进程。")
        return {"ok": True}

    @app.get("/")
    def root():
        if not static_file.exists():
            raise HTTPException(status_code=500, detail=f"Missing UI file: {static_file}")
        return FileResponse(str(static_file))

    return app
