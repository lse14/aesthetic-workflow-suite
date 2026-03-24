import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel


APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR
DEFAULT_MODEL_CACHE_ROOT = (ROOT.parent / "model" / "_models").resolve()


class TrainStartRequest(BaseModel):
    mode: str = "config"
    config_path: str | None = None
    annotations: str | None = None
    image_root: str | None = None
    train_split: str | None = None
    val_split: str | None = None
    val_ratio: float | None = None
    epochs: int | None = None
    batch_size: int | None = None
    num_workers: int | None = None
    lr: float | None = None
    weight_decay: float | None = None
    seed: int | None = None
    device: str | None = None
    output_dir: str | None = None
    model_name: str | None = None
    model_format: str | None = None
    loss: str | None = None
    cls_loss_weight: float | None = None
    cls_pos_weight: float | None = None
    eval_split: str | None = None
    eval_batch_size: int | None = None
    target_dims: list[str] | None = None
    skip_eval: bool = False


class TrainConfigSaveRequest(BaseModel):
    config_path: str
    yaml_text: str


class QuickConfigSaveRequest(BaseModel):
    quick_config: dict[str, Any]


class PathPickRequest(BaseModel):
    initial_path: str | None = None
    kind: str | None = None


DEFAULT_TARGET_DIMS = ["aesthetic", "composition", "color", "sexual"]
ALLOWED_MODEL_FORMATS = {"safetensors", "pt", "pth", "ckpt"}
DEFAULT_MODEL_FORMAT = "safetensors"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_model_cache_env(env: dict[str, str]) -> None:
    cache_root = Path(env.get("FUSION_MODEL_CACHE_ROOT") or DEFAULT_MODEL_CACHE_ROOT).resolve()
    hf_home = Path(env.get("HF_HOME") or (cache_root / "hf_home")).resolve()
    hf_hub_cache = Path(env.get("HF_HUB_CACHE") or (hf_home / "hub")).resolve()

    cache_root.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    hf_hub_cache.mkdir(parents=True, exist_ok=True)

    env.setdefault("FUSION_MODEL_CACHE_ROOT", str(cache_root))
    env.setdefault("HF_HOME", str(hf_home))
    env.setdefault("HF_HUB_CACHE", str(hf_hub_cache))
    env.setdefault("FUSION_JTP3_MODEL_ID", "RedRocket/JTP-3")
    env.setdefault("FUSION_JTP3_FALLBACK_MODEL_ID", "none")


def default_quick_config() -> dict[str, Any]:
    return {
        "annotations": "",
        "image_root": "",
        "train_split": "",
        "val_split": "",
        "val_ratio": None,
        "epochs": 16,
        "batch_size": 8,
        "num_workers": 4,
        "lr": 0.0002,
        "device": "",
        "output_dir": "",
        "model_name": "best",
        "model_format": DEFAULT_MODEL_FORMAT,
        "eval_split": "",
        "eval_batch_size": None,
        "target_dims": list(DEFAULT_TARGET_DIMS),
        "skip_eval": False,
    }


def normalize_quick_config(raw: Any) -> dict[str, Any]:
    out = default_quick_config()
    if isinstance(raw, dict):
        for k in out:
            if k in raw:
                out[k] = raw[k]

    for k in (
        "annotations",
        "image_root",
        "train_split",
        "val_split",
        "device",
        "output_dir",
        "model_name",
        "eval_split",
    ):
        v = out.get(k)
        out[k] = "" if v is None else str(v).strip()

    out["model_name"] = out["model_name"] or "best"
    fmt = str(out.get("model_format") or "").strip().lower().lstrip(".")
    if fmt not in ALLOWED_MODEL_FORMATS:
        fmt = DEFAULT_MODEL_FORMAT
    out["model_format"] = fmt

    for k in ("val_ratio", "lr"):
        v = out.get(k)
        if v in (None, ""):
            out[k] = None
            continue
        try:
            out[k] = float(v)
        except Exception:
            out[k] = default_quick_config()[k]

    for k in ("epochs", "batch_size", "num_workers", "eval_batch_size"):
        v = out.get(k)
        if v in (None, ""):
            out[k] = None
            continue
        try:
            out[k] = int(v)
        except Exception:
            out[k] = default_quick_config()[k]

    dims_raw = out.get("target_dims")
    if not isinstance(dims_raw, list):
        out["target_dims"] = list(DEFAULT_TARGET_DIMS)
    else:
        allowed = set(DEFAULT_TARGET_DIMS)
        dims = [str(x).strip().lower() for x in dims_raw if str(x).strip()]
        dims = [x for x in dims if x in allowed]
        out["target_dims"] = dims if dims else list(DEFAULT_TARGET_DIMS)

    out["skip_eval"] = bool(out.get("skip_eval"))
    return out


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "server": {"host": "127.0.0.1", "port": 9300},
            "training": {
                "default_config": "configs/fusion_1k_baseline.yaml",
                "quick_defaults": default_quick_config(),
            },
        }
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("server", {})
    cfg["server"].setdefault("host", "127.0.0.1")
    cfg["server"].setdefault("port", 9300)
    cfg.setdefault("training", {})
    cfg["training"].setdefault("default_config", "configs/fusion_1k_baseline.yaml")
    cfg["training"]["quick_defaults"] = normalize_quick_config(cfg["training"].get("quick_defaults"))
    return cfg


def create_app(config_path: Path) -> FastAPI:
    cfg = load_config(config_path)
    app = FastAPI(title="独立训练 UI", version="0.1.0")
    static_file = APP_DIR / "static" / "index.html"

    lock = threading.Lock()
    proc: subprocess.Popen[str] | None = None
    current_task: dict[str, Any] | None = None
    logs: list[dict[str, Any]] = []
    next_log_id = 1
    max_logs = 5000

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

    def resolve_cfg_path(raw: str | None) -> Path:
        path_str = (raw or "").strip() or str(cfg["training"]["default_config"])
        p = Path(path_str)
        if not p.is_absolute():
            p = ROOT / p
        return p

    def save_ui_config() -> None:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def require_localhost(req: Request) -> None:
        host = (req.client.host if req.client else "").strip().lower()
        if host in {"127.0.0.1", "::1", "localhost"}:
            return
        raise HTTPException(
            status_code=403,
            detail="path picker is only available from localhost",
        )

    def _split_initial_path(path_raw: str | None) -> tuple[Path, str]:
        if not path_raw:
            return ROOT, ""
        p = Path(path_raw).expanduser()
        if p.exists():
            if p.is_dir():
                return p, ""
            return p.parent, p.name
        parent = p.parent if str(p.parent) not in {"", "."} else ROOT
        if not parent.exists():
            parent = ROOT
        return parent, p.name if p.suffix else ""

    def _open_file_dialog(
        *,
        title: str,
        filetypes: list[tuple[str, str]],
        initial_path: str | None = None,
    ) -> str | None:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"无法调用系统文件选择器: {e}") from e

        initial_dir, initial_file = _split_initial_path(initial_path)
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        try:
            picked = filedialog.askopenfilename(
                title=title,
                initialdir=str(initial_dir),
                initialfile=initial_file,
                filetypes=filetypes,
            )
            return str(Path(picked).resolve()) if picked else None
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"打开文件选择器失败: {e}") from e
        finally:
            root.destroy()

    def _open_dir_dialog(*, title: str, initial_path: str | None = None) -> str | None:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"无法调用系统目录选择器: {e}") from e

        initial_dir, _ = _split_initial_path(initial_path)
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        try:
            picked = filedialog.askdirectory(
                title=title,
                initialdir=str(initial_dir),
                mustexist=False,
            )
            return str(Path(picked).resolve()) if picked else None
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"打开目录选择器失败: {e}") from e
        finally:
            root.destroy()

    def _clean(v: str | None) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    def _append_opt(cmd: list[str], flag: str, value: str | int | float | None) -> None:
        if value is None:
            return
        cmd.extend([flag, str(value)])

    def stream_output(p: subprocess.Popen[str]) -> None:
        if p.stdout is None:
            return
        for line in p.stdout:
            append_log(line)

    def wait_proc(p: subprocess.Popen[str], task_id: str) -> None:
        nonlocal proc
        rc = p.wait()
        with lock:
            if current_task and current_task.get("task_id") == task_id:
                if current_task.get("status") == "stopping":
                    current_task["status"] = "stopped"
                else:
                    current_task["status"] = "done" if rc == 0 else "failed"
                current_task["return_code"] = int(rc)
                current_task["finished_at"] = _now()
            proc = None

    @app.get("/api/health")
    def health():
        return {"ok": True, "mode": "training"}

    @app.get("/api/config")
    def api_config():
        return cfg

    @app.get("/api/train/config/load")
    def train_config_load(path: str | None = Query(default=None)):
        cfg_path = resolve_cfg_path(path)
        if not cfg_path.exists():
            raise HTTPException(status_code=404, detail=f"config not found: {cfg_path}")
        try:
            text = cfg_path.read_text(encoding="utf-8")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"读取配置失败: {e}") from e
        return {
            "config_path": str(cfg_path.relative_to(ROOT)) if cfg_path.is_relative_to(ROOT) else str(cfg_path),
            "resolved_path": str(cfg_path),
            "yaml_text": text,
        }

    @app.post("/api/train/config/save")
    def train_config_save(req: TrainConfigSaveRequest):
        cfg_path = resolve_cfg_path(req.config_path)
        yaml_text = str(req.yaml_text or "")
        if not yaml_text.strip():
            raise HTTPException(status_code=400, detail="YAML 内容为空。")
        try:
            parsed = yaml.safe_load(yaml_text)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"YAML 解析失败: {e}") from e
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=400, detail="YAML 顶层必须是对象（mapping）。")

        required_roots = {"data", "models", "model_head", "training"}
        missing = [k for k in sorted(required_roots) if k not in parsed]
        if missing:
            raise HTTPException(status_code=400, detail=f"YAML 缺少根字段: {', '.join(missing)}")

        try:
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(yaml_text, encoding="utf-8")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"保存配置失败: {e}") from e
        return {
            "ok": True,
            "config_path": str(cfg_path.relative_to(ROOT)) if cfg_path.is_relative_to(ROOT) else str(cfg_path),
            "resolved_path": str(cfg_path),
        }

    @app.get("/api/train/quick-config/load")
    def train_quick_config_load():
        training_cfg = cfg.setdefault("training", {})
        quick_cfg = normalize_quick_config(training_cfg.get("quick_defaults"))
        training_cfg["quick_defaults"] = quick_cfg
        return {"quick_config": quick_cfg}

    @app.post("/api/train/quick-config/save")
    def train_quick_config_save(req: QuickConfigSaveRequest):
        quick_cfg = normalize_quick_config(req.quick_config)
        cfg.setdefault("training", {})
        cfg["training"]["quick_defaults"] = quick_cfg
        try:
            save_ui_config()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"保存快速参数失败: {e}") from e
        return {"ok": True, "quick_config": quick_cfg}

    @app.post("/api/path/pick-file")
    def pick_file(req: PathPickRequest, request: Request):
        require_localhost(request)
        kind = (req.kind or "").strip().lower()
        if kind == "config":
            title = "选择训练配置文件"
            filetypes = [
                ("YAML Files", "*.yaml *.yml"),
                ("All Files", "*.*"),
            ]
        elif kind == "annotations":
            title = "选择标注文件"
            filetypes = [
                ("Annotation Files", "*.jsonl *.csv *.db"),
                ("All Files", "*.*"),
            ]
        else:
            title = "选择文件"
            filetypes = [("All Files", "*.*")]

        picked = _open_file_dialog(
            title=title,
            filetypes=filetypes,
            initial_path=req.initial_path,
        )
        return {"path": picked}

    @app.post("/api/path/pick-dir")
    def pick_dir(req: PathPickRequest, request: Request):
        require_localhost(request)
        kind = (req.kind or "").strip().lower()
        if kind == "image_root":
            title = "选择图片根目录"
        elif kind == "output_dir":
            title = "选择输出目录"
        else:
            title = "选择目录"
        picked = _open_dir_dialog(title=title, initial_path=req.initial_path)
        return {"path": picked}

    @app.get("/api/train/status")
    def train_status():
        with lock:
            return {
                "task": dict(current_task) if current_task else None,
                "running": proc is not None and proc.poll() is None,
                "log_count": len(logs),
                "last_log_id": logs[-1]["id"] if logs else 0,
            }

    @app.get("/api/train/logs")
    def train_logs(
        since_id: int = Query(default=0, ge=0),
        before_id: int | None = Query(default=None, ge=1),
        limit: int = Query(default=200, ge=1, le=1000),
    ):
        with lock:
            first_id = logs[0]["id"] if logs else 0
            last_id = logs[-1]["id"] if logs else 0
            if before_id is not None:
                older = [x for x in logs if int(x["id"]) < before_id]
                items = older[-limit:]
                has_more_before = len(older) > len(items)
            else:
                items = [x for x in logs if int(x["id"]) > since_id][:limit]
                has_more_before = bool(items) and int(items[0]["id"]) > int(first_id)
        return {
            "items": items,
            "first_id": first_id,
            "last_id": last_id,
            "has_more_before": has_more_before,
        }

    @app.post("/api/train/start")
    def train_start(req: TrainStartRequest):
        nonlocal proc, current_task, logs, next_log_id
        mode = (req.mode or "config").strip().lower()
        if mode not in {"config", "simple"}:
            raise HTTPException(status_code=400, detail="mode 必须是 config 或 simple。")

        cfg_path: Path | None = None
        cmd = [sys.executable, "-X", "utf8", str(ROOT / "scripts" / "train_fusion.py")]
        launch_desc = ""

        if mode == "config":
            cfg_path = resolve_cfg_path(req.config_path)
            if not cfg_path.exists():
                raise HTTPException(status_code=400, detail=f"config not found: {cfg_path}")
            cmd.extend(["--config", str(cfg_path)])
            launch_desc = f"config={cfg_path}"
        else:
            raw_cfg = _clean(req.config_path)
            if raw_cfg:
                cfg_path = resolve_cfg_path(raw_cfg)
                if not cfg_path.exists():
                    raise HTTPException(status_code=400, detail=f"config not found: {cfg_path}")
                cmd.extend(["--config", str(cfg_path)])

            annotations = _clean(req.annotations)
            if annotations is None and cfg_path is None:
                raise HTTPException(
                    status_code=400,
                    detail="快速模式至少要提供 annotations，或提供可用的 config。",
                )
            _append_opt(cmd, "--annotations", annotations)
            _append_opt(cmd, "--image-root", _clean(req.image_root))
            _append_opt(cmd, "--train-split", _clean(req.train_split))
            _append_opt(cmd, "--val-split", _clean(req.val_split))
            _append_opt(cmd, "--val-ratio", req.val_ratio)
            _append_opt(cmd, "--epochs", req.epochs)
            _append_opt(cmd, "--batch-size", req.batch_size)
            _append_opt(cmd, "--num-workers", req.num_workers)
            _append_opt(cmd, "--lr", req.lr)
            _append_opt(cmd, "--weight-decay", req.weight_decay)
            _append_opt(cmd, "--seed", req.seed)
            _append_opt(cmd, "--device", _clean(req.device))
            _append_opt(cmd, "--output-dir", _clean(req.output_dir))
            _append_opt(cmd, "--model-name", _clean(req.model_name))
            _append_opt(cmd, "--model-format", _clean(req.model_format))
            _append_opt(cmd, "--loss", _clean(req.loss))
            _append_opt(cmd, "--cls-loss-weight", req.cls_loss_weight)
            _append_opt(cmd, "--cls-pos-weight", req.cls_pos_weight)
            _append_opt(cmd, "--eval-split", _clean(req.eval_split))
            _append_opt(cmd, "--eval-batch-size", req.eval_batch_size)
            if req.target_dims:
                dims = [str(x).strip().lower() for x in req.target_dims if str(x).strip()]
                if dims:
                    _append_opt(cmd, "--target-dims", ",".join(dims))
            if bool(req.skip_eval):
                cmd.append("--skip-eval")
            launch_desc = "simple-mode"

        with lock:
            if proc is not None and proc.poll() is None:
                raise HTTPException(status_code=409, detail="训练任务正在运行中。")
            logs = []
            next_log_id = 1
            task_id = uuid.uuid4().hex[:12]
            current_task = {
                "task_id": task_id,
                "status": "running",
                "created_at": _now(),
                "started_at": _now(),
                "finished_at": None,
                "mode": mode,
                "config_path": str(cfg_path) if cfg_path else None,
                "pid": None,
                "return_code": None,
            }
        try:
            env = dict(os.environ)
            env["PYTHONUTF8"] = "1"
            _ensure_model_cache_env(env)
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
            raise HTTPException(status_code=500, detail=f"启动训练失败: {e}") from e

        with lock:
            proc = p
            if current_task:
                current_task["pid"] = p.pid
                current_task["command"] = subprocess.list2cmdline(cmd)
        append_log(f"[train_ui] 启动训练: {launch_desc}")
        append_log(f"[train_ui] 命令: {subprocess.list2cmdline(cmd)}")
        threading.Thread(target=stream_output, args=(p,), daemon=True).start()
        threading.Thread(target=wait_proc, args=(p, task_id), daemon=True).start()
        return {"ok": True, "task_id": task_id}

    @app.post("/api/train/stop")
    def train_stop():
        nonlocal proc
        with lock:
            if proc is None or proc.poll() is not None:
                raise HTTPException(status_code=400, detail="当前没有运行中的训练任务。")
            if current_task:
                current_task["status"] = "stopping"
            p = proc
        p.terminate()
        append_log("[train_ui] 收到停止请求，正在终止训练进程。")
        return {"ok": True}

    @app.get("/")
    def root():
        if not static_file.exists():
            raise HTTPException(status_code=500, detail=f"Missing UI file: {static_file}")
        return FileResponse(str(static_file))

    return app
