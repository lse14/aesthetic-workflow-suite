from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import threading
from pathlib import Path
from typing import Any

BATCH_ROOT = Path(__file__).resolve().parent
FALLBACK_APP_ROOT = BATCH_ROOT.parent
BATCH_INFER_PATH = BATCH_ROOT / "runtime" / "batch_infer.py"
FALLBACK_BATCH_INFER_PATH = FALLBACK_APP_ROOT / "infer_ui" / "scripts" / "batch_infer.py"
TARGET_DIMS = ("aesthetic", "composition", "color", "sexual")
DEFAULT_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
ZH_EN = {
    "批量图片分拣（单维度）": "Batch Image Sorter (Single Dimension)",
    "请选择模型与图片目录。": "Please select model and image folder.",
    "模型路径": "Model Path",
    "图片文件夹": "Image Folder",
    "打分维度": "Score Dimension",
    "遍历子文件夹": "Recursive",
    "输出文件夹(可选)": "Output Folder (Optional)",
    "推理设备": "Inference Device",
    "批大小(batch_size)": "Batch Size",
    "按显存推荐": "Recommend by VRAM",
    "分桶分制": "Score Scale",
    "目标域阈值": "In-domain Threshold",
    "进度: 0/0": "Progress: 0/0",
    "处理中，请稍候...": "Processing, please wait...",
    "已暂停，点击继续恢复。": "Paused. Click Resume to continue.",
    "正在终止，稍后将仅输出已完成打分的图像。": "Stopping. Only completed images will be exported.",
    "阶段: done 进度: 100%": "Phase: done Progress: 100%",
    "处理完成。": "Completed.",
    "处理失败。": "Failed.",
    "提示": "Tip",
    "完成": "Completed",
    "失败": "Failed",
    "未安装 torch，无法读取显存。": "Torch is not installed, cannot read VRAM.",
    "CPU 模式推荐 batch_size=4。": "CPU mode recommends batch_size=4.",
    "未检测到 CUDA，已回退推荐 batch_size=4。": "CUDA not detected, fallback recommendation batch_size=4.",
    "检测到显存约 ": "Detected VRAM about ",
    "，推荐 batch_size=": ", recommended batch_size=",
    "请先选择模型路径。": "Please select a model path first.",
    "请先选择图片文件夹。": "Please select an image folder first.",
    "请选择合法的打分维度。": "Please select a valid score dimension.",
    "请选择合法的推理设备。": "Please select a valid inference device.",
    "请选择 5 分制或 10 分制。": "Please select score scale 5 or 10.",
    "batch_size 必须是正整数。": "batch_size must be a positive integer.",
    "目标域阈值必须是 0~1 之间的小数。": "Threshold must be a float between 0 and 1.",
    "阶段: ": "Phase: ",
    " 进度: ": " Progress: ",
    "维度: ": "Dimension: ",
    "输出目录: ": "Output: ",
    "分拣目录: ": "Organized: ",
    "统计: ": "Stats: ",
    "任务已终止：仅输出已完成打分的图像结果。": "Task stopped: only completed images are exported.",
    "浏览": "Browse",
    "开始处理": "Start",
    "暂停": "Pause",
    "继续": "Resume",
    "终止": "Stop",
    "选择模型文件": "Select Model File",
    "选择图片文件夹": "Select Image Folder",
    "选择输出文件夹": "Select Output Folder",
}
EN_ZH = {v: k for k, v in ZH_EN.items()}


def _translate_text(text: str, lang: str) -> str:
    out = str(text)
    if lang == "en":
        mapping = ZH_EN
    else:
        mapping = EN_ZH
    for k in sorted(mapping.keys(), key=len, reverse=True):
        out = out.replace(k, mapping[k])
    return out


def _load_batch_infer_module():
    script_path = BATCH_INFER_PATH if BATCH_INFER_PATH.exists() else FALLBACK_BATCH_INFER_PATH
    if not script_path.exists():
        raise FileNotFoundError(f"batch infer script not found: {script_path}")
    spec = importlib.util.spec_from_file_location("infer_batch_infer", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _norm_dim(raw: str) -> str:
    s = str(raw).strip().lower()
    if s not in TARGET_DIMS:
        raise ValueError(f"dimension must be one of: {list(TARGET_DIMS)}")
    return s


def _parse_extensions(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_EXTS)
    out = []
    for x in str(raw).split(","):
        v = x.strip().lower()
        if not v:
            continue
        if not v.startswith("."):
            v = "." + v
        out.append(v)
    return out or list(DEFAULT_EXTS)


def _bucket_strategy_from_scale(scale: str) -> str:
    s = str(scale).strip()
    if s == "10":
        return "x2_floor_10"
    return "floor"


def _resolve_existing_file(path_like: str, *, field: str) -> Path:
    p = Path(path_like).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"{field} not found: {p}")
    return p


def _resolve_existing_dir(path_like: str, *, field: str) -> Path:
    p = Path(path_like).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"{field} not found: {p}")
    return p


def _run_sort(
    *,
    checkpoint_raw: str,
    input_dir_raw: str,
    dimension_raw: str,
    recursive: bool,
    output_dir_raw: str | None,
    organized_root_raw: str | None,
    mode: str,
    bucket_strategy: str,
    device: str | None,
    batch_size: int,
    special_threshold: float,
    image_exts_raw: str | None,
    include_special_group: bool,
    progress_cb=None,
    control=None,
) -> tuple[dict[str, Any], Path, Path, str]:
    checkpoint = _resolve_existing_file(checkpoint_raw, field="checkpoint")
    input_dir = _resolve_existing_dir(input_dir_raw, field="input_dir")
    dim = _norm_dim(dimension_raw)

    if output_dir_raw:
        output_dir = Path(output_dir_raw).expanduser().resolve()
    else:
        output_dir = (BATCH_ROOT / "outputs" / f"batch_sort_{dim}").resolve()
    organized_root = (
        Path(organized_root_raw).expanduser().resolve()
        if organized_root_raw
        else (output_dir / f"organized_by_{dim}").resolve()
    )

    batch_mod = _load_batch_infer_module()
    config_path = (BATCH_ROOT / "config.yaml").resolve()
    if not config_path.exists():
        fallback_cfg = (FALLBACK_APP_ROOT / "infer_ui" / "config.yaml").resolve()
        config_path = fallback_cfg
    if not config_path.exists():
        raise FileNotFoundError(f"infer config not found: {config_path}")

    overrides: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "recursive": bool(recursive),
        "image_extensions": _parse_extensions(image_exts_raw),
        "batch_size": int(batch_size),
        "special_threshold": float(special_threshold),
        "save_jsonl": True,
        "save_csv": True,
        "organize.enabled": True,
        "organize.root_dir": str(organized_root),
        "organize.mode": mode,
        "organize.include_special_group": bool(include_special_group),
        "organize.dimensions": [dim],
        "organize.bucket_strategy": str(bucket_strategy),
    }
    if device is not None and str(device).strip():
        overrides["device"] = str(device).strip()

    run_sig = inspect.signature(batch_mod.run_from_config)
    kwargs: dict[str, Any] = {"overrides": overrides}
    if "progress_cb" in run_sig.parameters:
        kwargs["progress_cb"] = progress_cb
    if "control" in run_sig.parameters:
        kwargs["control"] = control
    summary = batch_mod.run_from_config(config_path, **kwargs)
    return summary, output_dir, organized_root, dim


def _collect_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a trained model and organize images into folders by one selected score dimension."
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model file")
    parser.add_argument("--input-dir", type=str, default=None, help="Image folder to score")
    parser.add_argument("--dimension", type=str, choices=list(TARGET_DIMS), default=None, help="Target score dimension")
    parser.add_argument("--output-dir", type=str, default=None, help="Output folder for csv/jsonl/summary")
    parser.add_argument("--organized-root", type=str, default=None, help="Root folder for sorted images")
    parser.add_argument("--mode", type=str, choices=["copy", "move", "hardlink", "symlink"], default="copy")
    parser.add_argument("--score-scale", type=str, choices=["5", "10"], default="10", help="Score bucket scale. 5=floor(1..5), 10=floor(score*2, 1..10)")
    parser.add_argument("--bucket-strategy", type=str, choices=["nearest_int", "floor", "ceil", "x2_floor_10"], default=None)
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cpu / auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--special-threshold", type=float, default=0.5)
    parser.add_argument("--image-exts", type=str, default=None, help="Comma separated image extensions")
    parser.add_argument("--include-special-group", action="store_true", help="Create in_domain/special top folders")
    parser.add_argument("--no-recursive", action="store_true", help="Disable recursive image scan")
    parser.add_argument("--gui", action="store_true", help="Open simple GUI")
    return parser.parse_args()


def _run_gui() -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as e:
        raise RuntimeError("Tkinter is unavailable on this Python environment.") from e

    root = tk.Tk()
    lang_var = tk.StringVar(value="zh")

    def tr(text: str) -> str:
        return _translate_text(text, lang_var.get())

    root.title(tr("批量图片分拣（单维度）"))
    root.geometry("760x520")
    root.resizable(False, False)

    ckpt_var = tk.StringVar()
    input_var = tk.StringVar()
    output_var = tk.StringVar()
    dim_var = tk.StringVar(value=TARGET_DIMS[0])
    device_var = tk.StringVar(value="cuda")
    batch_var = tk.StringVar(value="8")
    special_threshold_var = tk.StringVar(value="0.5")
    score_scale_var = tk.StringVar(value="10")
    recursive_var = tk.BooleanVar(value=True)
    running_var = tk.BooleanVar(value=False)
    paused_var = tk.BooleanVar(value=False)
    status_var = tk.StringVar(value=tr("请选择模型与图片目录。"))
    pause_event = threading.Event()
    stop_event = threading.Event()

    frm = ttk.Frame(root, padding=12)
    frm.pack(fill=tk.BOTH, expand=True)

    lbl_ckpt = ttk.Label(frm, text=tr("模型路径"))
    lbl_ckpt.grid(row=0, column=0, sticky="w", pady=6)
    ckpt_entry = ttk.Entry(frm, textvariable=ckpt_var, width=72)
    ckpt_entry.grid(row=0, column=1, sticky="ew", padx=6)

    def pick_ckpt() -> None:
        p = filedialog.askopenfilename(
            title=tr("选择模型文件"),
            filetypes=[("Model", "*.safetensors *.pt *.pth *.ckpt"), ("All files", "*.*")],
        )
        if p:
            ckpt_var.set(p)

    lbl_input = ttk.Label(frm, text=tr("图片文件夹"))
    lbl_input.grid(row=1, column=0, sticky="w", pady=6)
    input_entry = ttk.Entry(frm, textvariable=input_var, width=72)
    input_entry.grid(row=1, column=1, sticky="ew", padx=6)

    def pick_input() -> None:
        p = filedialog.askdirectory(title=tr("选择图片文件夹"))
        if p:
            input_var.set(p)

    def pick_output() -> None:
        p = filedialog.askdirectory(title=tr("选择输出文件夹"))
        if p:
            output_var.set(p)

    lbl_dim = ttk.Label(frm, text=tr("打分维度"))
    lbl_dim.grid(row=2, column=0, sticky="w", pady=6)
    dim_combo = ttk.Combobox(frm, textvariable=dim_var, values=list(TARGET_DIMS), state="readonly", width=20)
    dim_combo.grid(row=2, column=1, sticky="w", padx=6)

    recursive_ck = ttk.Checkbutton(frm, text=tr("遍历子文件夹"), variable=recursive_var)
    recursive_ck.grid(row=2, column=1, sticky="e", padx=6)

    lbl_output = ttk.Label(frm, text=tr("输出文件夹(可选)"))
    lbl_output.grid(row=3, column=0, sticky="w", pady=6)
    output_entry = ttk.Entry(frm, textvariable=output_var, width=72)
    output_entry.grid(row=3, column=1, sticky="ew", padx=6)

    lbl_device = ttk.Label(frm, text=tr("推理设备"))
    lbl_device.grid(row=4, column=0, sticky="w", pady=6)
    device_combo = ttk.Combobox(frm, textvariable=device_var, values=["auto", "cuda", "cpu"], state="readonly", width=20)
    device_combo.grid(row=4, column=1, sticky="w", padx=6)

    lbl_batch = ttk.Label(frm, text=tr("批大小(batch_size)"))
    lbl_batch.grid(row=5, column=0, sticky="w", pady=6)
    batch_entry = ttk.Entry(frm, textvariable=batch_var, width=20)
    batch_entry.grid(row=5, column=1, sticky="w", padx=6)

    def recommend_batch_by_vram() -> None:
        try:
            import torch
        except Exception:
            messagebox.showwarning(tr("提示"), tr("未安装 torch，无法读取显存。"))
            return
        dev = device_var.get().strip().lower()
        if dev == "cpu":
            batch_var.set("4")
            status_var.set(tr("CPU 模式推荐 batch_size=4。"))
            return
        if not torch.cuda.is_available():
            batch_var.set("4")
            status_var.set(tr("未检测到 CUDA，已回退推荐 batch_size=4。"))
            return
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_gb <= 6:
            rec = 2
        elif total_gb <= 8:
            rec = 4
        elif total_gb <= 12:
            rec = 8
        elif total_gb <= 16:
            rec = 12
        elif total_gb <= 24:
            rec = 16
        elif total_gb <= 32:
            rec = 24
        else:
            rec = 32
        batch_var.set(str(rec))
        status_var.set(tr(f"检测到显存约 {total_gb:.1f} GB，推荐 batch_size={rec}。"))

    btn_recommend_batch = ttk.Button(frm, text=tr("按显存推荐"), command=recommend_batch_by_vram)
    btn_recommend_batch.grid(row=5, column=2, sticky="ew")

    lbl_scale = ttk.Label(frm, text=tr("分桶分制"))
    lbl_scale.grid(row=6, column=0, sticky="w", pady=6)
    score_scale_combo = ttk.Combobox(frm, textvariable=score_scale_var, values=["5", "10"], state="readonly", width=20)
    score_scale_combo.grid(row=6, column=1, sticky="w", padx=6)

    lbl_threshold = ttk.Label(frm, text=tr("目标域阈值"))
    lbl_threshold.grid(row=7, column=0, sticky="w", pady=6)
    special_threshold_entry = ttk.Entry(frm, textvariable=special_threshold_var, width=20)
    special_threshold_entry.grid(row=7, column=1, sticky="w", padx=6)

    progress_var = tk.DoubleVar(value=0.0)
    progress_text_var = tk.StringVar(value=tr("进度: 0/0"))
    progress_bar = ttk.Progressbar(frm, variable=progress_var, maximum=100.0, mode="determinate")
    progress_bar.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(6, 2))
    progress_text_lbl = ttk.Label(frm, textvariable=progress_text_var, foreground="#334155")
    progress_text_lbl.grid(row=9, column=0, columnspan=3, sticky="w")

    status_lbl = ttk.Label(frm, textvariable=status_var, foreground="#334155")
    status_lbl.grid(row=10, column=0, columnspan=3, sticky="w", pady=(8, 4))

    frm.columnconfigure(1, weight=1)

    widgets = [
        ckpt_entry,
        input_entry,
        output_entry,
        dim_combo,
        device_combo,
        batch_entry,
        score_scale_combo,
        special_threshold_entry,
        recursive_ck,
    ]

    def set_running(r: bool) -> None:
        running_var.set(r)
        if not r:
            paused_var.set(False)
        state = "disabled" if r else "normal"
        for w in widgets:
            try:
                w.configure(state=state if w is not dim_combo else ("disabled" if r else "readonly"))
            except Exception:
                pass
        btn_pick_ckpt.configure(state=state)
        btn_pick_input.configure(state=state)
        btn_pick_output.configure(state=state)
        btn_recommend_batch.configure(state=state)
        btn_run.configure(state=state)
        btn_pause.configure(state=("normal" if r and not paused_var.get() else "disabled"))
        btn_resume.configure(state=("normal" if r and paused_var.get() else "disabled"))
        btn_stop.configure(state=("normal" if r else "disabled"))

    def sync_pause_resume_state() -> None:
        if not running_var.get():
            btn_pause.configure(state="disabled")
            btn_resume.configure(state="disabled")
            return
        btn_pause.configure(state=("disabled" if paused_var.get() else "normal"))
        btn_resume.configure(state=("normal" if paused_var.get() else "disabled"))

    def pause_job() -> None:
        if not running_var.get() or paused_var.get():
            return
        pause_event.set()
        paused_var.set(True)
        status_var.set(tr("已暂停，点击继续恢复。"))
        sync_pause_resume_state()

    def resume_job() -> None:
        if not running_var.get() or not paused_var.get():
            return
        pause_event.clear()
        paused_var.set(False)
        status_var.set(tr("处理中，请稍候..."))
        sync_pause_resume_state()

    def stop_job() -> None:
        if not running_var.get():
            return
        stop_event.set()
        pause_event.clear()
        paused_var.set(False)
        status_var.set(tr("正在终止，稍后将仅输出已完成打分的图像。"))
        sync_pause_resume_state()

    def handle_result(ok: bool, payload: str) -> None:
        set_running(False)
        if ok:
            progress_var.set(100.0)
            progress_text_var.set(tr("阶段: done 进度: 100%"))
            status_var.set(tr("处理完成。"))
            messagebox.showinfo(tr("完成"), tr(payload))
        else:
            status_var.set(tr("处理失败。"))
            messagebox.showerror(tr("失败"), tr(payload))

    def run_job() -> None:
        if running_var.get():
            return
        checkpoint_raw = ckpt_var.get().strip()
        input_dir_raw = input_var.get().strip()
        output_dir_raw = output_var.get().strip()
        dim = dim_var.get().strip()
        dev = device_var.get().strip().lower()
        score_scale = score_scale_var.get().strip()
        batch_raw = batch_var.get().strip()
        special_threshold_raw = special_threshold_var.get().strip()
        if not checkpoint_raw:
            messagebox.showwarning(tr("提示"), tr("请先选择模型路径。"))
            return
        if not input_dir_raw:
            messagebox.showwarning(tr("提示"), tr("请先选择图片文件夹。"))
            return
        if dim not in TARGET_DIMS:
            messagebox.showwarning(tr("提示"), tr("请选择合法的打分维度。"))
            return
        if dev not in {"auto", "cuda", "cpu"}:
            messagebox.showwarning(tr("提示"), tr("请选择合法的推理设备。"))
            return
        if score_scale not in {"5", "10"}:
            messagebox.showwarning(tr("提示"), tr("请选择 5 分制或 10 分制。"))
            return
        try:
            batch_size = int(batch_raw)
            if batch_size <= 0:
                raise ValueError
        except Exception:
            messagebox.showwarning(tr("提示"), tr("batch_size 必须是正整数。"))
            return
        try:
            special_threshold = float(special_threshold_raw)
            if special_threshold < 0.0 or special_threshold > 1.0:
                raise ValueError
        except Exception:
            messagebox.showwarning(tr("提示"), tr("目标域阈值必须是 0~1 之间的小数。"))
            return

        set_running(True)
        pause_event.clear()
        stop_event.clear()
        paused_var.set(False)
        sync_pause_resume_state()
        status_var.set(tr("处理中，请稍候..."))
        progress_var.set(0.0)
        progress_text_var.set(tr("进度: 0/0"))

        def worker() -> None:
            try:
                def on_progress(payload: dict[str, Any]) -> None:
                    done = int(payload.get("done", 0) or 0)
                    total = int(payload.get("total", 0) or 0)
                    eta_sec = int(payload.get("eta_sec", -1) or -1)
                    phase = str(payload.get("phase", "infer"))
                    pct = (done * 100.0 / total) if total > 0 else 0.0
                    if eta_sec >= 0:
                        eta_txt = f"{eta_sec // 60:02d}:{eta_sec % 60:02d}"
                    else:
                        eta_txt = "--:--"
                    root.after(
                        0,
                        lambda: (
                            progress_var.set(max(0.0, min(100.0, pct))),
                            progress_text_var.set(tr(f"阶段: {phase} 进度: {done}/{total} ETA: {eta_txt}")),
                        ),
                    )

                summary, output_dir, organized_root, dim_name = _run_sort(
                    checkpoint_raw=checkpoint_raw,
                    input_dir_raw=input_dir_raw,
                    dimension_raw=dim,
                    recursive=bool(recursive_var.get()),
                    output_dir_raw=output_dir_raw or None,
                    organized_root_raw=None,
                    mode="copy",
                    bucket_strategy=_bucket_strategy_from_scale(score_scale),
                    device=None if dev == "auto" else dev,
                    batch_size=batch_size,
                    special_threshold=special_threshold,
                    image_exts_raw=None,
                    include_special_group=False,
                    progress_cb=on_progress,
                    control={"pause_event": pause_event, "stop_event": stop_event},
                )
                stopped = bool(summary.get("stopped", False))
                stop_note = "\n任务已终止：仅输出已完成打分的图像结果。" if stopped else ""
                msg = (
                    f"维度: {dim_name}\n"
                    f"输出目录: {output_dir}\n"
                    f"分拣目录: {organized_root}\n"
                    f"统计: {json.dumps(summary.get('organize', {}), ensure_ascii=False)}"
                    f"{stop_note}"
                )
                root.after(0, lambda: handle_result(True, msg))
            except Exception as e:
                err_msg = str(e)
                root.after(0, lambda m=err_msg: handle_result(False, m))

        threading.Thread(target=worker, daemon=True).start()

    btn_pick_ckpt = ttk.Button(frm, text=tr("浏览"), command=pick_ckpt)
    btn_pick_ckpt.grid(row=0, column=2, sticky="ew")
    btn_pick_input = ttk.Button(frm, text=tr("浏览"), command=pick_input)
    btn_pick_input.grid(row=1, column=2, sticky="ew")
    btn_pick_output = ttk.Button(frm, text=tr("浏览"), command=pick_output)
    btn_pick_output.grid(row=3, column=2, sticky="ew")
    btn_run = ttk.Button(frm, text=tr("开始处理"), command=run_job)
    btn_run.grid(row=11, column=0, columnspan=3, sticky="ew", pady=(8, 0))
    btn_pause = ttk.Button(frm, text=tr("暂停"), command=pause_job, state="disabled")
    btn_pause.grid(row=12, column=0, sticky="ew", pady=(6, 0))
    btn_resume = ttk.Button(frm, text=tr("继续"), command=resume_job, state="disabled")
    btn_resume.grid(row=12, column=1, sticky="ew", padx=6, pady=(6, 0))
    btn_stop = ttk.Button(frm, text=tr("终止"), command=stop_job, state="disabled")
    btn_stop.grid(row=12, column=2, sticky="ew", pady=(6, 0))

    def toggle_lang() -> None:
        lang_var.set("en" if lang_var.get() == "zh" else "zh")
        root.title(tr("批量图片分拣（单维度）"))
        lbl_ckpt.configure(text=tr("模型路径"))
        lbl_input.configure(text=tr("图片文件夹"))
        lbl_dim.configure(text=tr("打分维度"))
        recursive_ck.configure(text=tr("遍历子文件夹"))
        lbl_output.configure(text=tr("输出文件夹(可选)"))
        lbl_device.configure(text=tr("推理设备"))
        lbl_batch.configure(text=tr("批大小(batch_size)"))
        btn_recommend_batch.configure(text=tr("按显存推荐"))
        lbl_scale.configure(text=tr("分桶分制"))
        lbl_threshold.configure(text=tr("目标域阈值"))
        btn_pick_ckpt.configure(text=tr("浏览"))
        btn_pick_input.configure(text=tr("浏览"))
        btn_pick_output.configure(text=tr("浏览"))
        btn_run.configure(text=tr("开始处理"))
        btn_pause.configure(text=tr("暂停"))
        btn_resume.configure(text=tr("继续"))
        btn_stop.configure(text=tr("终止"))
        btn_lang.configure(text=("中文" if lang_var.get() == "en" else "EN"))
        status_var.set(tr("请选择模型与图片目录。"))
        progress_text_var.set(tr("进度: 0/0"))

    btn_lang = ttk.Button(frm, text="EN", command=toggle_lang)
    btn_lang.grid(row=13, column=2, sticky="e", pady=(6, 0))

    root.mainloop()


def main() -> None:
    args = _collect_args()
    if args.gui:
        _run_gui()
        return

    if not args.checkpoint:
        raise ValueError("missing --checkpoint (or use --gui)")
    if not args.input_dir:
        raise ValueError("missing --input-dir (or use --gui)")
    if not args.dimension:
        raise ValueError("missing --dimension (or use --gui)")

    bucket_strategy = args.bucket_strategy or _bucket_strategy_from_scale(args.score_scale)
    summary, output_dir, organized_root, dim = _run_sort(
        checkpoint_raw=args.checkpoint,
        input_dir_raw=args.input_dir,
        dimension_raw=args.dimension,
        recursive=not bool(args.no_recursive),
        output_dir_raw=args.output_dir,
        organized_root_raw=args.organized_root,
        mode=args.mode,
        bucket_strategy=bucket_strategy,
        device=args.device,
        batch_size=int(args.batch_size),
        special_threshold=float(args.special_threshold),
        image_exts_raw=args.image_exts,
        include_special_group=bool(args.include_special_group),
    )
    print("\n=== 完成 ===")
    print(f"维度: {dim}")
    print(f"输出目录: {output_dir}")
    print(f"分拣目录: {organized_root}")
    print(f"统计: {json.dumps(summary.get('organize', {}), ensure_ascii=False)}")
    print(f"summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
