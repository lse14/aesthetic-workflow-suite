from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
OUT_DIR = APP_DIR / "dist"


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _zip_dir(src_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in src_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src_dir))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a minimal portable Infer UI zip package (without .venv/_models/outputs)."
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(OUT_DIR / "portable_infer_ui.zip"),
        help="Output zip path.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint to include in package/models/.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output zip if exists.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_zip = Path(args.output).expanduser().resolve()
    if out_zip.exists() and not args.force:
        raise FileExistsError(f"output exists: {out_zip} (use --force)")

    required_files = [
        APP_DIR / "start.bat",
        APP_DIR / "run.py",
        APP_DIR / "run_web.py",
        APP_DIR / "app.py",
        APP_DIR / "config.yaml",
        APP_DIR / "requirements.txt",
        APP_DIR / "README_SHARE_CN.txt",
        APP_DIR / "prefetch_jtp3.bat",
        APP_DIR / "scripts" / "resolve_webui_port.py",
        APP_DIR / "scripts" / "batch_infer.py",
        APP_DIR / "scripts" / "prefetch_jtp3.py",
        APP_DIR / "src" / "fusion_scorer" / "__init__.py",
        APP_DIR / "src" / "fusion_scorer" / "extractors.py",
        APP_DIR / "src" / "fusion_scorer" / "model.py",
        APP_DIR / "static" / "index.html",
    ]
    for f in required_files:
        if not f.exists():
            raise FileNotFoundError(f"missing required file: {f}")

    with tempfile.TemporaryDirectory(prefix="portable_infer_ui_build_") as td:
        staging = Path(td) / "portable_infer_ui"
        app_stage = staging / "infer_ui"
        for src in required_files:
            rel = src.relative_to(APP_DIR)
            _copy_file(src, app_stage / rel)

        if args.checkpoint:
            ckpt = Path(args.checkpoint).expanduser().resolve()
            if not ckpt.exists() or not ckpt.is_file():
                raise FileNotFoundError(f"checkpoint not found: {ckpt}")
            _copy_file(ckpt, app_stage / "models" / ckpt.name)

        readme = staging / "README_PORTABLE.txt"
        readme.write_text(
            (
                "Portable Infer UI Package\n"
                "=========================\n\n"
                "1) Extract this zip.\n"
                "2) Open folder infer_ui.\n"
                "3) Double-click start.bat\n"
                "4) In browser, set checkpoint path and run.\n\n"
                "Notes:\n"
                "- This package excludes .venv, _models and outputs.\n"
                "- First run installs dependencies.\n"
                "- First model load may download weights.\n"
            ),
            encoding="utf-8",
        )

        _zip_dir(staging, out_zip)

    print(f"built: {out_zip}")


if __name__ == "__main__":
    main()
