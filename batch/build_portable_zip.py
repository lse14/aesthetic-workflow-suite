from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = APP_ROOT / "batch" / "dist"


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
        description="Build a shareable portable inference zip package (without .venv and model cache)."
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(OUT_DIR / "portable_infer_package.zip"),
        help="Output zip path.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional trained model (.safetensors/.pt) to include in package/models/",
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
        APP_ROOT / "batch" / "run_portable_infer.bat",
        APP_ROOT / "batch" / "sort_images_by_score.py",
        APP_ROOT / "batch" / "config.yaml",
        APP_ROOT / "batch" / "requirements.txt",
        APP_ROOT / "batch" / "runtime" / "batch_infer.py",
        APP_ROOT / "batch" / "runtime" / "prefetch_jtp3.py",
        APP_ROOT / "batch" / "src" / "fusion_scorer" / "__init__.py",
        APP_ROOT / "batch" / "src" / "fusion_scorer" / "extractors.py",
        APP_ROOT / "batch" / "src" / "fusion_scorer" / "model.py",
    ]
    for f in required_files:
        if not f.exists():
            raise FileNotFoundError(f"missing required file: {f}")

    with tempfile.TemporaryDirectory(prefix="portable_infer_build_") as td:
        staging = Path(td) / "portable_infer_package"

        for src in required_files:
            rel = src.relative_to(APP_ROOT)
            _copy_file(src, staging / rel)

        if args.checkpoint:
            ckpt = Path(args.checkpoint).expanduser().resolve()
            if not ckpt.exists() or not ckpt.is_file():
                raise FileNotFoundError(f"checkpoint not found: {ckpt}")
            _copy_file(ckpt, staging / "models" / ckpt.name)

        readme = staging / "README_PORTABLE.txt"
        readme.write_text(
            (
                "Portable Inference Package\n"
                "==========================\n\n"
                "1) Extract this zip.\n"
                "2) Double-click batch\\run_portable_infer.bat\n"
                "3) Fill checkpoint/image-folder/dimension in prompts.\n\n"
                "Tips:\n"
                "- This package intentionally excludes .venv and model cache (_models).\n"
                "- First run may download model weights.\n"
                "- You can pass args directly:\n"
                "  batch\\run_portable_infer.bat --checkpoint <ckpt> --input-dir <dir> --dimension aesthetic\n"
            ),
            encoding="utf-8",
        )

        _zip_dir(staging, out_zip)

    print(f"built: {out_zip}")


if __name__ == "__main__":
    main()
