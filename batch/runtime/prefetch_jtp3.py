from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _repo_to_dirname(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def _exists_jtp3_local(repo_dir: Path) -> bool:
    return (repo_dir / "model.py").exists() and (repo_dir / "models" / "jtp-3-hydra.safetensors").exists()


def main() -> int:
    parser = argparse.ArgumentParser(description="Prefetch JTP-3 repo into local model cache.")
    parser.add_argument("--repo-id", default="RedRocket/JTP-3")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-prefetch-openclip", action="store_true")
    parser.add_argument("--no-prefetch-waifu-head", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    local_repo_dir = (root / "repos" / _repo_to_dirname(args.repo_id)).resolve()
    hf_home = (root / "hf_home").resolve()
    hf_cache = (hf_home / "hub").resolve()

    root.mkdir(parents=True, exist_ok=True)
    local_repo_dir.parent.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    hf_cache.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if args.no_progress:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    else:
        os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
    token = os.getenv("HF_TOKEN")

    if _exists_jtp3_local(local_repo_dir) and not args.force:
        print("[prefetch_jtp3] JTP-3 already exists, skip.")
    else:
        from huggingface_hub import snapshot_download

        print(f"[prefetch_jtp3] repo: {args.repo_id}")
        print(f"[prefetch_jtp3] local_repo_dir: {local_repo_dir}")
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=str(local_repo_dir),
            token=token,
        )

    if not args.no_prefetch_openclip:
        from huggingface_hub import hf_hub_download

        openclip_repo = "timm/vit_large_patch14_clip_224.openai"
        openclip_file = "open_clip_model.safetensors"
        print(f"[prefetch_jtp3] prefetch open-clip: {openclip_repo}/{openclip_file}")
        hf_hub_download(
            repo_id=openclip_repo,
            filename=openclip_file,
            token=token,
        )

    if not args.no_prefetch_waifu_head:
        from huggingface_hub import hf_hub_download
        import shutil

        waifu_repo = "Eugeoter/waifu-scorer-v3"
        waifu_file = "model.safetensors"
        waifu_local_dir = (root / "waifu-scorer-v3").resolve()
        waifu_local_dir.mkdir(parents=True, exist_ok=True)
        waifu_local_path = waifu_local_dir / waifu_file

        print(f"[prefetch_jtp3] prefetch waifu head: {waifu_repo}/{waifu_file}")
        downloaded = Path(
            hf_hub_download(
                repo_id=waifu_repo,
                filename=waifu_file,
                token=token,
            )
        )
        if not waifu_local_path.exists() or args.force:
            shutil.copy2(downloaded, waifu_local_path)

    if not _exists_jtp3_local(local_repo_dir):
        raise FileNotFoundError(
            "JTP-3 local files incomplete: "
            f"{local_repo_dir / 'model.py'} or {local_repo_dir / 'models' / 'jtp-3-hydra.safetensors'} missing."
        )

    summary = {
        "root": str(root),
        "repo_id": args.repo_id,
        "local_repo_dir": str(local_repo_dir),
        "model_py": str(local_repo_dir / "model.py"),
        "weights": str(local_repo_dir / "models" / "jtp-3-hydra.safetensors"),
        "openclip_repo": "timm/vit_large_patch14_clip_224.openai",
        "openclip_file": "open_clip_model.safetensors",
        "waifu_repo": "Eugeoter/waifu-scorer-v3",
        "waifu_head": str((root / "waifu-scorer-v3" / "model.safetensors").resolve()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("[prefetch_jtp3] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
