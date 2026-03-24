import argparse
import json
import os
from pathlib import Path


def _repo_to_dirname(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def _exists_jtp3_local(repo_dir: Path) -> bool:
    return (repo_dir / "model.py").exists() and (repo_dir / "models" / "jtp-3-hydra.safetensors").exists()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download JTP-3 base repo into infer_ui local model cache."
    )
    parser.add_argument("--repo-id", default="RedRocket/JTP-3")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "_models")
    parser.add_argument("--local-repo-dir", type=Path, default=None)
    parser.add_argument("--token", default=None, help="HF token. Default reads HF_TOKEN env.")
    parser.add_argument("--show-progress", action="store_true", default=True)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    local_repo_dir = (
        args.local_repo_dir.resolve()
        if args.local_repo_dir is not None
        else (root / "repos" / _repo_to_dirname(args.repo_id)).resolve()
    )
    hf_home = (root / "hf_home").resolve()
    hf_cache = (hf_home / "hub").resolve()

    root.mkdir(parents=True, exist_ok=True)
    local_repo_dir.parent.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    hf_cache.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    show_progress = args.show_progress and (not args.no_progress)
    if show_progress:
        os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
    else:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    if _exists_jtp3_local(local_repo_dir) and not args.force:
        print("[prefetch_jtp3] JTP-3 already exists, skip download.")
    else:
        from huggingface_hub import snapshot_download

        token = args.token or os.getenv("HF_TOKEN")
        print(f"[prefetch_jtp3] repo: {args.repo_id}")
        print(f"[prefetch_jtp3] local_repo_dir: {local_repo_dir}")
        print(f"[prefetch_jtp3] HF_HOME: {hf_home}")
        print(f"[prefetch_jtp3] HF_HUB_CACHE: {hf_cache}")
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=str(local_repo_dir),
            token=token,
        )

    if not _exists_jtp3_local(local_repo_dir):
        raise FileNotFoundError(
            "Download finished but JTP-3 local files are incomplete: "
            f"{local_repo_dir}\\model.py or {local_repo_dir}\\models\\jtp-3-hydra.safetensors missing."
        )

    summary = {
        "root": str(root),
        "hf_home": str(hf_home),
        "hf_cache": str(hf_cache),
        "repo_id": args.repo_id,
        "local_repo_dir": str(local_repo_dir),
        "model_py": str(local_repo_dir / "model.py"),
        "weights": str(local_repo_dir / "models" / "jtp-3-hydra.safetensors"),
    }
    summary_path = root / "prefetch_jtp3_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[prefetch_jtp3] summary: {summary_path}")
    print("[prefetch_jtp3] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
