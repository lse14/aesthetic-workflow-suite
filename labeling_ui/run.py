import argparse
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="启动独立标注 UI 服务。")
    parser.add_argument("--config", type=Path, default=APP_DIR / "config.yaml")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn
    from labeling_pipeline.config import load_config
    from labeling_pipeline.webapp import create_app

    cfg = load_config(args.config if args.config.exists() else None)
    host = args.host or str(cfg["server"]["host"])
    port = args.port or int(cfg["server"]["port"])
    app = create_app(config_path=args.config)
    uvicorn.run(app, host=host, port=port, reload=args.reload, access_log=False)


if __name__ == "__main__":
    main()
