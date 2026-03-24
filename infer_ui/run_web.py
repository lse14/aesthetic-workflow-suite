import argparse
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="启动独立批量推理 UI 服务。")
    parser.add_argument("--config", type=Path, default=APP_DIR / "config.yaml")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn
    from app import create_app, load_config

    cfg = load_config(args.config)
    host = args.host or str(cfg["server"]["host"])
    port = args.port or int(cfg["server"]["port"])
    app = create_app(args.config)
    uvicorn.run(app, host=host, port=port, reload=args.reload, access_log=False)


if __name__ == "__main__":
    main()
