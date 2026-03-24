import argparse
import socket
from pathlib import Path

import yaml


def can_bind(host: str, port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def normalize_host(host: str) -> str:
    h = (host or "").strip()
    if not h:
        return "127.0.0.1"
    return h


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve host/port for WebUI startup.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--default-host", type=str, default="127.0.0.1")
    parser.add_argument("--default-port", type=int, default=9100)
    parser.add_argument("--max-offset", type=int, default=200)
    args = parser.parse_args()

    host = args.default_host
    port = int(args.default_port)

    if args.config.exists():
        with args.config.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        server = cfg.get("server", {}) if isinstance(cfg, dict) else {}
        host = normalize_host(str(server.get("host", host)))
        try:
            port = int(server.get("port", port))
        except Exception:
            port = int(args.default_port)

    if port < 1 or port > 65535:
        port = int(args.default_port)

    selected = None
    for p in range(port, min(port + args.max_offset + 1, 65536)):
        if can_bind(host, p):
            selected = p
            break

    if selected is None:
        selected = int(args.default_port)

    # stdout for batch parser: "<host> <port>"
    print(f"{host} {selected}")


if __name__ == "__main__":
    main()
