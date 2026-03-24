import argparse
import time
import urllib.request
import webbrowser


def is_ready(health_url: str, timeout_sec: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(health_url, timeout=timeout_sec) as resp:
            return int(getattr(resp, "status", 0)) == 200
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Open browser when health endpoint is ready.")
    parser.add_argument("--url", required=True, help="Final UI URL to open.")
    parser.add_argument("--health-url", required=True, help="Health check URL.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Max wait seconds.")
    parser.add_argument("--interval", type=float, default=0.8, help="Polling interval seconds.")
    args = parser.parse_args()

    deadline = time.time() + max(1.0, float(args.timeout))
    interval = max(0.1, float(args.interval))
    while time.time() < deadline:
        if is_ready(args.health_url):
            webbrowser.open(args.url, new=2)
            return
        time.sleep(interval)


if __name__ == "__main__":
    main()
