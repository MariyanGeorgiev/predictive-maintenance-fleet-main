"""Run frontend + backend and open browser with automatic redirect.

Usage:
    python run_front_and_backend.py
"""

from __future__ import annotations

import argparse
import threading
import time
import webbrowser

from src.web.fullstack_server import run_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-stack demo and redirect user to frontend.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8787, help="Port to bind")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open browser",
    )
    args = parser.parse_args()

    if not args.no_browser:
        browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host

        def open_browser() -> None:
            time.sleep(0.6)
            webbrowser.open(f"http://{browser_host}:{args.port}/")

        threading.Thread(target=open_browser, daemon=True).start()

    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
