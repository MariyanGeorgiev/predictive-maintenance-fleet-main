"""Minimal full-stack server for local demo UI.

Serves:
- Frontend static assets under /app/
- Backend API endpoints under /api/
- Redirects / to /app/
"""

from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from src.config.constants import OPERATING_MODES, WINDOWS_PER_DAY


REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = REPO_ROOT / "web" / "frontend"


class FullStackHandler(BaseHTTPRequestHandler):
    """Serve API + static frontend."""

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        content = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path

        if route == "/":
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/app/")
            self.end_headers()
            return

        if route == "/api/health":
            self._send_json({"status": "ok", "service": "predictive-maintenance-fullstack"})
            return

        if route == "/api/summary":
            payload = {
                "fleet_size": 200,
                "windows_per_day": WINDOWS_PER_DAY,
                "failure_modes": 8,
                "operating_modes": OPERATING_MODES,
                "feature_count": 221,
                "output_columns": 229,
            }
            self._send_json(payload)
            return

        if route in {"/app", "/app/"}:
            self._serve_file(FRONTEND_DIR / "index.html")
            return

        if route.startswith("/app/"):
            relative = route.replace("/app/", "", 1)
            requested = (FRONTEND_DIR / relative).resolve()
            if FRONTEND_DIR not in requested.parents and requested != FRONTEND_DIR:
                self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
                return
            self._serve_file(requested)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")


def run_server(host: str = "127.0.0.1", port: int = 8787) -> None:
    """Run the full-stack HTTP server."""
    server = ThreadingHTTPServer((host, port), FullStackHandler)

    access_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    print(f"Full-stack server running at http://{host}:{port}")
    print(f"Open in browser: http://{access_host}:{port}/")
    print("Frontend: /app/ | API: /api/health, /api/summary")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
