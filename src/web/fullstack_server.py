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

import pyarrow.parquet as pq

from src.config.constants import OPERATING_MODES, WINDOWS_PER_DAY


REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = REPO_ROOT / "web" / "frontend"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output"


def _demo_summary_payload() -> dict:
    """Return demo-safe payload when no generated parquet dataset is available."""
    return {
        "fleet_size": 200,
        "windows_per_day": WINDOWS_PER_DAY,
        "failure_modes": 8,
        "operating_modes": OPERATING_MODES,
        "feature_count": 221,
        "output_columns": 229,
        "source": "demo",
        "data_files": 0,
        "rows_total": 0,
        "message": "No parquet output found yet. Run generator to switch to live summary.",
    }


def _build_live_summary_from_output(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict:
    """Build summary from generated parquet data, fallback to demo if missing.

    Reads up to 20 latest parquet files for lightweight startup latency.
    """
    parquet_files = sorted(output_dir.glob("**/truck_*/day_*.parquet"))
    if not parquet_files:
        return _demo_summary_payload()

    sample_files = parquet_files[-20:]

    truck_ids: set[int] = set()
    fault_modes: set[str] = set()
    all_columns: set[str] = set()
    rows_total = 0

    for parquet_file in sample_files:
        table = pq.read_table(parquet_file)
        df = table.to_pandas()
        rows_total += len(df)
        all_columns.update(df.columns.tolist())

        if "truck_id" in df.columns:
            truck_ids.update(df["truck_id"].dropna().astype(int).tolist())

        if "fault_mode" in df.columns:
            for value in df["fault_mode"].dropna().astype(str).unique().tolist():
                if value and value.upper() != "HEALTHY":
                    fault_modes.add(value)

    metadata_columns = {"timestamp", "truck_id", "engine_type", "day_index"}
    label_columns = {"fault_mode", "fault_severity", "rul_hours", "path_a_label"}
    feature_columns = all_columns - metadata_columns - label_columns

    windows_per_day = WINDOWS_PER_DAY
    if sample_files:
        # Estimate from median row count over sampled files if present.
        row_counts = []
        for parquet_file in sample_files:
            row_counts.append(pq.read_metadata(parquet_file).num_rows)
        if row_counts:
            row_counts = sorted(row_counts)
            windows_per_day = int(row_counts[len(row_counts) // 2])

    return {
        "fleet_size": len(truck_ids),
        "windows_per_day": windows_per_day,
        "failure_modes": len(fault_modes),
        "operating_modes": OPERATING_MODES,
        "feature_count": len(feature_columns),
        "output_columns": len(all_columns),
        "source": "live",
        "data_files": len(parquet_files),
        "rows_total": rows_total,
        "message": "Summary computed from generated parquet output.",
    }


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
            payload = _build_live_summary_from_output()
            self._send_json(payload)
            return

        if route == "/api/guide":
            payload = {
                "pipeline_steps": [
                    "1) CLI приема параметри (брой камиони, дни, seed, output).",
                    "2) Fleet factory създава профили за камионите.",
                    "3) Fault schedule присвоява сценарии с повреди.",
                    "4) TruckDayGenerator симулира деня (window-by-window).",
                    "5) Features модулът извлича vibration/thermal/conditioning признаци.",
                    "6) Labels модулът създава ground-truth таргети (fault stage, severity, RUL).",
                    "7) Parquet writer записва финалния dataset.",
                    "8) Validation checks проверяват диапазони, progression и cross-feature логика.",
                ],
                "final_deliverables": [
                    "Работещ генератор на синтетични данни за fleet predictive maintenance.",
                    "Стабилен parquet dataset с фиксирана схема и коректни labels.",
                    "Доказано качество чрез тестове + валидатори.",
                    "GUI/демо, което ясно показва какво се генерира и как се интерпретира.",
                ],
                "run_commands": [
                    {"title": "1) Инсталация", "command": "pip install -r requirements.txt"},
                    {"title": "2) Пускане на тестове", "command": "pytest tests/ -v"},
                    {"title": "3) Стартиране на GUI + API", "command": "python run_front_and_backend.py"},
                    {"title": "4) Smoke генерация (1 truck × 1 day)", "command": "python -m src.generator.cli --single-truck 1 --single-day 0 --output-dir output/test/"},
                ],
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
    print("Frontend: /app/ | API: /api/health, /api/summary, /api/guide")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
