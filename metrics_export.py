#!/usr/bin/env python3
"""metrics_export.py — Prometheus metrics exporter for Evolution Engine.

Exposes cycle metrics as Prometheus gauges/counters via a threaded HTTP
server with the following endpoints:

    /metrics  — Prometheus text format (prometheus_client.generate_latest)
    /health   — JSON health check (status, fitness, cycle, uptime, last_error)
    /status   — JSON full engine state (all health fields plus detailed metrics)

GPU utilization is detected via pynvml with fallback to 0.0.

Usage:
    python3 -m metrics_export  # starts server on port 9100

    # or programmatically:
    from metrics_export import create_metrics_app, update_metrics, stop_metrics_app
    server = create_metrics_app(port=9100)
    update_metrics(cycle_data)
    stop_metrics_app(server)
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any

from prometheus_client import Counter, Gauge, generate_latest

# ── Prometheus metric definitions ───────────────────────────────────

evolution_fitness = Gauge("evolution_fitness", "Latest fitness value")
evolution_level = Gauge("evolution_level", "Latest curriculum level")
evolution_cycle_time_seconds = Gauge("evolution_cycle_time_seconds", "Time per cycle in seconds")
evolution_gpu_utilization = Gauge("evolution_gpu_utilization", "GPU utilization percentage (0 if no GPU)")
evolution_population_size = Gauge("evolution_population_size", "Population size")
evolution_cycle_total = Counter("evolution_cycle_total", "Total cycles completed")
evolution_novelty_archive_size = Gauge("evolution_novelty_archive_size", "Novelty archive size")
evolution_qd_occupied_cells = Gauge("evolution_qd_occupied_cells", "QD occupied cells")
evolution_champion_traces = Gauge("evolution_champion_traces", "Champion traces count")
evolution_diversity = Gauge("evolution_diversity", "Diversity metric")

# ── Module-level state tracking ─────────────────────────────────────

_start_time: float = time.time()
_last_error: str = ""
_last_update: str = ""

# ── GPU utilization detection ───────────────────────────────────────

_nvml_handle: Any = None


def _get_gpu_utilization() -> float:
    """Get GPU utilization percentage via pynvml, fallback to 0.0.

    Caches the NVML handle for efficiency. Handles missing pynvml gracefully.
    """
    global _nvml_handle
    try:
        if _nvml_handle is None:
            import pynvml

            pynvml.nvmlInit()
            _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle)
        return float(util.gpu)
    except Exception:
        return 0.0


# ── Custom threaded HTTP server ─────────────────────────────────────


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Threaded HTTP server that handles each request in a new daemon thread."""

    daemon_threads = True


class _MetricsHandler(BaseHTTPRequestHandler):
    """Request handler serving /metrics, /health, /status, and 404."""

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default stderr logging."""

    # -- helpers -------------------------------------------------------

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        """Send a JSON response with correct headers."""
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status_code: int, content_type: str, body: bytes) -> None:
        """Send a raw-text response with correct headers."""
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    # -- route handlers ------------------------------------------------

    def do_GET(self) -> None:
        path = self.path.rstrip("/") or "/"

        if path == "/metrics":
            self._handle_metrics()
        elif path == "/health":
            self._handle_health()
        elif path == "/status":
            self._handle_status()
        else:
            self._handle_not_found()

    def do_OPTIONS(self) -> None:
        """Respond to CORS preflight requests."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    # -- endpoint implementations --------------------------------------

    def _handle_metrics(self) -> None:
        body = generate_latest()
        self._send_text(200, "text/plain; version=0.0.4; charset=utf-8", body)

    def _handle_health(self) -> None:
        fitness = evolution_fitness._value.get()
        payload = {
            "status": "healthy" if fitness > 0 else "degraded",
            "fitness": fitness,
            "cycle": int(evolution_cycle_total._value.get()),
            "uptime": round(time.time() - _start_time, 2),
            "last_error": _last_error,
        }
        self._send_json(200, payload)

    def _handle_status(self) -> None:
        fitness = evolution_fitness._value.get()
        payload = {
            # health fields
            "status": "healthy" if fitness > 0 else "degraded",
            "fitness": fitness,
            "cycle": int(evolution_cycle_total._value.get()),
            "uptime": round(time.time() - _start_time, 2),
            "last_error": _last_error,
            # detailed engine state
            "level": evolution_level._value.get(),
            "cycle_time_seconds": evolution_cycle_time_seconds._value.get(),
            "gpu_utilization": evolution_gpu_utilization._value.get(),
            "population_size": evolution_population_size._value.get(),
            "novelty_archive_size": evolution_novelty_archive_size._value.get(),
            "qd_occupied_cells": evolution_qd_occupied_cells._value.get(),
            "champion_traces": evolution_champion_traces._value.get(),
            "diversity": evolution_diversity._value.get(),
            "last_update": _last_update,
        }
        self._send_json(200, payload)

    def _handle_not_found(self) -> None:
        payload = {"error": "not found", "path": self.path}
        self._send_json(404, payload)


# ── Metrics app management ──────────────────────────────────────────


def create_metrics_app(port: int = 9100) -> Any:
    """Start a threaded HTTP server exposing /metrics, /health, and /status.

    Returns a (server, thread) tuple (for later stopping). The server thread
    runs as a daemon so it doesn't block process exit.

    Endpoints:
        /metrics  — Prometheus text format (Content-Type: text/plain)
        /health   — JSON with status, fitness, cycle, uptime, last_error
        /status   — JSON with all health fields plus full engine state

    Args:
        port: HTTP port to listen on.

    Returns:
        Tuple of (HTTPServer, daemon_thread).
    """
    server = _ThreadingHTTPServer(("0.0.0.0", port), _MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return (server, thread)


def update_metrics(cycle_data: dict[str, Any]) -> None:
    """Update all Prometheus gauges from a cycle results dict.

    Keys expected (missing keys are silently skipped):
        final_fitness, final_level, elapsed_seconds, population_size,
        novelty_archive_size, qd_occupied_cells, champion_traces, diversity,
        last_error

    Args:
        cycle_data: Cycle results dict (matches run_evolution() return format).
    """
    global _last_error, _last_update

    if "final_fitness" in cycle_data:
        evolution_fitness.set(cycle_data["final_fitness"])
    if "final_level" in cycle_data:
        evolution_level.set(cycle_data["final_level"])
    if "elapsed_seconds" in cycle_data:
        evolution_cycle_time_seconds.set(cycle_data["elapsed_seconds"])

    # GPU utilization
    evolution_gpu_utilization.set(_get_gpu_utilization())

    if "population_size" in cycle_data:
        evolution_population_size.set(cycle_data["population_size"])
    if "novelty_archive_size" in cycle_data:
        evolution_novelty_archive_size.set(cycle_data["novelty_archive_size"])
    if "qd_occupied_cells" in cycle_data:
        evolution_qd_occupied_cells.set(cycle_data["qd_occupied_cells"])
    if "champion_traces" in cycle_data:
        evolution_champion_traces.set(cycle_data["champion_traces"])
    if "diversity" in cycle_data:
        evolution_diversity.set(cycle_data["diversity"])

    # Track last error
    _last_error = cycle_data.get("last_error", "")
    _last_update = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

    # Increment cycle counter
    evolution_cycle_total.inc()


def stop_metrics_app(server: Any) -> None:
    """Stop the HTTP server and join its thread.

    Args:
        server: The (server, thread) tuple returned by create_metrics_app().
    """
    if server is not None:
        try:
            http_server = server[0] if isinstance(server, tuple) else server
            http_server.shutdown()
        except Exception:
            pass


# ── Standalone mode ─────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    port = int(os.environ.get("EE_PROMETHEUS_PORT", "9100"))
    server = create_metrics_app(port)
    print(f"Evolution Engine metrics exporter ready on http://0.0.0.0:{port}")
    print("  /metrics  — Prometheus metrics")
    print("  /health   — Health check")
    print("  /status   — Full engine status")
    try:
        # Keep main thread alive; server runs in daemon thread
        import signal

        event = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: event.set())
        signal.signal(signal.SIGTERM, lambda *_: event.set())
        event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        stop_metrics_app(server)
        print("Prometheus exporter stopped")
