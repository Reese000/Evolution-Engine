#!/usr/bin/env python3
"""Integration test for metrics_export HTTP endpoints.

metrics_export exposes /metrics (Prometheus), /health, and /status for
monitoring long-running autonomous evolution. It was untested. Uses an
ephemeral port (0) so it can't collide with anything in CI.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import metrics_export as mx


def _get(url):
    return urllib.request.urlopen(url, timeout=5)


def test_metrics_export_endpoints():
    srv = mx.create_metrics_app(port=0)
    server = srv[0]
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}"
    try:
        mx.update_metrics(
            {
                "final_fitness": 412.2,
                "final_level": 10,
                "elapsed_seconds": 1.5,
                "population_size": 50,
                "champion_traces": 87,
                "diversity": 0.63,
            }
        )
        time.sleep(0.2)

        # /metrics — Prometheus text exposition
        r = _get(base + "/metrics")
        assert r.status == 200
        assert "text/plain" in r.headers.get("Content-Type", "")
        body = r.read().decode()
        for name in ("evolution_fitness", "evolution_level", "evolution_cycle_total"):
            assert name in body, f"missing metric {name}"

        # /health — JSON
        r = _get(base + "/health")
        assert r.status == 200
        health = json.loads(r.read())
        assert {"status", "fitness", "cycle", "uptime"}.issubset(health)

        # /status — richer JSON
        r = _get(base + "/status")
        assert r.status == 200
        status = json.loads(r.read())
        assert {"fitness", "diversity", "champion_traces"}.issubset(status)

        # unknown path -> 404
        try:
            _get(base + "/does-not-exist")
            assert False, "expected 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        mx.stop_metrics_app(srv)
