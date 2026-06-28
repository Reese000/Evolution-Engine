#!/usr/bin/env python3
"""v11 Dashboard — Live monitor for Evolution Engine runs.

Reads from:
  - evolution.log (legacy log parsing)
  - health.json (Phase 3.3 health monitor)
  - metrics/cycles.jsonl (Phase 3.2 MetricsCollector)
  - progress_report.json (Phase 7.2 progress reports)

Usage:
  python3 v11_dashboard.py                  # Live monitoring (auto-detect latest log)
  python3 v11_dashboard.py --log <path>     # Monitor a specific log file
  python3 v11_dashboard.py --oneshot        # Print one-shot summary
  python3 v11_dashboard.py --health         # Show health.json status
  python3 v11_dashboard.py --metrics        # Show metrics/cycles.jsonl summary
  python3 v11_dashboard.py --report         # Show progress_report.json
"""
import json, os, re, sys, time, math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

# ── Project paths ────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_HEALTH_PATH = os.path.join(_PROJECT_ROOT, "health.json")
_METRICS_PATH = os.path.join(_PROJECT_ROOT, "metrics", "cycles.jsonl")
_PROGRESS_REPORT_PATH = os.path.join(_PROJECT_ROOT, "progress_report.json")


def find_latest_log() -> Optional[str]:
    log_dir = "/root/continuous_iteration"
    patterns = ["run_mc_v11_*.log", "run_mc_v10_fixed2_*.log", "run_mc_v101_*.log",
                "run_mc_v10_*.log", "run_mc_v9_*.log"]
    best, best_mtime = None, 0
    import glob
    for pat in patterns:
        for f in glob.glob(os.path.join(log_dir, pat)):
            mtime = os.path.getmtime(f)
            if mtime > best_mtime:
                best_mtime, best = mtime, f
    return best


# ══════════════════════════════════════════════════════════════════════
# Phase 7.1: Read from health.json + metrics/cycles.jsonl
# ══════════════════════════════════════════════════════════════════════

def read_health_json(path: str = _HEALTH_PATH) -> Optional[Dict[str, Any]]:
    """Read health.json if it exists. Returns None if not found."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def read_metrics_jsonl(path: str = _METRICS_PATH, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read metrics/cycles.jsonl and return list of cycle records."""
    records: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        return records
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return records
    if last_n is not None and last_n > 0:
        return records[-last_n:]
    return records


def compute_metrics_trend(records: List[Dict[str, Any]], metric: str = "fitness",
                          window: int = 10) -> Dict[str, Any]:
    """Compute rolling trend from metrics records."""
    values = []
    for r in records:
        v = r.get(metric)
        if isinstance(v, (int, float)):
            values.append(float(v))
    if not values:
        return {"current": None, "trend": "unknown", "window_size": 0}
    recent = values[-window:]
    avg = sum(recent) / len(recent)
    if len(recent) >= 4:
        mid = len(recent) // 2
        first_half = sum(recent[:mid]) / mid
        second_half = sum(recent[mid:]) / (len(recent) - mid)
        delta = second_half - first_half
        threshold = max(abs(first_half) * 0.01, 0.001)
        if delta > threshold:
            trend = "improving"
        elif delta < -threshold:
            trend = "declining"
        else:
            trend = "stable"
    elif len(recent) >= 2:
        delta = recent[-1] - recent[0]
        trend = "improving" if delta > 0 else ("declining" if delta < 0 else "stable")
    else:
        trend = "stable"
    return {
        "current": values[-1],
        "avg": round(avg, 6),
        "min": min(recent),
        "max": max(recent),
        "trend": trend,
        "window_size": len(recent),
    }


def read_progress_report(path: str = _PROGRESS_REPORT_PATH) -> Optional[Dict[str, Any]]:
    """Read progress_report.json if it exists."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def show_health_status() -> None:
    """Print health.json status to console."""
    health = read_health_json()
    if health is None:
        print("No health.json found. Runner has not been started.")
        return
    print("=" * 60)
    print("Evolution Engine — Health Status")
    print("=" * 60)
    for k, v in health.items():
        if k == "uptime_seconds":
            print(f"  {k:25s}: {fmt_time(v)}")
        else:
            print(f"  {k:25s}: {v}")
    print("=" * 60)


def show_metrics_summary() -> None:
    """Print metrics/cycles.jsonl summary to console."""
    records = read_metrics_jsonl()
    if not records:
        print("No metrics data found (metrics/cycles.jsonl).")
        return
    print("=" * 60)
    print(f"Evolution Engine — Metrics Summary ({len(records)} cycles)")
    print("=" * 60)

    fitness_trend = compute_metrics_trend(records, "fitness")
    level_trend = compute_metrics_trend(records, "level")
    time_trend = compute_metrics_trend(records, "cycle_time_seconds")

    print(f"  Fitness:  {fitness_trend.get('current', '?'):.3f} "
          f"(avg={fitness_trend.get('avg', '?')}, trend={fitness_trend.get('trend', '?')})")
    print(f"  Level:    {level_trend.get('current', '?')} "
          f"(trend={level_trend.get('trend', '?')})")
    print(f"  Cycle time: {time_trend.get('current', '?')}s "
          f"(avg={time_trend.get('avg', '?')}s)")
    print()

    # Strategy distribution
    strategies = defaultdict(int)
    for r in records:
        strat = r.get("strategy")
        if strat:
            strategies[strat] += 1
    if strategies:
        print("  Strategy distribution:")
        total = sum(strategies.values()) or 1
        for s, c in sorted(strategies.items(), key=lambda x: -x[1]):
            pct = c / total * 100
            print(f"    {s:<15} {c:>5} ({pct:5.1f}%)")
    print()

    # Recent cycles
    recent = records[-5:]
    print("  Recent cycles:")
    for r in recent:
        f = r.get("fitness", 0)
        fl = " *" if isinstance(f, (int, float)) and f > 0 else ""
        print(f"    C{r.get('cycle', '?'):>3}: F={f:+.3f}{fl} "
              f"L={r.get('level', '?'):>2} t={r.get('cycle_time_seconds', '?')}s")
    print("=" * 60)


def show_progress_report() -> None:
    """Print progress_report.json to console."""
    report = read_progress_report()
    if report is None:
        print("No progress_report.json found. Runner has not completed a cycle.")
        return
    print("=" * 60)
    print("Evolution Engine — Progress Report")
    print(f"  Generated: {report.get('generated_at', '?')}")
    print("=" * 60)
    summary = report.get("summary", {})
    print(f"  Best fitness:    {summary.get('best_fitness', 0):.3f}")
    print(f"  Best level:      {summary.get('best_level', 0)}")
    print(f"  Cycles completed: {summary.get('cycles_completed', 0)}")
    print(f"  Total errors:    {summary.get('total_errors', 0)}")
    print(f"  Uptime:          {fmt_time(summary.get('uptime_seconds', 0))}")
    print()
    trends = report.get("trends", {})
    ft = trends.get("fitness", {})
    lt = trends.get("level", {})
    print(f"  Fitness trend: {ft.get('trend', '?')} (current={ft.get('current', '?')}, avg={ft.get('avg', '?')})")
    print(f"  Level trend:   {lt.get('trend', '?')} (current={lt.get('current', '?')})")
    print()
    resources = report.get("resource_usage", {})
    print(f"  Checkpoint disk: {resources.get('checkpoint_disk_mb', 0):.1f} MB")
    print(f"  Population size: {resources.get('population_size', '?')}")
    print("=" * 60)


def parse_log(log_path: str) -> Dict[str, Any]:
    if not os.path.isfile(log_path):
        return {"error": "log not found"}
    with open(log_path) as f:
        try:
            f.seek(0, 2)
            size = f.tell()
            chunk = min(500 * 1024, size)
            f.seek(max(0, size - chunk))
            content = f.read()
        except (OSError, PermissionError):
            content = ""
    lines = content.split('\n')

    result = {"log_path": log_path, "total_lines": os.path.getsize(log_path),
              "cycles_completed": 0, "mc_entries": [],
              "strategy_counts": defaultdict(int),
              "current_cycle": None, "current_gen": None,
              "current_level": None, "current_fitness": None,
              "last_mc": None, "monoculture_caps": 0,
              "cycle_completion_times": []}

    for line in lines:
        m = re.search(r'\[MetaController\] Cycle (\d+): F=([-\d.]+) L=(\d+) val=([-\d.]+) div=([-\d.]+)', line)
        if m:
            entry = {"cycle": int(m.group(1)), "fitness": float(m.group(2)),
                     "level": int(m.group(3)),
                     "val_loss": float(m.group(4)) if m.group(4) != "None" else None,
                     "diversity": float(m.group(5)), "_timestamp": time.time()}
            result["mc_entries"].append(entry)
            result["last_mc"] = entry

        m = re.search(r'CYCLE (\d+)/(\d+)', line)
        if m:
            result["current_cycle"] = int(m.group(1))
        m = re.search(r'Gen\s+(\d+)\s+\|', line)
        if m:
            result["current_gen"] = int(m.group(1))
        m = re.search(r'Level=(\d+)', line)
        if m:
            result["current_level"] = int(m.group(1))
        m = re.search(r'Fitness=\s*([-\d.]+)', line)
        if m:
            result["current_fitness"] = float(m.group(1))
        m = re.search(r'Cycle (\d+) complete.*?([\d.]+)s', line)
        if m:
            result["cycles_completed"] = int(m.group(1))
            result["last_cycle_time"] = float(m.group(2))
            result["cycle_completion_times"].append(("cycle", float(m.group(2)), time.time()))
        m = re.search(r"'(\w+)' won", line)
        if m:
            result["strategy_counts"][m.group(1)] += 1
        if "Monoculture Cap" in line:
            result["monoculture_caps"] += 1
    return result


def format_val_loss_trajectory(mc_entries: List[Dict], width: int = 25) -> str:
    valid = [e for e in mc_entries if e["val_loss"] is not None]
    if len(valid) < 2:
        return "(insufficient data)"
    vals = [e["val_loss"] for e in valid]
    vmin, vmax = min(vals), max(vals)
    if vmax - vmin < 0.001:
        return f"flat at {vmin:.4f}"
    bar = ""
    for v in vals[-width:]:
        p = int((v - vmin) / (vmax - vmin) * 6)
        bar += "▁▂▃▄▅▆▇█"[min(p, 7)]
    return f"{vmin:.4f} {bar} {vmax:.4f}"


def get_gpu_info() -> str:
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split(', ')
            if len(parts) >= 3:
                return f"GPU: {parts[0]}% | {parts[1]}/{parts[2]}MB | {parts[3]}C"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "GPU: N/A"


def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m:02d}m{s:02d}s"


def _draw_live(log_path: str) -> None:
    """Live dashboard — smooth 1s refresh with ETA, timing, health.json + metrics."""
    last_parse, data = 0, {"mc_entries": [], "strategy_counts": {}}
    start_time = time.time()

    # Estimate run start from first log line timestamp
    try:
        with open(log_path) as f:
            first = f.readline().strip()
            m = re.match(r'\[(\d{2}):(\d{2}):(\d{2})\]', first)
            if m:
                h, mm, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
                run_start = datetime.now().replace(hour=h, minute=mm, second=s, microsecond=0)
                start_time = run_start.timestamp()
    except (OSError, IndexError):
        pass

    print("\033[?25l", end="", flush=True)
    try:
        while True:
            now = time.time()
            if now - last_parse >= 1.0:
                data = parse_log(log_path)
                # Assign approximate timestamps to MC entries
                entries = data.get("mc_entries", [])
                base_ts = start_time
                for i, e in enumerate(entries):
                    if "_timestamp" not in e:
                        e["_timestamp"] = base_ts + i * 600  # ~10min per cycle
                last_parse = now

            # Read health.json + metrics (Phase 7.1 integration)
            health = read_health_json()
            metrics_records = read_metrics_jsonl(last_n=20)
            progress = read_progress_report()

            # Timing calculations
            run_elapsed = now - start_time
            mc_entries = data.get("mc_entries", [])
            completed = data.get("cycles_completed", 0)

            # Compute avg cycle time and ETA from MC entry timestamps
            avg_str, eta_str, per_hr, since_mc = "?", "?", "?", "?"
            if len(mc_entries) >= 2:
                gaps = []
                for i in range(1, len(mc_entries)):
                    t1 = mc_entries[i-1].get("_timestamp", 0)
                    t2 = mc_entries[i].get("_timestamp", 0)
                    if t1 > 0 and t2 > t1:
                        gaps.append(t2 - t1)
                if gaps:
                    avg_gap = sum(gaps) / len(gaps)
                    avg_str = f"{avg_gap/60:.1f}m" if avg_gap >= 120 else f"{avg_gap:.0f}s"
                    per_hr = f"{3600/avg_gap:.1f}" if avg_gap > 0 else "?"
                    remaining = max(0, 48 - completed) * avg_gap
                    eta_str = fmt_time(remaining)

            if mc_entries:
                lt = mc_entries[-1].get("_timestamp")
                if lt:
                    es = now - lt
                    since_mc = f"{es:.0f}s" if es < 120 else f"{es/60:.0f}m"

            # 1-second tick spinner
            tick = int(now) % 60
            spin = "|/-\\"[tick % 4]

            # ── Render ──
            print("\033[H", end="")

            # Header
            gpu = get_gpu_info()
            status_str = health.get("status", "?") if health else "no health.json"
            uptime_str = fmt_time(health.get("uptime_seconds", run_elapsed)) if health else fmt_time(run_elapsed)
            print(f"Evolution Engine v11 {spin}  [{status_str}]")
            print(f"Run: {uptime_str}  |  {gpu}")
            print(f"Log: {os.path.basename(log_path)[:45]}")
            print()

            mc = data.get("last_mc", {})
            cycle = data.get("current_cycle", "?")
            gen = data.get("current_gen", "?")
            # Prefer health.json for level/fitness if available
            level = health.get("current_level", data.get("current_level", "?")) if health else data.get("current_level", "?")
            fitness = data.get("current_fitness", health.get("current_fitness")) if health else data.get("current_fitness", "?")
            if fitness != "?" and fitness is not None:
                fs = f"{fitness:+.3f}" + (" *" if fitness > 0 else "")
            else:
                fs = "?"
            val_s = "?"
            div_s = "?"
            if mc:
                v = mc.get("val_loss")
                val_s = f"{v:.4f}" if v is not None else "None"
                div_s = f"{mc.get('diversity',0):.3f}"

            print(f"C{cycle} G{gen} L{level} F={fs}")
            print(f"val_loss={val_s} div={div_s} ({since_mc})")
            print()

            # Timing
            caps = data.get("monoculture_caps", 0)
            total_cycles = health.get("cycles_completed", completed) if health else completed
            target = health.get("target_fitness", 48)
            print(f"Cycles: {total_cycles}/{target} ({total_cycles/target*100:.0f}%)")
            print(f"Avg:    {avg_str} ({per_hr}/hr)")
            print(f"ETA:    {eta_str}")
            print(f"Errors: {health.get('errors', 0) if health else 0}  Caps: {caps}")
            print()

            # Val loss trajectory
            traj = format_val_loss_trajectory(mc_entries, 25)
            print(f"val_loss: {traj}")
            print()

            # Strategy distribution
            counts = data.get("strategy_counts", {})
            total = sum(counts.values()) or 1
            for s in ["GROW", "PERTURBATION_CAPPED", "HYBRID", "RANDOM"]:
                c = counts.get(s, 0)
                pct = c / total * 100
                bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
                print(f"{s:<12} {c:>5} ({pct:5.1f}%) |{bar}|")

            print()

            # Recent cycles from metrics (Phase 7.1)
            if metrics_records:
                recent_m = metrics_records[-5:]
                print("Recent (metrics):")
                for r in recent_m:
                    f = r.get("fitness", 0)
                    fl = " *" if isinstance(f, (int, float)) and f > 0 else ""
                    print(f"  C{r.get('cycle', '?'):>3}: F={f:+.3f}{fl} "
                          f"L={r.get('level', '?'):>2} t={r.get('cycle_time_seconds', '?')}s")
            else:
                # Fallback to log-parsed MC entries
                recent = mc_entries[-5:] if len(mc_entries) >= 5 else mc_entries
                for e in recent:
                    f = e["fitness"]
                    fl = " *" if f > 0 else ""
                    v = e["val_loss"]
                    vs = f"{v:.4f}" if v is not None else "None"
                    print(f"C{e['cycle']:>2}: F={f:+.3f}{fl} L={e['level']:>2} val={vs}")

            print()
            print("-" * 55)
            print(f"ETA: {eta_str}  Caps: {caps}  Ctrl+C to quit")
            print("\033[J", end="")

            try:
                time.sleep(1.0)
            except KeyboardInterrupt:
                break
    finally:
        print("\033[?25h", end="", flush=True)
        print("\nDashboard closed.")


def oneshot(log_path: Optional[str] = None) -> None:
    if log_path is None:
        log_path = find_latest_log()
    if log_path is None:
        print("No evolution engine log found.")
        return
    data = parse_log(log_path)
    gpu = get_gpu_info()
    print(f"=== v11 Dashboard — {os.path.basename(log_path)} ===")
    print(f"{gpu}")
    print()
    mc = data.get("last_mc", {})
    if mc:
        v = mc.get("val_loss")
        vs = f"{v:.4f}" if v is not None else "None"
        print(f"MC: C{mc['cycle']} F={mc['fitness']:+.3f} L={mc['level']} val={vs} div={mc['diversity']:.3f}")
    print(f"Cycles: {data.get('cycles_completed', 0)}/48 completed")
    print(f"Caps: {data.get('monoculture_caps', 0)}")
    print()
    counts = data.get("strategy_counts", {})
    total = sum(counts.values()) or 1
    for s in ["GROW", "PERTURBATION_CAPPED", "HYBRID", "RANDOM"]:
        c = counts.get(s, 0)
        pct = c / total * 100
        print(f"  {s:<12}: {c:>5} ({pct:5.1f}%)")
    print()
    entries = data.get("mc_entries", [])
    print(f"val_loss: {format_val_loss_trajectory(entries)}")
    recent = entries[-5:] if len(entries) >= 5 else entries
    print("\nRecent cycles:")
    for e in recent:
        f = e["fitness"]
        fl = " *" if f > 0 else ""
        v = e["val_loss"]
        vs = f"{v:.4f}" if v is not None else "None"
        print(f"  C{e['cycle']:>2}: F={f:+.3f}{fl} L={e['level']:>2} val={vs}")


def main() -> None:
    if "--oneshot" in sys.argv:
        return oneshot()
    if "--health" in sys.argv:
        return show_health_status()
    if "--metrics" in sys.argv:
        return show_metrics_summary()
    if "--report" in sys.argv:
        return show_progress_report()
    log_path = None
    for i, a in enumerate(sys.argv[1:]):
        if a == "--log" and i + 2 < len(sys.argv):
            log_path = sys.argv[i + 2]
    if log_path is None:
        log_path = find_latest_log()
    if log_path is None:
        # No log file found — try to show health/metrics summary instead
        health = read_health_json()
        metrics_records = read_metrics_jsonl()
        if health or metrics_records:
            print("No log file found. Showing available data:\n")
            if health:
                show_health_status()
                print()
            if metrics_records:
                show_metrics_summary()
            return
        print("No evolution engine log found. Use --log <path>")
        sys.exit(1)
    _draw_live(log_path)


if __name__ == "__main__":
    main()
