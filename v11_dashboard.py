#!/usr/bin/env python3
"""v11 Dashboard — Live monitor for Evolution Engine runs.

Usage:
  python3 v11_dashboard.py                  # Live monitoring (auto-detect latest log)
  python3 v11_dashboard.py --log <path>     # Monitor a specific log file
  python3 v11_dashboard.py --oneshot        # Print one-shot summary
"""
import json, os, re, sys, time, math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


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
    """Live dashboard — smooth 1s refresh with ETA and timing."""
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
            print(f"Evolution Engine v11 {spin}")
            print(f"Run: {fmt_time(run_elapsed)}  |  {gpu}")
            print(f"Log: {os.path.basename(log_path)[:45]}")
            print()

            mc = data.get("last_mc", {})
            cycle = data.get("current_cycle", "?")
            gen = data.get("current_gen", "?")
            level = data.get("current_level", "?")
            fitness = data.get("current_fitness", "?")
            if fitness != "?":
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
            print(f"Cycles: {completed}/48 ({completed/48*100:.0f}%)")
            print(f"Avg:    {avg_str} ({per_hr}/hr)")
            print(f"ETA:    {eta_str}")
            print(f"Caps:   {caps}")
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

            # Recent cycles
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
    log_path = None
    for i, a in enumerate(sys.argv[1:]):
        if a == "--log" and i + 2 < len(sys.argv):
            log_path = sys.argv[i + 2]
    if log_path is None:
        log_path = find_latest_log()
    if log_path is None:
        print("No evolution engine log found. Use --log <path>")
        sys.exit(1)
    _draw_live(log_path)


if __name__ == "__main__":
    main()
