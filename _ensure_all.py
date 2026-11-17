import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
lines = [
    "#!/usr/bin/env python3",
    "# Re-evaluate champion fitness for verified cycles with low final_fitness.",
    "from __future__ import annotations",
    "import argparse, json, re, shutil, sys",
    "from datetime import datetime",
    "from pathlib import Path",
    "PROJECT_ROOT = Path(__file__).resolve().parents[1]",
    "if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))",
    "def discover_jsonl():",
    "    try:",
    "        from scripts.progress_trajectory import METRICS_JSONL",
    "        p = Path(METRICS_JSONL)",
    "        if not p.is_absolute(): p = PROJECT_ROOT / p",
    "        if p.is_file(): return p",
    "    except Exception:",
    "        pass",
    "    for candidate in (PROJECT_ROOT/'cycle_metrics.jsonl', PROJECT_ROOT/'data'/'cycle_metrics.jsonl'):",
    "        if candidate.is_file(): return candidate",
    "    for path in PROJECT_ROOT.rglob('cycle_metrics.jsonl'): return path",
    "    raise FileNotFoundError('cycle_metrics.jsonl not found')",
    "def load_engine_from_checkpoint(checkpoint):",
    "    cp = Path(checkpoint)",
    "    if not cp.is_file(): cp = PROJECT_ROOT / checkpoint",
    "    if not cp.is_file(): raise FileNotFoundError(checkpoint)",
    "    import run_multi_cycle as rmc",
    "    for name in ('load_engine_from_checkpoint','restore_engine_from_checkpoint','_load_checkpoint'):",
    "        fn = getattr(rmc, name, None)",
    "        if callable(fn): return fn(str(cp))",
    "    import pickle",
    "    from evolution_engine import EvolutionEngine",
    "    with open(cp,'rb') as f: blob = pickle.load(f)",
    "    if isinstance(blob, EvolutionEngine): return blob",
    "    if isinstance(blob, dict):",
    "        if hasattr(EvolutionEngine,'from_state_dict'): return EvolutionEngine.from_state_dict(blob)",
    "        eng = EvolutionEngine()",
    "        for k,v in blob.items():",
    "            if hasattr(eng,k): setattr(eng,k,v)",
    "        return eng",
    "    raise TypeError(type(blob))",
    "def recompute_fitness(engine):",
    "    if engine.champion is None: raise ValueError('no champion')",
    "    trace = engine._evaluate(engine.champion)",
    "    fit = float(engine._compute_fitness(engine.champion, trace))",
    "    engine.champion_fitness = fit",
    "    return fit",
    "def should_backfill(row):",
    "    try:",
    "        return float(row.get('champion_verify_score') or 0) >= 1.0 and float(row.get('final_fitness') or 0) < 200",
    "    except (TypeError, ValueError): return False",
    "def main():",
    "    ap = argparse.ArgumentParser()",
    "    ap.add_argument('--dry-run', action='store_true')",
    "    ap.add_argument('--jsonl', type=Path, default=None)",
    "    args = ap.parse_args()",
    "    jsonl = args.jsonl or discover_jsonl()",
    "    rows = [json.loads(l) for l in jsonl.read_text(encoding='utf-8').splitlines() if l.strip()]",
    "    updated = []",
    "    for row in rows:",
    "        if not should_backfill(row): continue",
    "        cycle = row.get('cycle'); ckpt = row.get('checkpoint')",
    "        if not ckpt:",
    "            print(f'cycle {cycle}: skip no checkpoint'); continue",
    "        try:",
    "            eng = load_engine_from_checkpoint(str(ckpt))",
    "            new_fit = recompute_fitness(eng)",
    "        except Exception as e:",
    "            print(f'cycle {cycle}: FAILED {e}'); continue",
    "        verified = float(row.get('champion_verify_score') or 0) >= 1.0",
    "        row['final_fitness'] = new_fit",
    "        row['champion_fitness'] = new_fit",
    "        row['verified_fitness'] = new_fit if verified else float(row.get('verified_fitness') or 0)",
    "        if 'fitness_end' in row: row['fitness_end'] = new_fit",
    "        updated.append(cycle)",
    "        print(f'cycle {cycle}: updated final_fitness={new_fit:.3f}')",
    "    if args.dry_run:",
    "        print(f'dry-run would update {len(updated)}: {updated}'); return 0",
    "    if not updated:",
    "        print('no cycles updated'); return 0",
    "    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')",
    "    backup = jsonl.with_suffix(f'.jsonl.bak_{stamp}')",
    "    shutil.copy2(jsonl, backup)",
    "    with jsonl.open('w', encoding='utf-8') as f:",
    "        for row in rows: f.write(json.dumps(row, ensure_ascii=False)+'\\n')",
    "    print(f'backup: {backup}')",
    "    print(f'updated cycles ({len(updated)}): {updated}')",
    "    return 0",
    "if __name__ == '__main__': raise SystemExit(main())",
]
(ROOT / "scripts").mkdir(exist_ok=True)
(ROOT / "scripts" / "backfill_champion_fitness.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
text = (ROOT / "run_multi_cycle.py").read_text(encoding="utf-8")
report = {"backfill_written": True}
if "engine.champion_fitness = recomputed" in text:
    report["patch"] = "already_applied"
else:
    pat = re.compile(
        r"    try:\n        verified, verify_score = engine\._verify_champion\(\)\n.*?    except \(RuntimeError, TypeError, ValueError, AttributeError\):\n        pass",
        re.DOTALL,
    )
    repl = """    try:\n        if engine.champion is not None:\n            fresh_trace = engine._evaluate(engine.champion)\n            recomputed = float(\n                engine._compute_fitness(engine.champion, fresh_trace)\n            )\n            metrics[\"final_fitness\"] = recomputed\n            engine.champion_fitness = recomputed\n        verified, verify_score = engine._verify_champion()\n        max_pop = float(metrics.get(\"max_pop_verify\", 0) or 0)\n        effective_verify = max(float(verify_score), max_pop)\n        metrics[\"champion_verify_score\"] = effective_verify\n        metrics[\"champion_verified\"] = effective_verify >= 1.0 - 1e-6\n        metrics[\"verified_fitness\"] = (\n            float(metrics[\"final_fitness\"]) if metrics[\"champion_verified\"] else 0.0\n        )\n    except (RuntimeError, TypeError, ValueError, AttributeError) as e:\n        log(f\"  Cycle {cycle}: Champion verify/fitness recompute failed: {e}\")"""
    new_text, n = pat.subn(repl, text, 1)
    if n:
        (ROOT / "run_multi_cycle.py").write_text(new_text, encoding="utf-8")
        report["patch"] = "applied"
    else:
        report["patch"] = "failed"


def run(name, args):
    r = subprocess.run([sys.executable] + args, cwd=ROOT, capture_output=True, text=True)
    report[name] = {"rc": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    return r


run("backfill", ["scripts/backfill_champion_fitness.py"])
run("trajectory", ["scripts/progress_trajectory.py", "--rebuild"])
epi = run("epi", ["scripts/verify_epi_momentum_goal.py", "--json"])
if epi.stdout.strip():
    try:
        report["epi_json"] = json.loads(epi.stdout)
    except Exception as ex:
        report["epi_json_error"] = str(ex)
deploy = ROOT / "scripts" / "_deploy_run.py"
if deploy.is_file():
    d = run("deploy", [str(deploy)])
    report["daemon_restarted"] = d.returncode == 0
else:
    report["deploy"] = {"rc": -1, "stdout": "", "stderr": "missing"}
    report["daemon_restarted"] = False
(ROOT / "agent_emfile_handoff.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
