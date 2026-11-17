from pathlib import Path

p = Path(__file__).resolve().parents[1] / "run_multi_cycle.py"
t = p.read_text(encoding="utf-8")
needle = '        verified, verify_score = engine._verify_champion()\n        champion_verify = float(verify_score)\n        metrics["champion_verify_score"] = champion_verify'
insert = (
    '        if hasattr(engine, "_maybe_restore_best_certified_champion"):\n'
    "            engine._maybe_restore_best_certified_champion()\n"
    "        verified, verify_score = engine._verify_champion()\n"
    "        champion_verify = float(verify_score)\n"
    '        metrics["champion_verify_score"] = champion_verify'
)
if needle in t:
    p.write_text(t.replace(needle, insert, 1), encoding="utf-8")
    print("patched")
elif "_maybe_restore_best_certified_champion" in t:
    print("already patched")
else:
    raise SystemExit("needle not found")
