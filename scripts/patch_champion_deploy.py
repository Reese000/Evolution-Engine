import json
from pathlib import Path

ROOT = Path(r"C:\Users\reese\Projects\Evolution Engine")

test_path = ROOT / "tests/unit/test_champion_rollback.py"
text = test_path.read_text(encoding="utf-8")
if "test_find_best_certified_prefers_high_fitness" not in text:
    text = (
        text.rstrip()
        + """

def test_find_best_certified_prefers_high_fitness(tmp_path):
    \"\"\"Cycle 10 (fitness 264) beats cycle 11 (fitness 26) when both verify=1.0.\"\"\"
    results = tmp_path / "multi_cycle_results.jsonl"
    prog = ["0x00000013"] * 36 + ["0x00100073"]
    rows = [
        {
            "cycle": 10,
            "final_level": 30,
            "champion_verify_score": 1.0,
            "champion_words": 37,
            "final_fitness": 264.0,
            "champion_program": prog,
        },
        {
            "cycle": 11,
            "final_level": 30,
            "champion_verify_score": 1.0,
            "champion_words": 37,
            "final_fitness": 26.0,
            "champion_program": prog,
        },
    ]
    results.write_text("\\n".join(json.dumps(r) for r in rows) + "\\n", encoding="utf-8")

    best = find_best_certified_champion(str(results), 30, min_verify=1.0)
    assert best is not None
    assert best["cycle"] == 10
    assert best["final_fitness"] == 264.0
"""
    )
    test_path.write_text(text + "\n", encoding="utf-8")
    print("Added test_find_best_certified_prefers_high_fitness")
else:
    print("Test already present")

c3511 = None
for line in (ROOT / "multi_cycle_results.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    rec = json.loads(line)
    if rec.get("cycle") == 3511:
        c3511 = rec
        break
if not c3511 or not c3511.get("champion_program"):
    raise SystemExit("missing C3511 champion_program")

tick_path = ROOT / ".agents/tick_state.json"
tick = json.loads(tick_path.read_text(encoding="utf-8"))
tick["champion_rollback_program"] = c3511["champion_program"]
tick["champion_rollback_cycle"] = 3511
tick["apply_champion_rollback"] = True
tick["notes"] = (
    "C3512 verify regression; rollback to C3511 (fitness 264, verify 1.0). "
    "selection_mixin uses selection_fitness; restore uses find_best_certified_champion."
)
tick_path.write_text(json.dumps(tick, indent=2) + "\n", encoding="utf-8")
print("Updated tick_state.json (C3511 rollback)")
