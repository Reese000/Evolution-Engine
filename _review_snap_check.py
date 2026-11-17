"""Verify: snap gate state, after.json reproducibility, artifact mtimes."""

import json
import os
import sqlite3
import time

db = "C:/Windows/Temp/t_1dd79833/snap.db"
conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
cols = [r[1] for r in conn.execute("PRAGMA table_info(listings)")]
print("cnc_machine_ok column present:", "cnc_machine_ok" in cols)
try:
    print("gate meta:", conn.execute("SELECT key, value FROM cnc_machine_gate_meta").fetchall())
    print(
        "stamped counts (value, n):",
        conn.execute(
            "SELECT cnc_machine_ok, COUNT(*) FROM listings WHERE vertical IN ('cnc','cnc_machines') "
            "GROUP BY cnc_machine_ok"
        ).fetchall(),
    )
except Exception as e:
    print("meta err:", e)
try:
    print("total rows:", conn.execute("SELECT COUNT(*) FROM listings").fetchone())
except Exception as e:
    print(e)
conn.close()

after = json.load(
    open("C:/Users/reese/Projects/Universal Shopping Search/tests/benchmarks/cnc_relevance_results_after.json")
)
repro = json.load(open("C:/Users/reese/Projects/Evolution Engine/_review_repro.json"))
same = True
for qa, qr in zip(after["queries"], repro["queries"]):
    ka = (
        qa["result"]["nDCG_at_10"],
        qa["result"]["P_at_10"],
        qa["result"]["returned"],
        qa["result"]["total"],
        qa["result"]["candidate_count"],
        [r["id"] for r in qa["result"]["top10"]],
    )
    kr = (
        qr["result"]["nDCG_at_10"],
        qr["result"]["P_at_10"],
        qr["result"]["returned"],
        qr["result"]["total"],
        qr["result"]["candidate_count"],
        [r["id"] for r in qr["result"]["top10"]],
    )
    if ka != kr:
        same = False
        print("DIFF", qa["query"])
        print("  recorded:", ka[:5])
        print("  repro   :", kr[:5])
        print("  recorded ids:", ka[5])
        print("  repro ids   :", kr[5])
print("after.json top10 ids+metrics fully reproduced:", same)
print("aggregates equal:", after["aggregate"] == repro["aggregate"])

for p in [
    "data/gate_v4_applied.json",
    "data/gate_v4_model_patterns.json",
    "scripts/gate_v4_model_patterns.py",
    "tests/unit/test_cnc_listing_gate.py",
    "tests/unit/test_schema_filters.py",
    "tests/unit/test_travel_format.py",
    "tests/unit/test_sold_listing_filter.py",
    "tests/unit/test_sold_html_detectors.py",
    "tests/unit/test_source_display.py",
    "tests/unit/test_semantic.py",
    "tests/unit/test_semantic_extended.py",
    "tests/unit/test_cnc_parse_golden.py",
    "tests/unit/test_audit_parsing_retention.py",
    "src/search/fts5.py",
    "src/search/cache.py",
    "src/search/browse_sql.py",
    "src/search/analytics.py",
    "src/core/resilience.py",
]:
    fp = os.path.join("C:/Users/reese/Projects/Universal Shopping Search", p)
    if os.path.exists(fp):
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(fp))), p)
    else:
        print("MISSING", p)
