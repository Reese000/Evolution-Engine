"""Compare benchmark JSONs: schema, reasons, candidate counts, judge compatibility."""
import json, re, sys, pathlib

REPO = pathlib.Path(r"C:/Users/reese/Projects/Universal Shopping Search")
base = json.loads((REPO / "tests/benchmarks/cnc_relevance_results_baseline.json").read_text(encoding="utf-8"))
aft = json.loads((REPO / "tests/benchmarks/cnc_relevance_results_after.json").read_text(encoding="utf-8"))

print("top-level keys equal:", sorted(base) == sorted(aft), sorted(base))
for name, rep in (("baseline", base), ("after", aft)):
    print(f"\n== {name} ==")
    for q in rep["queries"]:
        r = q["result"]
        print(f"  {q['query']!r:36} ret={r['returned']:3d} tot={r['total']:5d} "
              f"nDCG={r['nDCG_at_10']:.4f} P={r['P_at_10']:.4f} cand={r['candidate_count']:4d} "
              f"backend={r['backend_used']} err={r['error']}")
        print(f"      query-entry keys: {sorted(q)}")
        print(f"      result keys:      {sorted(r)}")
        print(f"      note: {q.get('note','<MISSING>')[:90]}")

# Collect reason strings per query from both
print("\n== reasons present in baseline top10 vs after top10 ==")
for qb, qa in zip(base["queries"], aft["queries"]):
    rb = sorted({row["reason"] for row in qb["result"]["top10"]})
    ra = sorted({row["reason"] for row in qa["result"]["top10"]})
    print(f"  {qb['query']!r}")
    print(f"    base: {rb}")
    print(f"    aft : {ra}")

# Now: re-run CURRENT judges over every baseline top10 row (fields recorded:
# id, grade, reason, title, price, source_site) and compare grade+reason.
sys.path.insert(0, str(REPO))
import importlib
bm = importlib.import_module("tests.benchmarks.cnc_relevance_benchmark")

judge_by_query = {bq.query: bq.judge for bq in bm.QUERY_SET}
print("\n== current judge re-run over BASELINE recorded top10 rows ==")
mismatch = 0
total = 0
for q in base["queries"]:
    j = judge_by_query.get(q["query"])
    if j is None:
        print(f"  !! no judge for {q['query']!r} in current harness")
        continue
    for row in q["result"]["top10"]:
        listing = {"id": row["id"], "title": row["title"], "price": row["price"],
                   "source_site": row["source_site"]}  # specs/desc/status not recorded
        g, reason = j(listing)
        total += 1
        if g != row["grade"] or reason != row["reason"]:
            mismatch += 1
            print(f"  MISMATCH q={q['query']!r} id={row['id']}")
            print(f"    recorded: grade={row['grade']} reason={row['reason']!r} title={row['title']!r}")
            print(f"    current : grade={g} reason={reason!r}")
print(f"\ntotal rows checked={total}, mismatches={mismatch}")
