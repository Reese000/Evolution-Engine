import json
for r in open("multi_cycle_results.jsonl").readlines()[-18:]:
    d = json.loads(r)
    print(
        f"C{d['cycle']} {d.get('elapsed_seconds',0):.1f}s "
        f"div={d.get('diversity',0):.2f} "
        f"prom={d.get('promotion_raw_fitness',0):.4f}"
    )
