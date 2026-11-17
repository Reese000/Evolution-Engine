#!/usr/bin/env python3
"""Stream-rewrite multi_cycle_results.jsonl: strip inline archive_state blobs.

Legacy lines embed multi-MB archive_state; new cycles use archive_state_path sidecars.
Writes to multi_cycle_results.slim.jsonl then atomically replaces original when --commit.

Usage:
  python scripts/slim_multi_cycle_jsonl.py          # dry-run stats
  python scripts/slim_multi_cycle_jsonl.py --commit # rewrite file
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RESULTS = os.path.join(PROJECT_ROOT, "multi_cycle_results.jsonl")
SLIM = os.path.join(PROJECT_ROOT, "multi_cycle_results.slim.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", action="store_true", help="Replace original after successful write")
    parser.add_argument("--input", default=RESULTS)
    parser.add_argument("--output", default=SLIM)
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Missing {args.input}")
        return 1

    from evolution.archive_persistence import slim_result_for_jsonl

    in_size = os.path.getsize(args.input)
    lines_in = 0
    stripped = 0
    out_bytes = 0
    t0 = time.time()

    with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            lines_in += 1
            row = json.loads(line)
            if "archive_state" in row:
                stripped += 1
            slim = slim_result_for_jsonl(row)
            out_line = json.dumps(slim) + "\n"
            fout.write(out_line)
            out_bytes += len(out_line.encode("utf-8"))

    elapsed = time.time() - t0
    pct = (1 - out_bytes / in_size) * 100 if in_size else 0
    print(f"lines={lines_in} stripped_inline_archive={stripped}")
    print(f"in={in_size / 1e6:.1f}MB out={out_bytes / 1e6:.1f}MB saved={pct:.1f}% in {elapsed:.1f}s")

    if args.commit:
        backup = args.input + ".pre_slim.bak"
        if not os.path.isfile(backup):
            os.replace(args.input, backup)
            print(f"backup: {backup}")
        else:
            os.remove(args.input)
        os.replace(args.output, args.input)
        print(f"committed: {args.input}")
    else:
        print("dry-run — pass --commit to replace original")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
