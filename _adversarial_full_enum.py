"""Exhaustive ENCODING-only enumeration of the champion's imm space at loci
22-25 x all 14 wide deltas, subsets size 1..4. Proves whether the probe's
32,954 'distinct tested' equals every distinct reachable program (minus the
champion itself). No fitness scoring involved."""

import itertools
import os
import time

os.environ.setdefault("EE_USE_GPU", "0")
from milestone1.rv32i_mask import decode as rv32i_decode
from milestone1.rv32i_mask import encode as rv32i_encode

WIDE = (-64, -32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32, 64)
t0 = time.time()

rows = open("multi_cycle_results.jsonl", encoding="utf-8").readlines()
import json

row = json.loads(rows[-1])
champ = [int(x, 16) if isinstance(x, str) else int(x) for x in row["champion_program"]]
champ_t = tuple(champ)
EFF = (22, 23, 24, 25)


def apply_delta(prog, idx, delta):
    d = rv32i_decode(prog[idx], validate=False)
    w = rv32i_encode(
        name=str(d["name"]),
        rd=int(d.get("rd", 0)),
        rs1=int(d.get("rs1", 0)),
        rs2=int(d.get("rs2", 0)),
        imm=int(d.get("imm", 0)) + int(delta),
        validate=True,
    )
    v = list(prog)
    v[idx] = int(w)
    return v


all_progs = set()
per_size = {}
n_none = 0
for sz in (1, 2, 3, 4):
    s = set()
    for combo in itertools.combinations(EFF, sz):
        for deltas in itertools.product(WIDE, repeat=sz):
            v = list(champ)
            try:
                for idx, dl in zip(combo, deltas):
                    v = apply_delta(v, idx, dl)
            except (ValueError, TypeError, KeyError, AssertionError):
                n_none += 1
                continue
            s.add(tuple(v))
    per_size[sz] = len(s)
    all_progs |= s

non_champ = len(all_progs - {champ_t})
print(f"per-size distinct (incl possible champ): {per_size}")
print(f"encode failures (None): {n_none}")
print(f"union all sizes: {len(all_progs)}")
print(f"union EXCLUDING champ: {non_champ}")
print(f"probe claimed distinct tested: 32954  match: {non_champ == 32954}")
print(f"champ reachable in space (no-op deltas): {champ_t in all_progs}")
print(f"done in {time.time() - t0:.1f}s")
