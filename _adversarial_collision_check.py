"""Pin down the size-1 collision mechanism: is delta -1 == delta -2 at BLT loci?"""

import json
import os

os.environ.setdefault("EE_USE_GPU", "0")
from milestone1.rv32i_mask import decode as D
from milestone1.rv32i_mask import encode as E

rows = open("multi_cycle_results.jsonl", encoding="utf-8").readlines()
row = json.loads(rows[-1])
champ = [int(x, 16) if isinstance(x, str) else int(x) for x in row["champion_program"]]


def word(idx, delta):
    d = D(champ[idx], validate=False)
    return int(
        E(
            name=str(d["name"]),
            rd=int(d.get("rd", 0)),
            rs1=int(d.get("rs1", 0)),
            rs2=int(d.get("rs2", 0)),
            imm=int(d.get("imm", 0)) + delta,
            validate=True,
        )
    )


for idx in (22, 23, 24, 25):
    d = D(champ[idx], validate=False)
    w_plus1 = word(idx, 1)
    w_minus1 = word(idx, -1)
    w_minus2 = word(idx, -2)
    print(
        f"locus {idx} ({d['name']}, imm={d['imm']}): "
        f"+1==champ:{w_plus1 == champ[idx]} "
        f"-1==champ:{w_minus1 == champ[idx]} "
        f"-1==-2:{w_minus1 == w_minus2} "
        f"-1 word=0x{w_minus1:08X} -2 word=0x{w_minus2:08X}"
    )
