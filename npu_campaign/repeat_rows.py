# Per-row distinct-hash report over N logits traces of the SAME build: repeat_rows.py f1 f2 ... fN
import sys, hashlib, struct
from collections import defaultdict
def rows(path):
    out = []
    with open(path, 'rb') as f:
        while h := f.read(20):
            magic, ctx, row, cnt = struct.unpack('<IQiI', h)
            out.append((row, hashlib.sha256(f.read(cnt * 8)).hexdigest()[:10]))
    return out
runs = [rows(p) for p in sys.argv[1:]]
nrows = min(len(r) for r in runs)
tot_bad = 0
for r in range(nrows):
    hs = [run[r][1] for run in runs]
    d = len(set(hs))
    if d > 1: tot_bad += 1
    print(f"row {r}: distinct={d} {' '.join(hs)}")
print(f"RESULT runs={len(runs)} rows={nrows} rows_with_variation={tot_bad} run_hashes_distinct={len(set(tuple(x[1] for x in run) for run in runs))}")
