"""Break the last Hexagon OPBATCH into per-op and per-weight latency."""
import re
import sys
from collections import defaultdict

path = sys.argv[1]
line_re = re.compile(
    r"profile-op ([^|]+)\|([^|]*)\|([^|]*)\|([^|]*)\|[^|]*\|[^|]*\|usec (\d+) cycles (\d+) start (\d+)"
)

batches = []
cur = []
for line in open(path, "r", encoding="utf-8", errors="replace"):
    if "profile-op " not in line:
        continue
    m = line_re.search(line)
    if not m:
        continue
    name = m.group(1)
    if name == "OPBATCH":
        if cur:
            batches.append(cur)
        cur = []
        continue
    weight = m.group(2).split(" x ")[0].strip()
    dims = m.group(3)
    types = m.group(4)
    cur.append({
        "name": name,
        "weight": weight,
        "dims": dims,
        "types": types,
        "usec": int(m.group(5)),
        "cycles": int(m.group(6)),
        "start": int(m.group(7)),
    })
if cur:
    batches.append(cur)

ops = batches[-1]
print(f"batches {len(batches)} last_ops {len(ops)} sum_usec {sum(o['usec'] for o in ops)}")

by_name = defaultdict(lambda: [0, 0])
for o in ops:
    by_name[o["name"]][0] += 1
    by_name[o["name"]][1] += o["usec"]
print("\n== by opcode ==")
for name, (n, u) in sorted(by_name.items(), key=lambda kv: -kv[1][1]):
    print(f"{u:7d} us  n={n:4d}  avg={u/n:7.1f}  {name}")

buckets = [(0, 8), (8, 20), (20, 50), (50, 200), (200, 1000), (1000, 10**9)]
print("\n== usec buckets ==")
for lo, hi in buckets:
    sel = [o for o in ops if lo <= o["usec"] < hi]
    label = f"{lo}-{hi-1}" if hi < 10**8 else f"{lo}+"
    print(f"{label:10} n={len(sel):4d}  sum={sum(o['usec'] for o in sel):7d} us")

by_w = defaultdict(lambda: [0, 0, "", ""])
for o in ops:
    key = (o["name"], o["weight"])
    by_w[key][0] += 1
    by_w[key][1] += o["usec"]
    by_w[key][2] = o["dims"]
    by_w[key][3] = o["types"]
print("\n== by opcode + first tensor, top 40 ==")
rows = sorted(by_w.items(), key=lambda kv: -kv[1][1])
for (name, weight), (n, u, dims, types) in rows[:40]:
    print(f"{u:7d} us  n={n:3d}  avg={u/n:7.1f}  {name:22} {weight:40} {dims}  {types}")

print("\n== slowest single ops ==")
for o in sorted(ops, key=lambda o: -o["usec"])[:15]:
    print(f"{o['usec']:7d} us  {o['name']:22} {o['weight'][:48]:48} {o['dims']}")
