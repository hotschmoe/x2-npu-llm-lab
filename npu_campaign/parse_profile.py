"""Sum Hexagon profile-op times and the gaps inside each OPBATCH."""
import re
import sys

path = sys.argv[1]
op_re = re.compile(
    r"profile-op ([^|]+)\|.*\|usec (\d+) cycles (\d+) start (\d+) mhz ([0-9.]+)"
)
layer_re = re.compile(r"blk\.(\d+)")

batches = []
cur = None

def flush():
    global cur
    if cur is not None and (cur["ops"] or cur["batch_usec"] is not None):
        batches.append(cur)
    cur = None

with open(path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        if "profile-op " not in line:
            continue
        m = op_re.search(line)
        if not m:
            continue
        name, usec, cycles, start, mhz = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), float(m.group(5))
        if name == "OPBATCH":
            flush()
            cur = {"batch_usec": usec, "batch_cycles": cycles, "mhz": mhz, "ops": []}
            continue
        if cur is None:
            cur = {"batch_usec": None, "batch_cycles": None, "mhz": mhz, "ops": []}
        lm = layer_re.search(line)
        cur["ops"].append({
            "name": name,
            "usec": usec,
            "cycles": cycles,
            "start": start,
            "mhz": mhz,
            "layer": int(lm.group(1)) if lm else None,
        })
flush()

def gap_us(ops):
    gaps = []
    for a, b in zip(ops, ops[1:]):
        mhz = a["mhz"] or b["mhz"] or 1.0
        # start is a cycle stamp. The next op should begin at start+cycles if the DSP never waited.
        idle = b["start"] - (a["start"] + a["cycles"])
        gaps.append(idle / mhz)
    return gaps

print(f"batches {len(batches)}")
grand_ops = 0
grand_usec = 0
grand_gap = 0.0
grand_batch = 0
for i, b in enumerate(batches):
    ops = b["ops"]
    if not ops and b["batch_usec"] is None:
        continue
    su = sum(o["usec"] for o in ops)
    gaps = gap_us(ops) if len(ops) > 1 else []
    pos = sum(g for g in gaps if g > 0)
    neg = sum(g for g in gaps if g < 0)
    layers = [o["layer"] for o in ops if o["layer"] is not None]
    span = f"blk {min(layers)}-{max(layers)}" if layers else "no-blk"
    bu = b["batch_usec"]
    inside = (bu - su) if bu is not None else None
    grand_ops += len(ops)
    grand_usec += su
    grand_gap += pos
    if bu:
        grand_batch += bu
    names = {}
    for o in ops:
        names[o["name"]] = names.get(o["name"], 0) + o["usec"]
    top = sorted(names.items(), key=lambda kv: -kv[1])[:8]
    top_s = ", ".join(f"{n} {u}" for n, u in top)
    print(
        f"batch {i:3d} ops {len(ops):4d} {span:16s} "
        f"sum_usec {su:8d} batch_usec {str(bu):8s} "
        f"batch_minus_sum {str(inside):8s} "
        f"pos_gap_us {pos:8.0f} neg_gap_us {neg:8.0f} | {top_s}"
    )

print(
    f"TOTAL ops {grand_ops} sum_usec {grand_usec} ({grand_usec/1000:.1f} ms) "
    f"pos_gap_us {grand_gap:.0f} ({grand_gap/1000:.1f} ms) "
    f"opbatch_usec {grand_batch} ({grand_batch/1000:.1f} ms)"
)
