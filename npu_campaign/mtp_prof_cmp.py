"""Per-batch-class Hexagon op profile summary (GGML_HEXAGON_PROFILE=1 logs).

usage: mtp_prof_cmp.py LOG[:NOPS] [LOG[:NOPS] ...] [--top N] [--by name|weight]
Groups OPBATCH records by n-ops. Without :NOPS prints a class table; with :NOPS prints the
mean per-op-name (or per weight/op) usec of that class, side by side across logs.
"""
import re
import sys
from collections import defaultdict

hdr_re = re.compile(r"profile-op OPBATCH\|[^|]*\|n-ops (\d+)\|.*usec (\d+)")
op_re = re.compile(r"profile-op ([^|]+)\|([^|]*)\|([^|]*)\|([^|]*)\|[^|]*\|([^|]*)\|usec (\d+)")


def load(path):
    batches = []
    cur = None
    for line in open(path, encoding="utf-8", errors="replace"):
        if "profile-op" not in line:
            continue
        m = hdr_re.search(line)
        if m:
            cur = {"nops": int(m.group(1)), "usec": int(m.group(2)), "ops": []}
            batches.append(cur)
            continue
        m = op_re.search(line)
        if m and cur is not None:
            cur["ops"].append((m.group(1), m.group(2), m.group(3), m.group(5), int(m.group(6))))
    return batches


def key_of(op, by):
    name, names, dims, mode, _ = op
    if by == "name":
        return name
    w = names.split(" x ")[0].split(" -> ")[0]
    w = re.sub(r"blk\.\d+\.", "blk.N.", w)
    w = re.sub(r"-\d+$", "-N", w)
    return f"{name} {w} {mode.split(' ')[0]}"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    top = 30
    by = "name"
    if "--top" in sys.argv:
        top = int(sys.argv[sys.argv.index("--top") + 1])
        args.remove(str(top))
    if "--by" in sys.argv:
        by = sys.argv[sys.argv.index("--by") + 1]
        args.remove(by)
    cols = []
    for a in args:
        path, _, nops = a.partition(":")
        batches = load(path)
        if not nops:
            cls = defaultdict(list)
            for b in batches:
                cls[b["nops"]].append(b["usec"])
            print(path)
            for k, v in sorted(cls.items()):
                vs = sorted(v)
                print(f"  n-ops {k:5d} count {len(v):3d} mean {sum(v)/len(v)/1e3:8.2f} ms  median {vs[len(vs)//2]/1e3:8.2f} ms")
            continue
        sel = [b for b in batches if b["nops"] == int(nops)]
        sel = sel[1:] if len(sel) > 2 else sel  # drop first (warm) instance when possible
        acc = defaultdict(lambda: [0, 0])
        for b in sel:
            for op in b["ops"]:
                k = key_of(op, by)
                acc[k][0] += op[4]
                acc[k][1] += 1
        n = max(1, len(sel))
        cols.append((f"{path.split('/')[-1].split(chr(92))[-1][:28]}:{nops} (n={len(sel)}, {sum(b['usec'] for b in sel)/n/1e3:.2f}ms)",
                     {k: (v[0] / n, v[1] / n) for k, v in acc.items()}))
    if cols:
        keys = set()
        for _, d in cols:
            keys |= set(d)
        keys = sorted(keys, key=lambda k: -max(d.get(k, (0, 0))[0] for _, d in cols))[:top]
        print(" | ".join(c for c, _ in cols))
        for k in keys:
            cells = []
            for _, d in cols:
                us, cnt = d.get(k, (0, 0))
                cells.append(f"{us/1e3:7.2f}ms x{cnt:5.0f}")
            print(f"{k[:60]:60s} " + "  ".join(cells))
        for c, d in cols:
            print(f"sum-ops {c}: {sum(v[0] for v in d.values())/1e3:.2f} ms, ops {sum(v[1] for v in d.values()):.0f}")


if __name__ == "__main__":
    main()
