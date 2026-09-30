"""ops650: per-op-type fit (time = a + b*bytes) and per-layer-position profile from a GGML_HEXAGON_PROFILE=1 log.
usage: ops650_layers.py LOG [--nops N]"""
import re, sys, statistics as st
from collections import defaultdict
import ops650_prof as P
a = sys.argv[1:]
bs = P.load(a[0])
from collections import Counter
cls = Counter(b['n'] for b in bs)
nops = int(a[a.index('--nops') + 1]) if '--nops' in a else max((k for k in cls if cls[k] >= 3), key=lambda k: cls[k] * (k > 200))
sel = [b for b in bs if b['n'] == nops and len(b['ops']) == nops][1:]
B = len(sel)
mhz = st.mean(b['cyc'] / b['usec'] for b in sel)
def layer_of(o):
    for txt in (o['names'].split(' -> ')[-1], o['names']):
        m = re.search(r'blk\.(\d+)\.', txt) or re.search(r'-(\d+)(?: |$)', txt)
        if m: return int(m.group(1))
    return -1
def total_bytes(o):
    try:
        b, ds, ts = P.tbytes(o['dims'], o['types']); return sum(b)
    except Exception: return 0
print(f'{a[0]}  class n-ops={nops}, {B} warm batches, batch {st.mean(b["usec"] for b in sel)/1e3:.2f} ms')
# per op type fit
by = defaultdict(list)
for b in sel:
    for o in b['ops']:
        w, act = P.touched_bytes(o)
        by[o['name']].append((w + act if o['name'].startswith('MUL_MAT') else total_bytes(o), o['cyc'] / mhz))
print(f"{'op':22s} {'n/tok':>6s} {'ms/tok':>7s} {'mean us':>8s} {'a (us)':>7s} {'b (us/MB)':>10s} {'R2':>5s} {'mean KB':>9s}")
for name, v in sorted(by.items(), key=lambda kv: -sum(t for _, t in kv[1]))[:26]:
    xs = [x for x, _ in v]; ys = [y for _, y in v]
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx > 1e-9 and len(set(xs)) > 1:
        bc = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx; ac = my - bc * mx
        ss_res = sum((y - ac - bc * x) ** 2 for x, y in zip(xs, ys)); ss_tot = sum((y - my) ** 2 for y in ys)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"{name[:22]:22s} {len(v)/B:6.1f} {sum(ys)/B/1e3:7.3f} {my:8.2f} {ac:7.2f} {bc*1e6:10.3f} {r2:5.2f} {mx/1e3:9.1f}")
    else:
        print(f"{name[:22]:22s} {len(v)/B:6.1f} {sum(ys)/B/1e3:7.3f} {my:8.2f} {'-':>7s} {'-':>10s} {'-':>5s} {mx/1e3:9.1f}")
# per layer
lay = defaultdict(lambda: [0.0, 0])
for b in sel:
    for o in b['ops']:
        L = layer_of(o); lay[L][0] += o['cyc'] / mhz; lay[L][1] += 1
tot = sum(v[0] for v in lay.values()) / B
print('\nper layer position (ms/token, ops): layers 0.. ; -1 = embedding/head/misc')
row = []
for L in sorted(lay):
    row.append(f'{L}:{lay[L][0]/B/1e3:.2f}({lay[L][1]//B})')
print(' '.join(row))
gdn = [lay[L][0] / B / 1e3 for L in lay if L >= 0 and (L + 1) % 4 != 0]
att = [lay[L][0] / B / 1e3 for L in lay if L >= 0 and (L + 1) % 4 == 0]
print(f'GDN-type layers (30): mean {st.mean(gdn):.3f} ms, total {sum(gdn):.2f} ms | attention-type layers (10): mean {st.mean(att):.3f} ms, total {sum(att):.2f} ms | other {lay[-1][0]/B/1e3:.2f} ms | sum {tot/1e3:.2f} ms')
