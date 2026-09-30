"""ops650: per-op profile analysis of GGML_HEXAGON_PROFILE=1 logs (Qwen decode).
usage: ops650_prof.py LOG [--nops N] [--top K] [--dump]
Picks the most common batch class (or --nops), drops the first instance, and prints
 - batch totals (batch usec vs sum of op usec vs sum of op cycles/clock, inter-op gaps from cycle stamps)
 - per op-name table: count, total us, mean, median, min, share
 - size classes (small ops: fixed cost), bytes model: time = a + b*bytes fit for matmul-like ops
"""
import re, sys, statistics as st
from collections import defaultdict, Counter

BPE = {'f32': 4, 'f16': 2, 'bf16': 2, 'i32': 4, 'i64': 8, 'q4_0': 18/32, 'q4_1': 20/32, 'q5_0': 22/32, 'q5_1': 24/32, 'q8_0': 34/32,
       'q6_K': 210/256, 'q5_K': 176/256, 'q4_K': 144/256, 'q3_K': 110/256, 'q2_K': 84/256, 'q8_K': 292/256, 'mxfp4': 17/32}
hdr = re.compile(r"profile-op OPBATCH\|[^|]*\|n-ops (\d+)\|.*usec (\d+) cycles (\d+) start (\d+)")
opl = re.compile(r"profile-op ([^|]+)\|([^|]*)\|([^|]*)\|([^|]*)\|[^|]*\|([^|]*)\|usec (\d+) cycles (\d+) start (\d+)")

def load(path):
    bs, cur = [], None
    for line in open(path, encoding='utf-8', errors='replace'):
        if 'profile-op' not in line: continue
        m = hdr.search(line)
        if m:
            cur = dict(n=int(m.group(1)), usec=int(m.group(2)), cyc=int(m.group(3)), ops=[]); bs.append(cur); continue
        m = opl.search(line)
        if m and cur is not None:
            cur['ops'].append(dict(name=m.group(1), names=m.group(2), dims=m.group(3), types=m.group(4), mode=m.group(5), us=int(m.group(6)), cyc=int(m.group(7)), start=int(m.group(8))))
    return bs

def nel(d):
    n = 1
    for x in d.split(':'): n *= int(x)
    return n

def tbytes(dims, types):
    ds = re.split(r' x | -> ', dims); ts = re.split(r' x | -> ', types)
    out = []
    for d, t in zip(ds, ts):
        out.append(nel(d) * BPE.get(t, 4))
    return out, ds, ts

def touched_bytes(op):
    """bytes moved: weights (active experts only for ID ops) + activations in/out"""
    try:
        b, ds, ts = tbytes(op['dims'], op['types'])
    except Exception:
        return 0, 0
    name = op['name']
    if name.startswith('MUL_MAT_ID'):
        nw = 2 if name.endswith('_NX') else 1
        w = 0
        for i in range(nw):
            e = [int(x) for x in ds[i].split(':')]
            per = e[0] * e[1] * BPE.get(ts[i], 1)
            dst = [int(x) for x in ds[-1].split(':')]
            act = min(e[2], dst[1] * (dst[2] if len(dst) > 2 else 1)) if len(e) > 2 and len(dst) > 1 else 8
            w += per * act
        return w, sum(b[nw:])
    if name.startswith('MUL_MAT'):
        nw = 2 if name.startswith('MUL_MAT_NX') else 1
        return sum(b[:nw]), sum(b[nw:])
    return 0, sum(b)

def main():
    a = sys.argv[1:]
    path = a[0]
    nops = int(a[a.index('--nops') + 1]) if '--nops' in a else None
    top = int(a[a.index('--top') + 1]) if '--top' in a else 25
    bs = load(path)
    cls = Counter(b['n'] for b in bs)
    print('batch classes (n-ops:count):', dict(cls.most_common(6)))
    if nops is None:
        nops = max(cls, key=lambda k: (cls[k] >= 3, k) if False else (cls[k] * (k > 200), k))
    sel = [b for b in bs if b['n'] == nops and len(b['ops']) == nops]
    if len(sel) > 2: sel = sel[1:]
    print(f'class n-ops={nops}, instances={len(sel)}')
    B = len(sel)
    tot_batch = st.mean(b['usec'] for b in sel)
    mhz = st.mean(b['cyc'] / b['usec'] for b in sel)
    sum_op_us = st.mean(sum(o['us'] for o in b['ops']) for b in sel)
    sum_op_cyc = st.mean(sum(o['cyc'] for o in b['ops']) for b in sel)
    gaps = []
    for b in sel:
        ops = b['ops']
        g = 0
        for i in range(len(ops) - 1):
            d = (ops[i + 1]['start'] - (ops[i]['start'] + ops[i]['cyc'])) & 0xffffffff
            if d > 1 << 30: d = 0
            g += d
        gaps.append(g)
    gap_us = st.mean(gaps) / mhz
    print(f'batch {tot_batch/1e3:.2f} ms | sum(op usec) {sum_op_us/1e3:.2f} ms | sum(op cycles)/clk {sum_op_cyc/mhz/1e3:.2f} ms (clk {mhz:.0f} MHz) | inter-op gaps (cycle stamps) {gap_us/1e3:.2f} ms | batch - sum(op cyc) {(tot_batch - sum_op_cyc/mhz)/1e3:.2f} ms (start/end flush, prep)')
    acc = defaultdict(list)
    for b in sel:
        for o in b['ops']:
            acc[o['name']].append(o['cyc'] / mhz)
    print(f"{'op':34s} {'n/batch':>7s} {'ms/batch':>9s} {'share':>6s} {'mean us':>8s} {'med us':>7s} {'min us':>7s}")
    rows = sorted(acc.items(), key=lambda kv: -sum(kv[1]))
    tot = sum(sum(v) for v in acc.values()) / B
    for k, v in rows[:top]:
        print(f'{k[:34]:34s} {len(v)/B:7.1f} {sum(v)/B/1e3:9.3f} {100*sum(v)/B/tot:5.1f}% {st.mean(v):8.2f} {st.median(v):7.2f} {min(v):7.2f}')
    # size classes
    allo = [(o['cyc'] / mhz, o) for b in sel for o in b['ops']]
    for lo, hi in [(0, 8), (8, 12), (12, 20), (20, 50), (50, 150), (150, 1e9)]:
        s = [t for t, o in allo if lo <= t < hi]
        print(f'  ops with {lo:>4}..{hi if hi<1e9 else "inf":>4} us: {len(s)/B:7.1f}/batch  {sum(s)/B/1e3:7.3f} ms/batch ({100*sum(s)/B/tot:4.1f}%)')
    # fixed cost: no-weight ops
    small = [(t, o) for t, o in allo if not o['name'].startswith('MUL_MAT') and o['name'] not in ('FLASH_ATTN_EXT', 'GATED_DELTA_NET', 'GATED_DELTA_NET+CPY', 'SHEXP_EP')]
    print(f'non-matmul/FA/GDN ops: {len(small)/B:.1f}/batch, {sum(t for t,_ in small)/B/1e3:.3f} ms/batch, median {st.median([t for t,_ in small]):.2f} us')
    # OLS fit for matmul-likes: t = a + b*bytes
    xs, ys = [], []
    for t, o in allo:
        if o['name'].startswith('MUL_MAT'):
            w, act = touched_bytes(o)
            xs.append(w + act); ys.append(t)
    if len(xs) > 10:
        mx, my = st.mean(xs), st.mean(ys)
        sxx = sum((x - mx) ** 2 for x in xs); sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        bcoef = sxy / sxx; acoef = my - bcoef * mx
        ss_res = sum((y - acoef - bcoef * x) ** 2 for x, y in zip(xs, ys)); ss_tot = sum((y - my) ** 2 for y in ys)
        print(f'matmul fit: t(us) = {acoef:.2f} + {bcoef*1e6:.3f} us/MB (=> {1/(bcoef*1e6)*1e3:.1f} GB/s marginal), R2={1-ss_res/ss_tot:.3f}, n={len(xs)}, total weight+act MB/batch {sum(xs)/B/1e6:.0f}')
        # per-name
        byn = defaultdict(lambda: [0, 0, 0.0])
        for t, o in allo:
            if o['name'].startswith('MUL_MAT'):
                w, act = touched_bytes(o); r = byn[o['name']]; r[0] += 1; r[1] += w; r[2] += t
        for k, (n, w, t) in byn.items():
            print(f'   {k:20s} n={n/B:6.1f} weightMB={w/B/1e6:8.1f} ms={t/B/1e3:7.3f} eff GB/s={w/t/1e3 if t else 0:6.1f}' )
    if len(xs) > 10:
        print('   matmul-like by touched-bytes bucket:')
        edges = [0, 32e3, 128e3, 512e3, 2e6, 8e6, 32e6, 1e12]
        for lo, hi in zip(edges, edges[1:]):
            sel2 = [(x, y) for x, y in zip(xs, ys) if lo <= x < hi]
            if sel2:
                mxb = st.mean(x for x, _ in sel2); mt = st.mean(y for _, y in sel2)
                print(f'     {lo/1e3:8.0f}K..{hi/1e3:9.0f}K n={len(sel2)/B:6.1f}/batch mean {mxb/1e3:9.1f} KB {mt:7.2f} us  total {sum(y for _,y in sel2)/B/1e3:6.3f} ms  GB/s {mxb/mt/1e3:6.1f}  (t-6.5us: {mxb/max(mt-6.5,0.1)/1e3:6.1f})')
    if '--dump' in a:
        for o in sel[0]['ops']:
            print(f"{o['cyc']/mhz:8.2f} {o['name']:20s} {o['names'][:90]} | {o['dims'][:70]}")

if __name__ == '__main__':
    main()
