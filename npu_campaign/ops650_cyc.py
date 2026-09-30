"""ops650: analyse a GGML_HEXAGON_OPS650_SPOOF_CYCLE run (profile=1): batch time and per-op-type time per spoof mode.
usage: ops650_cyc.py LOG [--nops 1384] [--byname] [--seg]"""
import re, sys, statistics as st
from collections import defaultdict
cyc = re.compile(r"OPS650 CYCLE call (\d+) mode (\d+)")
hdr = re.compile(r"profile-op OPBATCH\|[^|]*\|n-ops (\d+)\|.*usec (\d+) cycles (\d+) start (\d+)")
opl = re.compile(r"profile-op ([^|]+)\|([^|]*)\|([^|]*)\|([^|]*)\|[^|]*\|([^|]*)\|usec (\d+) cycles (\d+) start (\d+)")
segre = re.compile(r" seg (\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+)")
BIG = ('MUL_MAT', 'FLASH_ATTN_EXT', 'GATED_DELTA_NET', 'SSM_SCAN', 'SHEXP')
a = sys.argv[1:]
log = a[0]
nops = int(a[a.index('--nops') + 1]) if '--nops' in a else 1384
mode = 0; cur = None; batches = []
for line in open(log, encoding='utf-8', errors='replace'):
    m = cyc.search(line)
    if m: mode = int(m.group(2)); continue
    if 'profile-op' not in line: continue
    m = hdr.search(line)
    if m:
        cur = dict(mode=mode, n=int(m.group(1)), usec=int(m.group(2)), cyc=int(m.group(3)), ops=[], seg=[]); batches.append(cur); continue
    m = opl.search(line)
    if m and cur is not None:
        sg = segre.search(line)
        cur['ops'].append((m.group(1), int(m.group(7))))
        cur['seg'].append((m.group(1), [int(x) for x in sg.groups()] if sg else None))
sel = [b for b in batches if b['n'] == nops and len(b['ops']) == nops]
print(f'{len(sel)} decode batches of {nops} ops (of {len(batches)} total)')
names = {0: 'real', 1: 'skip all', 2: 'skip non-big', 3: 'skip big', 4: 'non-big -> empty fork/join', 5: 'non-big -> 16KB DMA rt', 6: 'all -> empty fork/join',
         7: 'non-big -> fj + DMA rt', 8: 'mm -> stream, rest skip', 9: 'mm -> stream, rest real', 10: 'mm -> stream, non-big skip, FA/GDN real',
         11: 'skip all + noflush', 12: 'real, non-big noflush', 13: 'real, all noflush'}
bym = defaultdict(list)
for b in sel: bym[b['mode']].append(b)
print(f"{'mode':>4s} {'description':42s} {'n':>3s} {'batch ms':>9s} {'sd':>6s} {'small-op mean us':>17s} {'big-op sum ms':>14s}")
for md in sorted(bym):
    bs = bym[md]
    mhz = st.mean(b['cyc'] / b['usec'] for b in bs)
    small = []; big = []
    for b in bs:
        for nme, c in b['ops']:
            (big if nme.startswith(BIG) else small).append(c / mhz)
    us = [b['usec'] for b in bs]
    print(f"{md:4d} {names.get(md, '?'):42s} {len(bs):3d} {st.mean(us) / 1e3:9.3f} {(st.pstdev(us) / 1e3 if len(us) > 1 else 0):6.3f} {st.mean(small):17.2f} {sum(big) / len(bs) / 1e3:14.3f}")
    if '--byname' in a and md in (1, 4, 6, 0, 11):
        acc = defaultdict(list)
        for b in bs:
            for nme, c in b['ops']: acc[nme].append(c / mhz)
        for nme, v in sorted(acc.items(), key=lambda kv: -sum(kv[1]))[:14]:
            print(f'        {nme:24s} n={len(v) / len(bs):6.1f} mean {st.mean(v):7.2f} us')
if '--seg' in a:
    print()
    print('segment means (us): prep | src flush | dst flush_all | dirty_all | merge (calls) | execute | active ranges   (small = non-matmul/FA/GDN)')
    for md in sorted(bym):
        bs = bym[md]; mhz = st.mean(b['cyc'] / b['usec'] for b in bs)
        agg = defaultdict(lambda: [0, [0.0] * 5, 0.0, 0.0])
        for b in bs:
            for nme, sg in b['seg']:
                if sg is None: continue
                kind = 'big' if nme.startswith(BIG) else 'small'
                A = agg[kind]; A[0] += 1
                for k, idx in enumerate((0, 1, 2, 3, 5)): A[1][k] += sg[idx] / mhz
                A[2] += (sg[4] & 0xffffff) / mhz; A[3] += (sg[4] >> 24)
        for kind, A in sorted(agg.items()):
            n = A[0]; v = A[1]
            print(f"mode {md:2d} {kind:5s} n={n / len(bs):6.0f}/batch  " + f"{v[0]/n:5.2f} | {v[1]/n:5.2f} | {v[2]/n:5.2f} | {v[3]/n:5.2f} | {A[2]/n:5.2f} ({A[3]/n:.1f}) | {v[4]/n:6.2f}")
