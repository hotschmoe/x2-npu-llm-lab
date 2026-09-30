import sys, numpy as np
from hip_lib import *
combos = [('ub', w) for w in ['b', 'ubit', 'c', 'n', 'sbit', 'sc', 'sm', 'n2x']] + [('hf', w) for w in ['n', 'n2x', 'ubit', 'b', 'c', 'sm', 'sc']] + [('f8', 'f8'), ('ub', 'hf'), ('hf', 'f8'), ('f8', 'hf')]
pats = {'01': 0x01, '11': 0x11, '55': 0x55, 'FF': 0xFF}
reads = ['uh21', 'uh22', 'hf', 'ub']
jobs, meta = [], []
for ak, wk in combos:
    j = Job()
    for pn, pv in pats.items():
        if ak == 'ub': a = np.full(2048, 1, np.uint8)
        elif ak == 'hf': a = np.full(1024, 1.0, np.float16).view(np.uint8)
        else: a = np.full(2048, 0x38, np.uint8)   # e4m3 1.0 = 0x38
        w = np.full(2048, pv, np.uint8)
        if wk == 'hf': w = np.full(1024, 1.0, np.float16).view(np.uint8)
        if wk == 'f8': w = np.full(2048, 0x38, np.uint8)
        sl = mac_reads(j, ak, wk, a, w, reads)
        meta.append((ak, wk, pn, sl))
    jobs.append(j)
outs = run_jobs(jobs, isolate=True)
for j, o, (ak, wk) in zip(jobs, outs, combos):
    print(f'=== act {ak} x wei {wk}', 'NO OUTPUT' if o is None else '')
    if o is None: continue
    cyc, O = o
    for (a2, w2, pn, sl) in [m for m in meta if m[0] == ak and m[1] == wk]:
        line = []
        for rk, s in zip(reads, sl):
            v = decode(O[s*4096:(s+1)*4096], rk)
            u = np.unique(v)
            line.append(f'{rk}: n={len(u)} nz={int((v!=0).sum())} vals={u[:6].tolist()}')
        print(' pat', pn, ' | '.join(line))
