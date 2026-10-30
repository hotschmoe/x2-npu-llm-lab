"""trace agent: compare Clover draft-path expert routing between tensor traces (first trace = reference).
Per layer: coreset (experts left unmasked by build_clover_coreset, shared by the whole draft block) differences and
per-token top-8 set differences. Needs clover_router_masked-L and ffn_moe_topk-L in the traces (filter '.').
Usage: python coreset_cmp.py ref.trace cand.trace [cand2.trace ...] [--layers 0,1,2]
"""
import argparse
import re
import struct

import numpy as np

PAT = re.compile(r'^(clover_router_masked|ffn_moe_topk)-(\d+)$')


def load(path, layers):
    out = {}
    with open(path, 'rb') as f:
        while h := f.read(88):
            magic, ln, kind, op, *r = struct.unpack('<IIII4q4QQ', h)
            name = f.read(ln).decode()
            shape, strides, size = tuple(r[:4]), tuple(r[4:8]), r[8]
            m = PAT.match(name)
            if m and (layers is None or int(m.group(2)) in layers):
                buf = f.read(size)
                dt = {0: '<f4', 26: '<i4'}[kind]
                out[name] = np.ndarray(shape, dtype=dt, buffer=buf, strides=strides).copy(order='F')[:, :, 0, 0]
            else:
                f.seek(size, 1)
    return out


ap = argparse.ArgumentParser()
ap.add_argument('traces', nargs='+')
ap.add_argument('--layers')
a = ap.parse_args()
layers = {int(x) for x in a.layers.split(',')} if a.layers else None
tr = [load(p, layers) for p in a.traces]
ref = tr[0]
ls = sorted({int(PAT.match(n).group(2)) for n in ref if n.startswith('ffn_moe_topk')})
for ci, cand in enumerate(tr[1:], 1):
    print(f'== ref {a.traces[0].split(chr(92))[-1]}  cand {a.traces[ci].split(chr(92))[-1]}')
    tot_core, tot_tok = 0, 0
    for L in ls:
        rm, cm = ref.get(f'clover_router_masked-{L}'), cand.get(f'clover_router_masked-{L}')
        rt, ct = ref[f'ffn_moe_topk-{L}'], cand[f'ffn_moe_topk-{L}']
        core = ''
        if rm is not None and cm is not None:
            ra, ca = rm[:, 0] > -1e29, cm[:, 0] > -1e29
            nd = int((ra != ca).sum())
            tot_core += nd > 0
            core = f'coreset size {int(ra.sum())}/{int(ca.sum())} differ {nd}'
        nt = sum(set(rt[:, t]) != set(ct[:, t]) for t in range(rt.shape[1]))
        tot_tok += nt
        print(f'  layer {L:2d}: {core:32s} tokens with different top-8 set {nt}/{rt.shape[1]}')
    print(f'  layers with a different coreset: {tot_core}/{len(ls)}; token-layer top-8 differences {tot_tok}/{len(ls) * rt.shape[1]}')
