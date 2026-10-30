"""trace agent: per-tensor error table between two LLAMA_TENSOR_TRACE_FILE dumps (reference = CPU, candidate = HTP).

Records are aligned by (name, occurrence). Prints, in reference order, NMSE, max |d|, relative RMS and the fraction of
bit-different elements, flagging the first tensor whose NMSE exceeds --flag (default 1e-5).
Usage: python trace_nmse_table.py cpu.trace htp.trace [--name-regex RE] [--flag 1e-5] [--rows]
"""
import argparse
import re
import struct
from collections import Counter

import numpy as np

DT = {0: '<f4', 1: '<f2', 26: '<i4', 27: '<i8'}


def records(path):
    with open(path, 'rb') as f:
        while header := f.read(88):
            magic, length, kind, op, *rest = struct.unpack('<IIII4q4QQ', header)
            assert magic == 0x54524331, 'bad magic'
            name = f.read(length).decode()
            shape, strides, size = tuple(rest[:4]), tuple(rest[4:8]), rest[8]
            payload = f.read(size)
            if kind == 30:  # bf16
                raw = np.ndarray(shape, dtype='<u2', buffer=payload, strides=strides).ravel(order='F')
                vals = (raw.astype(np.uint32) << 16).view(np.float32)
            elif kind in DT:
                vals = np.ndarray(shape, dtype=DT[kind], buffer=payload, strides=strides).ravel(order='F').copy()
            else:
                vals = np.frombuffer(payload, dtype='u1').copy()
            yield name, shape, kind, op, vals


def load(path, pat, tail=False):
    out, seen = {}, Counter()
    order = []
    for name, shape, kind, op, vals in records(path):
        k = (name, seen[name])
        seen[name] += 1
        if pat and not pat.search(name):
            continue
        out[k] = (shape, kind, op, vals)
        order.append(k)
    if tail:
        # align occurrences from the end (e.g. the draft forward of an e2e run, whose prefill came first, against
        # an isolated run that only ran the draft forward)
        rk = {k: (k[0], k[1] - seen[k[0]]) for k in order}
        order = [rk[k] for k in order]
        out = {rk[k]: v for k, v in out.items()}
    return order, out


ap = argparse.ArgumentParser()
ap.add_argument('reference')
ap.add_argument('candidate')
ap.add_argument('--name-regex')
ap.add_argument('--flag', type=float, default=1e-5)
ap.add_argument('--rows', action='store_true', help='per-row (ne0) relative error summary for flagged tensors')
ap.add_argument('--limit', type=int, default=100000)
ap.add_argument('--tail', action='store_true', help='align occurrences of each name from the end of the traces')
a = ap.parse_args()
pat = re.compile(a.name_regex) if a.name_regex else None
ro, rd = load(a.reference, pat, a.tail)
co, cd = load(a.candidate, pat, a.tail)
flagged = False
n = 0
for k in ro:
    if k not in cd:
        print(f'{k[0]}#{k[1]}: missing in candidate')
        continue
    shape, kind, op, av = rd[k]
    cshape, ckind, _, bv = cd[k]
    if shape != cshape or kind != ckind:
        print(f'{k[0]}#{k[1]}: layout mismatch {shape}/{kind} vs {cshape}/{ckind}')
        continue
    n += 1
    if n > a.limit:
        break
    if kind in (0, 1, 30):
        x = av.astype(np.float64)
        y = bv.astype(np.float64)
        fin = np.isfinite(x) & np.isfinite(y)
        d = (x - y)[fin]
        den = max(float(np.dot(x[fin], x[fin])), 1e-30)
        nmse = float(np.dot(d, d)) / den
        mx = float(np.max(np.abs(d))) if d.size else 0.0
        diff = float(np.mean(x != y))
        nonfin = int((~np.isfinite(x)).sum()), int((~np.isfinite(y)).sum())
        mark = ''
        if nmse > a.flag and not flagged:
            mark = '  <== FIRST > flag'
            flagged = True
        print(f'{k[0]:<34s}#{k[1]} {str(shape):<28s} nmse={nmse:.3e} max={mx:.3e} rms_ref={np.sqrt(den / max(fin.sum(), 1)):.3e} '
              f'diff_frac={diff:.3f}' + (f' nonfinite={nonfin}' if any(nonfin) else '') + mark)
        if a.rows and mark and shape[0] > 0:
            xr = x.reshape(-1, shape[0])
            yr = y.reshape(-1, shape[0])
            e = np.sqrt(((xr - yr) ** 2).sum(1) / np.maximum((xr ** 2).sum(1), 1e-30))
            worst = np.argsort(-e)[:8]
            print('   worst rows (row, rel_rms):', [(int(i), float(f'{e[i]:.3e}')) for i in worst])
    else:
        mism = int(np.count_nonzero(av != bv))
        print(f'{k[0]:<34s}#{k[1]} {str(shape):<28s} int/raw mismatches={mism}/{av.size}')
