"""trace agent: frozen-input oracle for the Clover routed-expert path (layer L) from CPU/HTP tensor traces.

Needs traces with attn_post_norm-L, ffn_moe_topk-L, ffn_moe_gate-L, ffn_moe_up-L, ffn_moe_swiglu-L, ffn_moe_down-L.
For each device, recompute gate/up (Q4_0 experts x its own input) and down (x its own swiglu) in f64 and report
NMSE vs the device output, plus the CPU Q8_0-activation emulation. Tokens whose top-8 sets differ are reported.
Usage: python check_frozen_clover_moe.py cpu.trace htp.trace --gguf model.gguf [--layer 0]
"""
import argparse
import struct
import sys

import numpy as np

sys.path.insert(0, r'C:\Users\hotschmoe\Documents\llm_inference\llama.cpp-hexagon-clover-trace\gguf-py')
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402


def load(path, want):
    out = {}
    with open(path, 'rb') as f:
        while h := f.read(88):
            magic, ln, kind, op, *r = struct.unpack('<IIII4q4QQ', h)
            name = f.read(ln).decode()
            shape, strides, size = tuple(r[:4]), tuple(r[4:8]), r[8]
            if name in want and name not in out:
                buf = f.read(size)
                dt = {0: '<f4', 26: '<i4'}[kind]
                out[name] = np.ndarray(shape, dtype=dt, buffer=buf, strides=strides).copy(order='F'), shape
            else:
                f.seek(size, 1)
    return out


def q8_0(x):
    xb = x.reshape(x.shape[0], -1, 32).astype(np.float32)
    d = (np.abs(xb).max(-1, keepdims=True) / 127.0).astype(np.float32)
    q = np.round(xb * np.where(d > 0, 1.0 / d, 0.0).astype(np.float32))
    return (q * d.astype(np.float16).astype(np.float32)).reshape(x.shape)


def nmse(ref, x):
    d = ref - x
    return float((d * d).sum() / max((ref * ref).sum(), 1e-30))


ap = argparse.ArgumentParser()
ap.add_argument('cpu')
ap.add_argument('htp')
ap.add_argument('--gguf', required=True)
ap.add_argument('--layer', type=int, default=0)
a = ap.parse_args()
L = a.layer
names = {f'{n}-{L}' for n in ('attn_post_norm', 'ffn_moe_topk', 'ffn_moe_gate', 'ffn_moe_up', 'ffn_moe_swiglu', 'ffn_moe_down')}
rd = gguf.GGUFReader(a.gguf)
tens = {t.name: t for t in rd.tensors}


def experts(name):
    t = tens[name]
    ne0, ne1, ne2 = (int(x) for x in t.shape)
    return dequantize(t.data, t.tensor_type).astype(np.float32).reshape(ne2, ne1, ne0)  # [expert, out, in]


Wg, Wu, Wd = experts(f'blk.{L}.ffn_gate_exps.weight'), experts(f'blk.{L}.ffn_up_exps.weight'), experts(f'blk.{L}.ffn_down_exps.weight')
res = {}
for dev, path in (('cpu', a.cpu), ('htp', a.htp)):
    t = load(path, names)
    missing = names - t.keys()
    if missing:
        sys.exit(f'{dev}: missing {sorted(missing)}')
    x = t[f'attn_post_norm-{L}'][0][:, :, 0, 0].T                   # [tok, 2048]
    ids = t[f'ffn_moe_topk-{L}'][0][:, :, 0, 0].T                   # [tok, 8]
    g = np.transpose(t[f'ffn_moe_gate-{L}'][0][:, :, :, 0], (2, 1, 0))   # [tok, slot, 512]
    u = np.transpose(t[f'ffn_moe_up-{L}'][0][:, :, :, 0], (2, 1, 0))
    s = np.transpose(t[f'ffn_moe_swiglu-{L}'][0][:, :, :, 0], (2, 1, 0))
    dn = np.transpose(t[f'ffn_moe_down-{L}'][0][:, :, :, 0], (2, 1, 0))  # [tok, slot, 2048]
    T, S = ids.shape
    eg = np.zeros(g.shape); eu = np.zeros(u.shape); ed = np.zeros(dn.shape); qd = np.zeros(dn.shape); qg = np.zeros(g.shape)
    xq = q8_0(x).astype(np.float64)
    sq = q8_0(s.reshape(T * S, -1)).reshape(s.shape).astype(np.float64)
    for ti in range(T):
        for si in range(S):
            e = ids[ti, si]
            eg[ti, si] = Wg[e].astype(np.float64) @ x[ti].astype(np.float64)
            eu[ti, si] = Wu[e].astype(np.float64) @ x[ti].astype(np.float64)
            qg[ti, si] = Wg[e].astype(np.float64) @ xq[ti]
            ed[ti, si] = Wd[e].astype(np.float64) @ s[ti, si].astype(np.float64)
            qd[ti, si] = Wd[e].astype(np.float64) @ sq[ti, si]
    r = np.abs(s).max(-1) / np.sqrt((s.astype(np.float64) ** 2).mean(-1) + 1e-30)
    print(f'{dev}: gate vs exact {nmse(eg, g):.2e} (vs Q8_0-act emu {nmse(qg, g):.2e}) | up vs exact {nmse(eu, u):.2e} | '
          f'down vs exact {nmse(ed, dn):.2e} (vs Q8_0-act emu {nmse(qd, dn):.2e}) | swiglu absmax/rms med {np.median(r):.1f} max {r.max():.1f}')
    res[dev] = ids
same = np.array([set(a_) == set(b_) for a_, b_ in zip(res['cpu'], res['htp'])])
print(f'top-8 sets equal for {same.mean() * 100:.1f}% of tokens')
