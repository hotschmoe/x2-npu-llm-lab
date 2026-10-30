"""trace agent: frozen-input oracle for Clover dense mul_mats from CPU/HTP LLAMA_TENSOR_TRACE dumps.

For each (weight, input tensor, output tensor) triple: dequantize the GGUF weight and compare both device outputs
against exact f64 on that device's own input, plus emulations of the CPU path (activation -> Q8_0 blocks of 32 with
fp16 scale) and the HTP/HMX path (activation and weight -> fp16, wide accumulate, output -> fp16).
Usage: python check_frozen_clover_mm.py cpu.trace htp.trace --gguf model.gguf [--layer 0] [--rows N]
"""
import argparse
import struct
import sys

import numpy as np

sys.path.insert(0, r'C:\Users\hotschmoe\Documents\llm_inference\llama.cpp-hexagon-clover-trace\gguf-py')
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402


def find(path, want):
    out = {}
    with open(path, 'rb') as f:
        while h := f.read(88):
            magic, ln, kind, op, *r = struct.unpack('<IIII4q4QQ', h)
            name = f.read(ln).decode()
            shape, strides, size = tuple(r[:4]), tuple(r[4:8]), r[8]
            if name in want and name not in out and kind == 0:
                buf = f.read(size)
                out[name] = np.ndarray(shape, dtype='<f4', buffer=buf, strides=strides).ravel(order='F')
            else:
                f.seek(size, 1)
            if len(out) == len(want):
                break
    return out


def nmse(ref, x):
    d = ref.astype(np.float64) - x.astype(np.float64)
    return float((d * d).sum() / max((ref.astype(np.float64) ** 2).sum(), 1e-30))


def q8_0(x):
    xb = x.reshape(x.shape[0], -1, 32).astype(np.float32)
    d = (np.abs(xb).max(-1, keepdims=True) / 127.0).astype(np.float32)
    q = np.round(xb * np.where(d > 0, 1.0 / d, 0.0).astype(np.float32))
    return (q * d.astype(np.float16).astype(np.float32)).reshape(x.shape)


ap = argparse.ArgumentParser()
ap.add_argument('cpu')
ap.add_argument('htp')
ap.add_argument('--gguf', required=True)
ap.add_argument('--layer', type=int, default=0)
ap.add_argument('--rows', type=int, default=0)
a = ap.parse_args()
L = a.layer
triples = [
    (f'blk.{L}.attn_qkv.weight', f'attn_norm-{L}', f'linear_attn_qkv_mixed-{L}'),
    (f'blk.{L}.attn_gate.weight', f'attn_norm-{L}', f'z-{L}'),
    (f'blk.{L}.ssm_alpha.weight', f'attn_norm-{L}', f'alpha-{L}'),
    (f'blk.{L}.ssm_beta.weight', f'attn_norm-{L}', f'beta-{L}'),
    (f'blk.{L}.ssm_out.weight', f'final_output-{L}', f'linear_attn_out-{L}'),
    (f'blk.{L}.ffn_gate_inp.weight', f'attn_post_norm-{L}', f'ffn_moe_logits-{L}'),
]
rd = gguf.GGUFReader(a.gguf)
tens = {t.name: t for t in rd.tensors}
names = {n for tr in triples for n in tr[1:]}
c, h = find(a.cpu, names), find(a.htp, names)
print(f'{"weight":<28s} {"type":<5s} {"in d":>9s} {"out d":>9s} | {"cpu-exact":>9s} {"cpu-q8emu":>9s} | {"htp-exact":>9s} {"htp-f16emu":>10s} | absmax/rms')
for wn, xn, yn in triples:
    if wn not in tens or xn not in c or yn not in c:
        print(f'{wn}: missing')
        continue
    t = tens[wn]
    W = dequantize(t.data, t.tensor_type).astype(np.float32).reshape(int(t.shape[1]), int(t.shape[0]))
    K, N = W.shape[1], W.shape[0]
    Xc, Xh = c[xn].reshape(-1, K), h[xn].reshape(-1, K)
    Yc, Yh = c[yn].reshape(-1, N), h[yn].reshape(-1, N)
    if a.rows:
        Xc, Xh, Yc, Yh = Xc[:a.rows], Xh[:a.rows], Yc[:a.rows], Yh[:a.rows]
    W64 = W.astype(np.float64)
    ex_c, ex_h = Xc.astype(np.float64) @ W64.T, Xh.astype(np.float64) @ W64.T
    emu_c = q8_0(Xc).astype(np.float64) @ W64.T
    emu_h = (Xh.astype(np.float16).astype(np.float64) @ W.astype(np.float16).astype(np.float64).T).astype(np.float16)
    r = np.abs(Xc).max(1) / np.sqrt((Xc.astype(np.float64) ** 2).mean(1))
    print(f'{wn:<28s} {t.tensor_type.name:<5s} {nmse(Xc, Xh):9.2e} {nmse(Yc, Yh):9.2e} | {nmse(ex_c, Yc):9.2e} {nmse(emu_c, Yc):9.2e} | '
          f'{nmse(ex_h, Yh):9.2e} {nmse(emu_h, Yh):10.2e} | med {np.median(r):.1f} max {r.max():.1f}')
    if 'gate_inp' in wn:
        k = 8
        sc = np.argsort(-Yc, axis=1, kind='stable')[:, :k]
        sh = np.argsort(-Yh, axis=1, kind='stable')[:, :k]
        se = np.argsort(-ex_c, axis=1, kind='stable')[:, :k]
        diff = [len(set(x) ^ set(y)) // 2 for x, y in zip(sc, sh)]
        diff_e = [len(set(x) ^ set(y)) // 2 for x, y in zip(sc, se)]
        diff_he = [len(set(x) ^ set(y)) // 2 for x, y in zip(sh, np.argsort(-ex_h, axis=1, kind='stable')[:, :k])]
        print(f'   router top-8 sets: cpu vs htp differ in {np.mean(np.array(diff) > 0) * 100:.1f}% of tokens '
              f'(mean {np.mean(diff):.3f} experts); cpu vs exact(cpu in) {np.mean(np.array(diff_e) > 0) * 100:.1f}%; '
              f'htp vs exact(htp in) {np.mean(np.array(diff_he) > 0) * 100:.1f}%')
