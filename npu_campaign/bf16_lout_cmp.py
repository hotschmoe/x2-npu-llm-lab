"""bf16 agent: per-layer NMSE of l_out between two LLAMA_TENSOR_TRACE dumps (reference first). Usage: python -I bf16_lout_cmp.py ref.trace cand.trace"""
import struct, sys
import numpy as np
def recs(path):
    out = []
    with open(path, 'rb') as f:
        while header := f.read(88):
            magic, length, kind, op, *rest = struct.unpack('<IIII4q4QQ', header)
            name = f.read(length).decode(); size = rest[8]; payload = f.read(size)
            if kind == 0 and name.startswith('l_out-') and '(copy)' not in name:
                out.append((name, tuple(rest[:4]), np.frombuffer(payload, dtype='<f4').astype(np.float64)))
    return out
a, b = recs(sys.argv[1]), recs(sys.argv[2])
print(len(a), len(b))
for (n1, s1, v1), (n2, s2, v2) in zip(a, b):
    if v1.shape != v2.shape: print(n1, s1, 'shape mismatch', s2); continue
    nm = ((v1 - v2) ** 2).sum() / max((v1 ** 2).sum(), 1e-30)
    print(f'{n1:10s} {str(s1[:2]):12s} nmse {nm:.3e}  bitdiff {np.mean(v1 != v2):.3f}')
