"""trace agent: KLD of a llama-clover-draft-dump prefill (one full perplexity chunk, CLOVER_ALL_LOGITS=1) against the
matching chunk of a llama-perplexity --kl-divergence-base file (rows n_ctx/2 .. n_ctx-2, like llama-perplexity).
Usage: python dump_vs_kldbase.py base.bin dump.logits --chunk 4
"""
import argparse
import struct

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('base')
ap.add_argument('dump')
ap.add_argument('--chunk', type=int, required=True)
a = ap.parse_args()
f = open(a.base, 'rb')
assert f.read(8) == b'_logits_'
n_ctx, n_vocab, n_chunk = struct.unpack('<iii', f.read(12))
tokens = np.frombuffer(f.read(4 * n_chunk * n_ctx), dtype='<i4')
nv = 2 * ((n_vocab + 1) // 2) + 4
first = n_ctx // 2
n_rows = n_ctx - 1 - first
f.seek(2 * nv * n_rows * (a.chunk - 1), 1)
raw = np.frombuffer(f.read(2 * nv * n_rows), dtype='<u2').reshape(n_rows, nv)
hdr = raw[:, :4].copy().view('<f4')
lp_base = hdr[:, 1:2].astype(np.float64) + hdr[:, 0:1] * raw[:, 4:4 + n_vocab].astype(np.float64)

d = open(a.dump, 'rb').read()
hv = np.frombuffer(d[:20], dtype='<i4')
off = 20 + 4 * int(hv[1])
dt = np.frombuffer(d[20:off], dtype='<i4')
assert np.array_equal(dt, tokens[(a.chunk - 1) * n_ctx:a.chunk * n_ctx]), 'dump tokens != chunk tokens'
rows = np.frombuffer(d[off:], dtype='<f4').reshape(-1, n_vocab)[first:first + n_rows].astype(np.float64)
m = rows.max(1, keepdims=True)
lp = rows - m - np.log(np.exp(rows - m).sum(1, keepdims=True))
# same estimator as llama-perplexity: only base log-probs > -16 contribute
p = np.where(lp_base > -16, np.exp(lp_base), 0.0)
kl = (p * (lp_base - lp)).sum(1)
same = lp_base.argmax(1) == lp.argmax(1)
print(f'chunk {a.chunk}: rows {n_rows} mean KLD(base||dump) {kl.mean():.5f} same-top {same.mean() * 100:.2f}% max {kl.max():.4f}')
