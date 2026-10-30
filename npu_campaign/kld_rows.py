"""trace agent: per-row KLD between two llama-perplexity --kl-divergence-base logit files (e.g. CPU base vs an HTP
run saved with --kl-divergence-base). Prints summary, per-chunk KLD and the worst rows (chunk, pos, token, KLD).
Usage: python kld_rows.py base.bin cand.bin [--top 20] [--gguf model.gguf]
"""
import argparse
import struct

import numpy as np


def read(path):
    f = open(path, 'rb')
    assert f.read(8) == b'_logits_'
    n_ctx, n_vocab, n_chunk = struct.unpack('<iii', f.read(12))
    tokens = np.frombuffer(f.read(4 * n_chunk * n_ctx), dtype='<i4')
    nv = 2 * ((n_vocab + 1) // 2) + 4
    first = n_ctx // 2
    n_rows = n_ctx - 1 - first
    return f, n_ctx, n_vocab, n_chunk, tokens, nv, first, n_rows


def logprobs(raw, n_vocab):
    hdr = raw[:, :4].copy().view('<f4')  # scale, min_log_prob
    q = raw[:, 4:4 + n_vocab].astype(np.float32)
    lp = hdr[:, 1:2] + hdr[:, 0:1] * q
    return lp


ap = argparse.ArgumentParser()
ap.add_argument('base')
ap.add_argument('cand')
ap.add_argument('--top', type=int, default=20)
ap.add_argument('--gguf')
a = ap.parse_args()
fa, n_ctx, n_vocab, n_chunk, tok_a, nv, first, n_rows = read(a.base)
fb, n_ctx_b, n_vocab_b, n_chunk_b, tok_b, nv_b, _, _ = read(a.cand)
assert (n_ctx, n_vocab) == (n_ctx_b, n_vocab_b) and np.array_equal(tok_a[:n_chunk_b * n_ctx], tok_b[:n_chunk_b * n_ctx])
vocab = None
if a.gguf:
    import gguf
    r = gguf.GGUFReader(a.gguf)
    fld = r.fields['tokenizer.ggml.tokens']
    vocab = [bytes(fld.parts[i]).decode('utf-8', 'replace') for i in fld.data]
klds, same, rows = [], [], []
for c in range(min(n_chunk, n_chunk_b)):
    ra = np.frombuffer(fa.read(2 * nv * n_rows), dtype='<u2').reshape(n_rows, nv)
    rb = np.frombuffer(fb.read(2 * nv * n_rows), dtype='<u2').reshape(n_rows, nv)
    la, lb = logprobs(ra, n_vocab), logprobs(rb, n_vocab)
    pa = np.where(la > -16, np.exp(la), 0.0)  # llama-perplexity estimator
    kl = (pa * (la - lb)).sum(1)
    sm = la.argmax(1) == lb.argmax(1)
    klds.append(kl)
    same.append(sm)
    print(f'chunk {c + 1}: mean KLD {kl.mean():.5f} same-top {sm.mean() * 100:.2f}% max {kl.max():.4f}')
    for i in range(n_rows):
        rows.append((float(kl[i]), c, first + i, int(tok_a[c * n_ctx + first + i]), int(tok_a[c * n_ctx + first + i + 1]), bool(sm[i])))
k = np.concatenate(klds)
print(f'all: mean KLD {k.mean():.6f} median {np.median(k):.6f} same-top {np.concatenate(same).mean() * 100:.2f}%  rows {k.size}')
rows.sort(reverse=True)
for kl, c, pos, t, nxt, sm in rows[:a.top]:
    s = f' {vocab[t]!r} -> {vocab[nxt]!r}' if vocab else ''
    print(f'  KLD {kl:.4f} chunk {c + 1} pos {pos} tok {t} next {nxt} same_top {sm}{s}')
