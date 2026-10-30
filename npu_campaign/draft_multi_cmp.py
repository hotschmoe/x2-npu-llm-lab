"""trace agent: compare teacher-forced multi-draft dumps (llama-clover-draft-dump with CLOVER_MULTI_STRIDE).
Each block: anchor = true token at p, rows 0..n-1 at positions p..p+n-1. Prints, per candidate vs the reference:
mask-row (rows 1..n-1) mean KLD(ref||cand), argmax agreement, and for every file the draft accuracy vs the true
tokens (row r predicts the token at p+r+1, 'shift', or at p+r, 'same'), plus how often the first k mask predictions
are all correct (accepted-prefix proxy).
Usage: python draft_multi_cmp.py ref.logits cand.logits [cand2.logits ...] [--draft-n 16]
"""
import argparse

import numpy as np


def load(path, n_draft):
    raw = open(path, 'rb').read()
    n_vocab, n_tok, n_pre, n_rows, anchor = np.frombuffer(raw[:20], dtype='<i4')
    assert n_pre == 0 and anchor < 0, 'not a multi-draft file'
    off = 20 + 4 * n_tok
    toks = np.frombuffer(raw[20:off], dtype='<i4')
    rows = np.frombuffer(raw[off:], dtype='<f4').reshape(n_rows // n_draft, n_draft, n_vocab)
    return toks, -anchor, rows


def logsoftmax(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


ap = argparse.ArgumentParser()
ap.add_argument('ref')
ap.add_argument('cands', nargs='+')
ap.add_argument('--draft-n', type=int, default=16)
a = ap.parse_args()
n = a.draft_n
toks, stride, R = load(a.ref, n)
B = R.shape[0]
ps = stride * (1 + np.arange(B))
true_shift = np.stack([toks[p + 1:p + n + 1] if p + n + 1 <= len(toks) else np.r_[toks[p + 1:], -1] for p in ps])
true_same = np.stack([toks[p:p + n] for p in ps])


def acc(rows, name):
    am = rows.argmax(-1)
    s = (am[:, 1:] == true_shift[:, 1:]).mean()
    m = (am[:, 1:] == true_same[:, 1:]).mean()
    ok = am[:, 1:] == true_same[:, 1:]   # mask row i (position p+i) predicts the token at p+i (measured: 'same' >> 'shift')
    pref = np.array([np.argmin(np.r_[r, False]) for r in ok])   # leading correct mask predictions per block
    print(f'{name:28s} mask-row accuracy vs truth: shift {s * 100:5.1f}%  same {m * 100:5.1f}% | row0 (anchor) next-token '
          f'acc {(am[:, 0] == true_shift[:, 0]).mean() * 100:5.1f}% | mean correct mask prefix (same) {pref.mean():.2f}')


print(f'{B} blocks x {n} rows, stride {stride}')
acc(R, a.ref.split('\\')[-1])
lr = None
for c in a.cands:
    t2, s2, C = load(c, n)
    assert np.array_equal(t2, toks) and s2 == stride and C.shape == R.shape
    acc(C, c.split('\\')[-1])
    kl_all, agree = [], []
    for b in range(B):
        la, lb = logsoftmax(R[b]), logsoftmax(C[b])
        kl_all.append((np.exp(la) * (la - lb)).sum(-1))
        agree.append(R[b].argmax(-1) == C[b].argmax(-1))
    kl = np.array(kl_all)
    ag = np.array(agree)
    print(f'  vs ref: mask rows KLD mean {kl[:, 1:].mean():.5f} median {np.median(kl[:, 1:]):.5f} | argmax agree '
          f'{int(ag[:, 1:].sum())}/{ag[:, 1:].size} ({ag[:, 1:].mean() * 100:.1f}%) | row0 KLD {kl[:, 0].mean():.5f} agree '
          f'{int(ag[:, 0].sum())}/{B} | per-block mask KLD min/max {kl[:, 1:].mean(1).min():.4f}/{kl[:, 1:].mean(1).max():.4f}')
