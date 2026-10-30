"""trace agent: compare two llama-clover-draft-dump logit files (reference first, e.g. CPU; candidate e.g. HTP0).
Reports per-row KLD(ref||cand), argmax agreement and relative RMS of logits, for prefill rows and draft rows.
Usage: python draft_cmp.py ref.logits cand.logits [--rows]
"""
import argparse

import numpy as np


def load(path):
    raw = open(path, 'rb').read()
    n_vocab, n_prompt, n_pre, n_draft, anchor = np.frombuffer(raw[:20], dtype='<i4')
    off = 20 + 4 * n_prompt
    toks = np.frombuffer(raw[20:off], dtype='<i4')
    rows = np.frombuffer(raw[off:], dtype='<f4').reshape(-1, n_vocab)
    assert rows.shape[0] == n_pre + n_draft
    return dict(n_vocab=n_vocab, n_prompt=n_prompt, n_pre=n_pre, n_draft=n_draft, anchor=anchor, toks=toks,
                pre=rows[:n_pre], draft=rows[n_pre:])


def logsoftmax(x):
    x = x.astype(np.float64)
    m = x.max(1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))


def stats(name, a, b, per_row):
    if a.shape[0] == 0 or b.shape[0] == 0:
        return
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n], b[:n]
    la, lb = logsoftmax(a), logsoftmax(b)
    kl = (np.exp(la) * (la - lb)).sum(1)
    am, bm = a.argmax(1), b.argmax(1)
    rel = np.sqrt(((a - b) ** 2).sum(1) / (a ** 2).sum(1))
    print(f'{name}: rows {n} mean KLD {kl.mean():.6f} max {kl.max():.5f} argmax agree {int((am == bm).sum())}/{n} '
          f'mean rel-RMS {rel.mean():.3e}')
    if per_row:
        for i in range(n):
            print(f'  row {i:3d} argmax ref {am[i]:6d} cand {bm[i]:6d} {"==" if am[i] == bm[i] else "!="} KLD {kl[i]:.5f} rel {rel[i]:.3e}')


ap = argparse.ArgumentParser()
ap.add_argument('ref')
ap.add_argument('cand')
ap.add_argument('--rows', action='store_true')
a = ap.parse_args()
r, c = load(a.ref), load(a.cand)
assert np.array_equal(r['toks'], c['toks']), 'prompt tokens differ'
print(f'prompt {r["n_prompt"]} tokens; anchors ref {r["anchor"]} cand {c["anchor"]}')
stats('prefill', r['pre'], c['pre'], a.rows and r['n_pre'] <= 64)
if r['n_draft']:
    stats('draft (all 16 rows)', r['draft'], c['draft'], a.rows)
    stats('draft rows 1..15 (masks)', r['draft'][1:], c['draft'][1:], False)
