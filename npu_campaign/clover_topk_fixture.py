"""Build frozen top-k fixtures from captured Clover router logits (CPU run, ffn_moe_logits-<il>.f32 rows of 256 f32).

Usage: python clover_topk_fixture.py <capture_dir> <out_dir>
  fixture-raw.f32     : rows of real router logits (causal-path inputs of the router top-k), sampled over layers/positions
  fixture-coresetnorm.f32 : the per-block normalized expert mass vector (input of the coreset 256-wide sort, mostly exact zeros)
  fixture-masked.f32  : the same logits after the Clover coreset mask (softmax, top-8, per-block mass, sort, cumsum, top_p=0.9),
                        built per 16-token block like the diffusion draft graph: many exact -1e30 ties
Use with: set CLOVER_TOPK_FIXTURE=<out_dir>\\fixture-raw.f32;<out_dir>\\fixture-masked.f32 ; test-backend-ops test -b HTP0 -o TOP_K
"""
import sys, os, glob, re
import numpy as np

cap, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
files = sorted(glob.glob(os.path.join(cap, 'ffn_moe_logits-*.f32')), key=lambda p: int(re.findall(r'-(\d+)\.f32', p)[0]))
assert files, 'no captured logits'
rng = np.random.default_rng(0)
raw, masked, norms = [], [], []
TOP_P, K, W = 0.9, 8, 16
for p in files:
    a = np.fromfile(p, dtype=np.float32).reshape(-1, 256)
    n = a.shape[0]
    idx = rng.choice(n, size=min(n, 32), replace=False)
    raw.append(a[np.sort(idx)])
    for b in rng.choice(max(n // W, 1), size=min(max(n // W, 1), 2), replace=False):
        blk = a[b * W:(b + 1) * W]
        if blk.shape[0] == 0:
            continue
        e = np.exp(blk.astype(np.float64) - blk.max(1, keepdims=True))
        probs = (e / e.sum(1, keepdims=True)).astype(np.float32)
        ids = np.stack([sorted(range(256), key=lambda j: (-r[j], j))[:K] for r in blk])
        total = np.zeros(256, dtype=np.float32)
        for t in range(blk.shape[0]):
            total[ids[t]] += probs[t, ids[t]]
        norm = total / total.sum()
        norms.append(norm[None, :])
        order = sorted(range(256), key=lambda j: (-norm[j], j))
        ordered = norm[order]
        prev = np.cumsum(ordered, dtype=np.float32) - ordered
        keep = (TOP_P - prev) > 0
        allowed = np.zeros(256, dtype=np.float32)
        allowed[np.array(order)[keep]] = 1.0
        mask = allowed * np.float32(1e30) - np.float32(1e30)
        masked.append((blk + mask[None, :]).astype(np.float32))
raw = np.concatenate(raw)
masked = np.concatenate(masked)
norms = np.concatenate(norms).astype(np.float32)
norms.tofile(os.path.join(out, 'fixture-coresetnorm.f32'))
raw.tofile(os.path.join(out, 'fixture-raw.f32'))
masked.tofile(os.path.join(out, 'fixture-masked.f32'))
print('raw rows', raw.shape[0], 'masked rows', masked.shape[0],
      'avg masked entries/row', float((masked < -1e29).sum(1).mean()))
