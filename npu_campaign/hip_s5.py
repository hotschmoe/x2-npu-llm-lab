# derive the exact weight-nibble -> k map of the int4 x u8 HMX MAC, then validate a random 32x32x32 product against numpy
import numpy as np
from hip_lib import *
from hip_scan import *

rng = np.random.default_rng(77)


def act_tile(a):
    t = np.zeros(2048, np.uint8)
    for r in range(32):
        for k in range(32):
            idx = (r // 2) * 64 + 2 * k + (r % 2)
            t[2 * idx + 1] = a[r, k]
    return t


def read_out(raw):
    u = np.frombuffer(raw[:2048], np.uint16).astype(np.int64)
    o = np.zeros((32, 32), np.int64)
    for pos in range(1024):
        r = 2 * (pos // 64) + pos % 2
        c = (pos % 64) // 2
        o[r, c] = u[pos] - 65536 if u[pos] >= 32768 else u[pos]
    return o


# stage 1: ramp act (value 2*(k+1) in column k of every row), single weight nibble e = 1  -> out(r, c(e)) = k(e) + 1
ramp = np.zeros((32, 32), np.uint8)
for k in range(32):
    ramp[:, k] = 2 * (k + 1)
rt = act_tile(ramp)
exps = [('ub', 'n', 'uh21', rt, pack_elem('n', e, 1), dict(rt_a=2047, rt_w=511)) for e in range(1024)]
res = run_batches(exps)
kmap = {}
for e, raw in enumerate(res):
    o = read_out(raw)
    nz = np.argwhere(o != 0)
    cols = sorted(set(nz[:, 1].tolist()))
    vals = sorted(set(o[o != 0].tolist()))
    kmap[e] = (cols, vals)
bad = [e for e, (c, v) in kmap.items() if len(c) != 1 or len(v) != 1]
print('elements with non-unique response:', len(bad), bad[:10])
col_of = {e: kmap[e][0][0] for e in kmap if e not in bad}
k_of = {e: kmap[e][1][0] - 1 for e in kmap if e not in bad}
print('col(e) == (e//8)%32 for all:', all(col_of[e] == (e // 8) % 32 for e in col_of))
print('first 40 k(e):', [k_of[e] for e in range(40)])
print('k(e) at e=256..263:', [k_of[e] for e in range(256, 264)], ' e=512..519:', [k_of[e] for e in range(512, 520)])
np.save('hip_data/s5_kmap.npy', np.array([k_of.get(e, -1) for e in range(1024)]))


# stage 2: random validation with the derived map
def wei_tile(w):
    t = np.zeros(2048, np.uint8)
    inv = {}
    for e in range(1024):
        inv[(col_of[e], k_of[e])] = e
    for k in range(32):
        for c in range(32):
            e = inv[(c, k)]
            v = int(w[k, c]) & 15
            byte, sub = divmod(e, 2)
            t[byte] |= v << (4 * sub)
    return t


tests = []
for tno in range(20):
    hi = 256 if tno < 8 else 32
    a = rng.integers(0, hi, (32, 32), dtype=np.uint8)
    w = rng.integers(-8, 8, (32, 32))
    tests.append((a, w))
j = Job()
slots = [experiment(j, 'ub', 'n', 'uh21', act_tile(a), wei_tile(w), rt_a=2047, rt_w=511) for a, w in tests]
outs = run_jobs([j])
raw = outs[0][1]
for tno, ((a, w), s) in enumerate(zip(tests, slots)):
    got = read_out(raw[s * 4096:(s + 1) * 4096])
    exact = a.astype(np.int64) @ w.astype(np.int64)
    # candidate model A: floor(exact*256/255/2) ; B: exact//2 ; report both errors
    mA = np.floor(exact * (256.0 / 255.0) / 2.0)
    mB = exact // 2
    print('test %2d: max|exact|/2 = %5d   err vs floor(exact*256/255/2): max %4d   err vs exact//2: max %4d' % (
        tno, np.abs(exact).max() // 2, np.abs(got - mA).max(), np.abs(got - mB).max()))
