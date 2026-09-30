# functional validation of the int4 x u8 HMX MAC against a numpy model, plus deep-mode semantics
import numpy as np
from hip_lib import *
from hip_scan import *

rng = np.random.default_rng(1234)


def act_tile(a):
    """a: [32 rows][32 k] uint8 -> 2 KB tile (value in the high byte of each halfword, fp16-tile layout)"""
    t = np.zeros(2048, np.uint8)
    for r in range(32):
        for k in range(32):
            idx = (r // 2) * 64 + 2 * k + (r % 2)
            t[2 * idx + 1] = a[r, k]
    return t


def wei_tile_n(w):
    """w: [32 k][32 cols] int in [-8,7] -> 512 B: group g (k=8g..8g+7), word c: nibble j at byte g*128+c*4+j//2"""
    t = np.zeros(2048, np.uint8)
    for g in range(4):
        for c in range(32):
            for j in range(8):
                v = int(w[8 * g + j, c]) & 15
                off = g * 128 + c * 4 + j // 2
                t[off] |= v << (4 * (j % 2))
    return t


def read_out(raw):
    u = np.frombuffer(raw[:2048], np.uint16).astype(np.int64)
    o = np.zeros((32, 32), np.int64)
    for pos in range(1024):
        r = 2 * (pos // 64) + pos % 2
        c = (pos % 64) // 2
        o[r, c] = u[pos] - 65536 if u[pos] >= 32768 else u[pos]
    return o


tests = []
for tno in range(12):
    if tno < 4:
        a = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    else:
        a = rng.integers(0, 32, (32, 32), dtype=np.uint8)
    w = rng.integers(-8, 8, (32, 32))
    tests.append((a, w))
j = Job()
slots = []
for a, w in tests:
    slots.append(experiment(j, 'ub', 'n', 'uh21', act_tile(a), wei_tile_n(w), rt_a=2047, rt_w=511))
# deep-mode semantics with constant data
d_slots = []
for ak, wk, rta, rtw in (('ub', 'b_deep', 2047, 2047), ('ub_deep', 'b', 2047, 2047), ('ub', 'b', 2047, 1023), ('ub', 'n_deep', 2047, 2047), ('ub', 'n', 2047, 511)):
    a = np.zeros(2048, np.uint8)
    a[1::2] = 2
    wb = np.full(2048, 0x11 if wk.startswith('n') else 0x01, np.uint8)
    d_slots.append((ak, wk, rta, rtw, experiment(j, ak, wk, 'uh21', a, wb, rt_a=rta, rt_w=rtw)))
outs = run_jobs([j])
raw = outs[0][1]
bad = 0
for (a, w), s in zip(tests, slots):
    got = read_out(raw[s * 4096:(s + 1) * 4096])
    exact = a.astype(np.int64) @ w.astype(np.int64)     # [32 rows][32 cols]
    # model: hardware = floor-ish( (256/255) * exact / 2 ), ones' complement quirk for negatives
    model = np.floor(exact * (256.0 / 255.0) / 2.0)
    err = got - model
    print('test: max|got-model| = %d  (max|exact/2|=%d)  exact-scale err rel = %.4g' % (np.abs(err).max(), np.abs(exact / 2).max(),
          np.abs(got - exact / 2.0).max() / max(1, np.abs(exact / 2.0).max())))
    bad += int(np.abs(err).max() > 2)
print('functional model mismatches (>2 LSB):', bad)
for ak, wk, rta, rtw, s in d_slots:
    v = np.frombuffer(raw[s * 4096:s * 4096 + 2048], np.uint16)
    u, cnt = np.unique(v, return_counts=True)
    print('act %s x wei %s rt %d/%d: values %s counts %s' % (ak, wk, rta, rtw, u[:6].tolist(), cnt[:6].tolist()))
