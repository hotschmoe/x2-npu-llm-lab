# impulse scans of the integer HMX modes; raw u16 outputs saved for offline decoding
import numpy as np, sys
from hip_lib import *
from hip_scan import *


def scan_act(ak, wk, rk, nimp, wones, val=1):
    exps = []
    for i in range(nimp):
        a = np.zeros(2048, np.uint8)
        a[i] = val
        exps.append((ak, wk, rk, a, wones, {}))
    return exps


def scan_wei(ak, wk, rk, nel, aones, val=1):
    exps = []
    for e in range(nel):
        exps.append((ak, wk, rk, aones, pack_elem(wk, e, val), {}))
    return exps


def to_u16(res):
    return np.stack([np.frombuffer(r[:2048], np.uint16) if r is not None else np.zeros(1024, np.uint16) for r in res])


aones = np.full(2048, 1, np.uint8)
data = {}
for wk, nel in (('n', 2048), ('b', 1200)):
    ones = np.full(2048, ONES_BYTE[wk], np.uint8)
    r = run_batches(scan_act('ub', wk, 'uh21', 2048 if wk == 'n' else 256, ones))
    data['act_' + wk] = to_u16(r)
    r = run_batches(scan_wei('ub', wk, 'uh21', nel, aones))
    data['wei_' + wk] = to_u16(r)
    # value tests: weight element 0 with all raw values; activation impulse 0 with 255
    vals = list(range(16)) if wk == 'n' else [0, 1, 2, 127, 128, 255]
    r = run_batches([('ub', wk, 'uh21', aones, pack_elem(wk, 0, v), {}) for v in vals])
    data['wval_' + wk] = to_u16(r)
    a255 = np.zeros(2048, np.uint8)
    a255[0] = 255
    r = run_batches([('ub', wk, 'uh21', a255, np.full(2048, ONES_BYTE[wk], np.uint8), {})])
    data['a255_' + wk] = to_u16(r)
np.savez('hip_data/s2b.npz', **data)
print('saved', {k: v.shape for k, v in data.items()})
