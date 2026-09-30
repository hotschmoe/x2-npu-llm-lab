# layout scans of the int4 x u8 MAC with values large enough to survive the >>1 readout, plus activation-value mapping
import numpy as np
from hip_lib import *
from hip_scan import *


def to_u16(res):
    return np.stack([np.frombuffer(r[:2048], np.uint16) if r is not None else np.zeros(1024, np.uint16) for r in res])


data = {}
NA, NW = 1100, 1100
w22 = np.full(2048, 0x22, np.uint8)
exps = []
for i in range(NA):
    a = np.zeros(2048, np.uint8)
    a[i] = 2
    exps.append(('ub', 'n', 'uh21', a, w22, {}))
data['act_n'] = to_u16(run_batches(exps))
a2 = np.full(2048, 2, np.uint8)
exps = []
for e in range(NW):
    exps.append(('ub', 'n', 'uh21', a2, pack_elem('n', e, 2), {}))
data['wei_n'] = to_u16(run_batches(exps))
# activation value mapping: constant act a, weights nibble 1 -> acc = 32*a_eff
vals = [0, 1, 2, 3, 4, 8, 16, 32, 64, 100, 127, 128, 129, 200, 254, 255]
exps = [('ub', 'n', 'uh21', np.full(2048, v, np.uint8), np.full(2048, 0x11, np.uint8), {}) for v in vals]
data['amap'] = to_u16(run_batches(exps))
np.savez('hip_data/s3b.npz', **data)
for v, row in zip(vals, data['amap']):
    print('act const', v, '-> uh21', np.unique(row)[:3].tolist())
print('saved', {k: v.shape for k, v in data.items()})
