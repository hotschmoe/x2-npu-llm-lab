# accumulation / range behaviour of the int accumulator (ub x n, uh21 readout)
import numpy as np
from hip_lib import *
from hip_scan import *


def nib(v):
    return np.full(2048, ((v & 15) << 4) | (v & 15), np.uint8)


K_UBN = ACT_KINDS['ub'] | (WEI_KINDS['n'] << 8)
j = Job()
res = []
a1 = np.full(2048, 1, np.uint8)
for nmac in (1, 2, 3, 4, 8):
    ai = len([s for s in j.steps if s[0] in (MAC, MACX)])
    aoff = woff = ai * 2048
    j.A[aoff:aoff + 2048] = a1
    j.W[woff:woff + 2048] = nib(1)
    j.step(CLR, 0)
    for _ in range(nmac):
        j.step(MAC, K_UBN, aoff, woff)
    slot = j.obytes // 4096
    j.step(READ, READ_KINDS['uh21'], slot * 4096, 0)
    j.obytes = (slot + 1) * 4096
    res.append(('accum', nmac, slot))
outs = run_jobs([j], isolate=False)
j2 = Job()
res2 = []
for wv in (7, -8, -1):
    for nmac in (1, 2):
        ai = len([s for s in j2.steps if s[0] in (MAC, MACX)])
        aoff = woff = ai * 2048
        j2.A[aoff:aoff + 2048] = 255
        j2.W[woff:woff + 2048] = nib(wv)
        j2.step(CLR, 0)
        for _ in range(nmac):
            j2.step(MAC, K_UBN, aoff, woff)
        for rk in ('uh21', 'satuh21'):
            slot = j2.obytes // 4096
            j2.step(READ, READ_KINDS[rk], slot * 4096, 0)
            j2.obytes = (slot + 1) * 4096
            res2.append((wv, nmac, rk, slot))
outs2 = run_jobs([j2], isolate=False)
for (lab, n, slot) in res:
    v = np.frombuffer(outs[0][1][slot * 4096:slot * 4096 + 2048], np.uint16)
    print(lab, n, np.unique(v)[:5])
for (wv, nmac, rk, slot) in res2:
    v = np.frombuffer(outs2[0][1][slot * 4096:slot * 4096 + 2048], np.uint16)
    u = np.unique(v)[:4]
    print('w=%d nmac=%d %s: raw %s as s16 %s' % (wv, nmac, rk, u.tolist(), u.astype(np.int16).tolist()))
