# unrolled HMX pair issue-rate sweep over activation/weight load variants and range arguments (cycles per 32x32x32 pair)
import numpy as np
from hip_lib import *

ACTS = ['ub', 'ub:deep', 'ub:cm', 'ub:deep:cm', 'ub:single', 'ub:above', 'hf', 'hf:deep']
WEIS = ['b', 'n', 'n:2x', 'c', 'sc', 'sm', 'ubit', 'sbit', 'n:deep', 'b:deep', 'n:single', 'b:single', 'n:2x:single', 'hf']
IT = 2000
FREQ = 2.112


def mk(ka, kw, rt):
    j = Job()
    j.A[:16384] = 1
    j.W[:16384] = 0x11
    j.step(CLR, 0)
    j.step(BENCH3, ka * 32 + kw, IT, rt)
    j.obytes = 4096
    return j


BENCH3 = 21
cases = []
for ka in range(6):
    for kw in range(13):
        cases.append((ka, kw, 2047 | (2047 << 16), 'act %s x wei %s rt 2047/2047' % (ACTS[ka], WEIS[kw])))
for rta in (2047, 1023, 511):
    for rtw in (2047, 1023, 511, 255):
        for (ka, kw) in ((0, 0), (0, 1)):
            cases.append((ka, kw, rta | (rtw << 16), 'act %s x wei %s rt %d/%d' % (ACTS[ka], WEIS[kw], rta, rtw)))
cases.append((6, 13, 2047 | (2047 << 16), 'fp16 baseline (hf x hf)'))
cases.append((7, 13, 2047 | (2047 << 16), 'fp16 hf:deep x hf'))
for k, lab in ((0, 'f8 x f8'), (1, 'f8:deep x f8'), (2, 'hf x hf:single'), (3, 'hf:single x hf:single')):
    cases.append((8, k, 2047 | (2047 << 16), lab))

# run in groups of 16 per process; on failure re-run the group isolated
res = {}
for g in range(0, len(cases), 16):
    grp = cases[g:g + 16]
    jobs = [mk(ka, kw, rt) for (ka, kw, rt, _) in grp]
    outs = run_jobs(jobs, isolate=True)
    for (ka, kw, rt, lab), o in zip(grp, outs):
        if o is None:
            print('%-50s NO OUT (crash?)' % lab)
        else:
            cyc = int(o[0][2])
            print('%-50s %7.2f cyc/pair  %6.2f ns' % (lab, cyc / (IT * 8), cyc / (IT * 8) / FREQ), flush=True)
