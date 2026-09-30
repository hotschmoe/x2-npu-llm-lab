# HMX MAC-pair issue rate for several mode pairs (same tiles repeated; drained by a readout inside the BENCH step)
import numpy as np
from hip_lib import *
combos = [('hf', 'hf'), ('ub', 'b'), ('ub', 'n'), ('ub', 'n2x'), ('ub', 'c'), ('ub', 'sc'), ('ub', 'sm'), ('ub', 'ubit'), ('f8', 'f8'), ('hf', 'n'), ('hf_deep', 'hf'), ('ub_deep', 'b')]
N = 4000
jobs = []
for ak, wk in combos:
    j = Job()
    j.A[:2048] = 1; j.W[:2048] = 0x11
    j.step(CLR, 1 if ak.startswith('hf') else 0)
    j.step(BENCH, ACT_KINDS[ak] | (WEI_KINDS[wk] << 8), N, 2047)
    j.obytes = 4096
    jobs.append(j)
outs = run_jobs(jobs, isolate=True)
for (ak, wk), o, j in zip(combos, outs, jobs):
    if o is None: print(ak, wk, 'NO OUT'); continue
    cyc = o[0]
    print(f'{ak:8s} x {wk:5s}: BENCH step cycles={cyc[2]} -> {cyc[2]/N:.2f} cyc/pair (incl one drain readout)')
