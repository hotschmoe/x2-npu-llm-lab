# unrolled asm HMX issue-rate benchmark
import numpy as np
from hip_lib import *
names = {0: 'fp16 pair', 1: 'ub x n (int4) pair', 2: 'ub x b (int8) pair', 3: 'ub x c pair', 4: 'fp16 pair + hf readout 2KB each',
         5: 'int4 pair + uh2x1 readout each', 6: 'clracc + int4 pair + uh2x1 readout each', 7: 'clracc.hf + fp16 pair + hf readout each', 8: 'fp16 pairs on 8 distinct tiles'}
IT = 5000
jobs = []
for k in range(9):
    j = Job()
    j.A[:16384] = 1; j.W[:16384] = 0x11
    j.step(CLR, 0)
    j.step(BENCH2, k, IT, 2047)
    j.obytes = 4096
    jobs.append(j)
outs = run_jobs(jobs, isolate=True)
FREQ = 2.112  # GHz, from HVX bench (pcycles/us)
for k, o in enumerate(outs):
    if o is None: print(k, 'NO OUT'); continue
    cyc = int(o[0][2]); pairs = IT * 8
    print(f'{names[k]:45s}: {cyc/pairs:7.2f} cyc/pair  = {cyc/pairs/FREQ:6.2f} ns/pair  -> {65536/(cyc/pairs/FREQ)/1e3:6.2f} TFLOP/s-equivalent (32x32x32 pair)')
