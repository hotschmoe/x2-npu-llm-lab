import numpy as np
from hip_lib import *
j = Job()
one = np.full(1024, 1.0, np.float16).view(np.uint8)
experiment(j, 'hf', 'hf', 'hf', one, one)
# identity-ish: activation impulse at (m=3,k=5) with ones weights
a = np.zeros(1024, np.float16); a[3*32+5] = 2.0
experiment(j, 'hf', 'hf', 'hf', a.view(np.uint8), one)
outs = run_jobs([j])
cyc, O = outs[0]
print('cycles per step', cyc)
for s in range(2):
    t = np.frombuffer(O[s*4096:s*4096+2048], np.float16)
    print(s, 'distinct', np.unique(t)[:10], 'nonzero idx', np.nonzero(t)[0][:40])
