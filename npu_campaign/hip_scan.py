# Impulse scans: which output elements respond to which activation byte / weight element (bilinear structure decode)
import numpy as np
from hip_lib import *

ONES_BYTE = dict(b=0x01, n=0x11, n2x=0x11, c=0x55, sc=0x55, sm=0x11, ubit=0xFF, sbit=0xFF, hf=None, f8=0x38)

def pack_elem(wk, index, value, nbytes=2048):
    """weight tile (nbytes) with a single element (byte/nibble/crumb/bit by mode) at element `index` = value, everything else 0."""
    w = np.zeros(nbytes, np.uint8)
    bits = dict(b=8, n=4, n2x=4, c=2, sc=2, sm=4, ubit=1, sbit=1)[wk]
    per = 8 // bits
    byte, sub = divmod(index, per)
    w[byte] = (value & ((1 << bits) - 1)) << (sub * bits)
    return w

def run_batches(exps, isolate=False, per_job=28):
    """exps: list of (ak, wk, rk, act_bytes, wt_bytes, kwargs). returns list of raw output bytes."""
    jobs, idx = [], []
    for s in range(0, len(exps), per_job):
        j = Job(); slots = []
        for (ak, wk, rk, a, w, kw) in exps[s:s + per_job]:
            slots.append(experiment(j, ak, wk, rk, a, w, **kw))
        jobs.append((j, slots))
    outs = run_jobs([j for j, _ in jobs], isolate=isolate)
    res = []
    for (j, slots), o in zip(jobs, outs):
        for sl in slots:
            res.append(None if o is None else o[1][sl * 4096:(sl + 1) * 4096])
    return res
