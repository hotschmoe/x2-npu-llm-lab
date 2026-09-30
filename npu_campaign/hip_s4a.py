# epilogue / dequant / HMX chain cost for one Qwen expert matmul tile-set (M=32, N=512, K=2048)
import struct, numpy as np, os, sys
from hip_lib import *
from hip_hvx import run_raw


def epi_job(reps):
    hdr = np.zeros(1024, np.int32)
    hdr[0] = MAGIC
    hdr[1] = -2
    hdr[2] = reps
    hdr[4] = 64
    w = np.zeros(4096, np.uint8).tobytes()
    return struct.pack('<I', len(w)) + w + hdr.tobytes() + bytes(4096)


if __name__ == '__main__':
    outs = run_raw([epi_job(7)] * 3)
    FREQ = 2.112
    for o in outs:
        if o is None:
            print('NO OUT')
            continue
        r = np.frombuffer(o[:64], np.uint32)
        names = [''][:0] + ['Q4_0->fp16 dequant (fork kernel), 8 thr', 'epilogue v1 hf-acc, col scale', 'epilogue v2 hf-acc, col+row scale',
                 'epilogue v3 fp32-acc, col scale', 'epilogue v4 fp32-acc, col+row scale', 'epilogue v5 fixed-point s16xs16->s32 vmpyacc']
        print('nthreads', r[6])
        for i, n in enumerate(names):
            print('  %-45s %8.1f us' % (n, r[i] / 1000.0))
        hm = ['HMX fp16 chain (1024 pairs, 16 readouts)', 'HMX int4 MAC+readout per (n,k) tile', 'HMX int4 clracc+MAC+readout per tile', 'HMX int4 accumulate 64 k-tiles, 16 readouts']
        for i, n in enumerate(hm):
            print('  %-45s %8.1f us  (%.2f cyc/pair)' % (n, r[8 + i] / FREQ / 1000.0, r[8 + i] / 1024.0))
