# Driver library for the HMX low-bit probe (agent int4u8). See llama.cpp-hexagon-int4u8/ggml/src/ggml-hexagon/htp/hmx-int-probe.h
import os, struct, subprocess, sys, tempfile
import numpy as np

ROOT = r'C:\Users\hotschmoe\Documents\llm_inference'
PKG = os.environ.get('HIP_PKG', os.path.join(ROOT, 'npu_campaign', 'pkg-int4u8-probe6'))
EXE = os.path.join(PKG, 'bin', 'test-hmx-int-probe.exe')
LOCKRUN = os.path.join(ROOT, 'npu_campaign', 'int4u8_lockrun.py')
MAGIC = 0x50495848

# op codes (keep in sync with the .h)
CLR, ACT, WEI, READ, BIAS, SWAP, MAC, BENCH, SYNC, SHL, MACSEQ, MEMBIAS, FILLO, MACX, ARGS, READB, BENCH2, CVT, STCVT, BIAS1 = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20

ACT_KINDS = dict(ub=0, hf=1, ub_deep=2, hf_deep=3, f8=4, ub_cm=5, ub_deep_cm=6, ub_single=7, hf_single=8, ub_above=9, f8_deep=10)
WEI_KINDS = dict(b=0, ubit=1, c=2, hf=3, n=4, sbit=5, sc=6, sm=7, n2x=8, f8=9, b_after=10, n_after=11, hf_after=12, b_single=13, n_single=14,
                 hf_single=15, b_deep=16, n_deep=17, b_drop=18, n_drop=19, sc_after=20, c_after=21, sm_after=22, ubit_after=23, n2x_after=24,
                 n2x_single=25, sbit_deep=26, b_dilate=27)
READ_KINDS = dict(uh21=0, uh22=1, satuh21=2, hf=3, ub=4, satub=5, cmub=6, satuh22=7, poshf=8, cmsatub=9, retuh21=10, rethf=11, bfuh21=12, bfhf=13,
                  bfub=14, bfuh22=15)

A_BYTES = 128 * 1024
W_BYTES = 128 * 1024
O_BYTES = 128 * 1024
SCALE_OFF = A_BYTES - 4096   # scale table lives at the end of A


def scale_table(scale_bits=0x3c00):
    t = np.zeros(256, np.uint8)
    w = np.full(32, scale_bits, np.uint32)
    t[:128] = w.view(np.uint8)
    return t


class Job:
    def __init__(self):
        self.steps = []
        self.A = np.zeros(A_BYTES, np.uint8)
        self.W = np.zeros(W_BYTES, np.uint8)
        self.obytes = 0
        self.A[SCALE_OFF:SCALE_OFF + 256] = scale_table()
        self.steps.append((BIAS, 0, 0, 0))

    def step(self, op, a=0, b=0, c=0):
        self.steps.append((op, a, b, c))

    def pack(self):
        hdr = np.zeros(1024, np.int32)
        hdr[0] = MAGIC
        hdr[1] = len(self.steps)
        hdr[2] = int(np.max(np.nonzero(self.W)[0]) + 1 if self.W.any() else 4096)
        hdr[2] = W_BYTES
        hdr[3] = A_BYTES
        hdr[4] = self.obytes
        hdr[5] = SCALE_OFF
        hdr[6] = 0
        for i, s in enumerate(self.steps):
            hdr[16 + 4 * i:16 + 4 * i + 4] = s
        wb = self.W.tobytes()
        src1 = hdr.tobytes() + self.A.tobytes()
        return struct.pack('<I', len(wb)) + wb + src1


def run_jobs(jobs, tag='j', isolate=False):
    """Run a list of Job under the NPU lock; returns list of (cycles[nsteps], O bytes)."""
    d = tempfile.mkdtemp(prefix='hip_', dir=os.path.join(ROOT, 'npu_campaign'))
    paths = []
    for i, j in enumerate(jobs):
        p = os.path.join(d, f'{tag}{i:04d}.bin')
        open(p, 'wb').write(j.pack())
        paths.append(p)
    env = dict(os.environ)
    env['ADSP_LIBRARY_PATH'] = os.path.join(PKG, 'lib') + ';'
    env['PATH'] = os.path.join(PKG, 'bin') + os.pathsep + env['PATH']
    lk = [] if os.environ.get('NPU_LOCK_HELD') == '1' else [sys.executable, LOCKRUN, '--']
    if isolate:
        cmd = lk + [sys.executable, os.path.join(ROOT, 'npu_campaign', 'hip_batch.py'), EXE] + paths
    else:
        cmd = lk + [EXE] + paths
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)
    outs = []
    for i, p in enumerate(paths):
        try:
            b = open(p + '.out', 'rb').read()
        except FileNotFoundError:
            outs.append(None)
            continue
        cyc = np.frombuffer(b[:4096], np.uint32)[:len(jobs[i].steps)]
        outs.append((cyc, b[4096:]))
    if isolate: print(r.stdout[-3000:])
    if any(o is None for o in outs):
        print('STDOUT', r.stdout[-2000:]); print('STDERR', r.stderr[-2000:])
    for p in paths:
        for q in (p, p + '.out'):
            if os.path.exists(q): os.remove(q)
    try: os.rmdir(d)
    except OSError: pass
    return outs


def experiment(job, actk, weik, readk, act_tile, wt_tile, clr=None, rt_a=2047, rt_w=2047, rt_r=0, slot=None, tile_bytes=2048, out_bytes=4096):
    """Append one clr/mac/read triple. Returns the slot index whose output lives at O[slot*out_bytes:]."""
    if slot is None:
        slot = job.obytes // out_bytes
    ai = len([s for s in job.steps if s[0] == MAC or s[0] == MACX])
    aoff = ai * tile_bytes
    woff = ai * tile_bytes
    ab = np.frombuffer(bytes(act_tile), np.uint8)
    wb = np.frombuffer(bytes(wt_tile), np.uint8)
    job.A[aoff:aoff + len(ab)] = ab
    job.W[woff:woff + len(wb)] = wb
    if clr is None:
        clr = 1 if (actk in ('hf', 'hf_deep', 'hf_single')) else 0
    job.step(CLR, clr)
    if rt_a == 2047 and rt_w == 2047:
        job.step(MAC, ACT_KINDS[actk] | (WEI_KINDS[weik] << 8), aoff, woff)
    else:
        job.step(MACX, ACT_KINDS[actk] | (WEI_KINDS[weik] << 8), aoff, woff)
        job.step(ARGS, rt_a, rt_w, 0)
    job.step(READ, READ_KINDS[readk], slot * out_bytes, rt_r)
    job.obytes = max(job.obytes, (slot + 1) * out_bytes)
    return slot


def mac_reads(job, actk, weik, act_tile, wt_tile, reads, clr=None, rt_a=2047, rt_w=2047, rt_r=0):
    """One clr+mac followed by several readouts of the same accumulator; returns list of slots."""
    ai = len([s for s in job.steps if s[0] in (MAC, MACX)])
    aoff = woff = ai * 2048
    ab = np.frombuffer(bytes(act_tile), np.uint8); wb = np.frombuffer(bytes(wt_tile), np.uint8)
    job.A[aoff:aoff + len(ab)] = ab; job.W[woff:woff + len(wb)] = wb
    if clr is None:
        clr = 1 if actk in ('hf', 'hf_deep', 'hf_single') else 0
    job.step(CLR, clr)
    if rt_a == 2047 and rt_w == 2047:
        job.step(MAC, ACT_KINDS[actk] | (WEI_KINDS[weik] << 8), aoff, woff)
    else:
        job.step(MACX, ACT_KINDS[actk] | (WEI_KINDS[weik] << 8), aoff, woff)
        job.step(ARGS, rt_a, rt_w, 0)
    slots = []
    for rk in reads:
        slot = job.obytes // 4096
        job.step(READ, READ_KINDS[rk], slot * 4096, rt_r)
        job.obytes = (slot + 1) * 4096
        slots.append(slot)
    return slots


def decode(raw, rk):
    if rk in ('hf', 'poshf', 'rethf', 'bfhf'):
        return np.frombuffer(raw[:2048], np.float16).astype(np.float32)
    if rk in ('uh21', 'uh22', 'satuh21', 'satuh22', 'retuh21', 'bfuh21', 'bfuh22'):
        return np.frombuffer(raw[:2048], np.uint16).astype(np.float32)
    return np.frombuffer(raw[:1024], np.uint8).astype(np.float32)
