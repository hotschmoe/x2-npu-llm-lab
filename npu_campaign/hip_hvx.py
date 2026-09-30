import struct, numpy as np, sys
from hip_lib import *
def hvx_job(kernel, iters, nthreads):
    hdr = np.zeros(1024, np.int32)
    hdr[0] = MAGIC; hdr[1] = -1; hdr[2] = iters; hdr[3] = nthreads; hdr[4] = kernel
    w = np.zeros(4096, np.uint8).tobytes()
    return struct.pack('<I', len(w)) + w + hdr.tobytes() + bytes(4096)
import os, tempfile, subprocess
def run_raw(blobs):
    d = tempfile.mkdtemp(prefix='hipx_', dir=os.path.join(ROOT, 'npu_campaign'))
    paths = []
    for i, b in enumerate(blobs):
        p = os.path.join(d, f'h{i:03d}.bin'); open(p, 'wb').write(b); paths.append(p)
    env = dict(os.environ); env['ADSP_LIBRARY_PATH'] = os.path.join(PKG, 'lib') + ';'; env['PATH'] = os.path.join(PKG, 'bin') + os.pathsep + env['PATH']
    lk = [] if os.environ.get('NPU_LOCK_HELD') == '1' else [sys.executable, LOCKRUN, '--']
    r = subprocess.run(lk + [sys.executable, os.path.join(ROOT, 'npu_campaign', 'hip_batch.py'), EXE] + paths, env=env, capture_output=True, text=True, timeout=7200)
    print(r.stdout[-1500:], r.stderr[-500:])
    res = []
    for p in paths:
        try: res.append(open(p + '.out', 'rb').read())
        except FileNotFoundError: res.append(None)
    for p in paths:
        for q in (p, p + '.out'):
            if os.path.exists(q): os.remove(q)
    os.rmdir(d)
    return res
if __name__ == '__main__':
    names = {0: 'vrmpy u8xi8->i32 (128 MAC)', 1: 'vrmpy i8xi8->i32 (128 MAC)', 2: 'qf16 vmpy+vadd (64 MAC)', 3: 'vmpyacc u8xi8->i16 pair (128 MAC)', 4: 'vmpyacc hf x hf -> sf pair (64 MAC)'}
    iters = 2000000
    cfg = [(k, nt) for k in range(5) for nt in (1, 4, 6, 8)]
    outs = run_raw([hvx_job(k, iters, nt) for k, nt in cfg])
    for (k, nt), o in zip(cfg, outs):
        if o is None: print(k, nt, 'NO OUT'); continue
        res = np.frombuffer(o[:160], np.uint32).reshape(10, 4)[:nt]
        cyc = res[:, 0].astype(np.float64); us = res[:, 1].astype(np.float64) / 19.2
        per = {0: 8, 1: 8, 2: 8, 3: 4, 4: 4}[k]      # instructions per loop iteration
        macs_per_inst = {0: 128, 1: 128, 2: 64, 3: 128, 4: 64}[k]
        inst = iters * per
        thr_macs = inst * macs_per_inst / (us * 1e-6)   # per thread MAC/s
        print(f'{names[k]:40s} thr={nt}: cyc/inst {cyc.mean()/inst:.3f}  us {us.mean():.0f}  per-thread {thr_macs.mean()/1e9:.1f} GMAC/s  aggregate {thr_macs.sum()/1e9:.1f} GMAC/s = {2*thr_macs.sum()/1e12:.2f} TOP/s')
