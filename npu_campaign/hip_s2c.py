# readout / cvt / scale-table exploration after an int4 MAC (act ones, weights 0x11 -> uh21 reads 16)
import numpy as np
from hip_lib import *
from hip_scan import *


def set_table(job, words32, second=None):
    tab = np.zeros(256, np.uint8)
    tab[:128] = np.array(words32, np.uint32).view(np.uint8)
    if second is not None:
        tab[128:] = np.array(second, np.uint32).view(np.uint8)
    job.A[SCALE_OFF:SCALE_OFF + 256] = tab


tables = {
    'hf1_lo': np.full(32, 0x3c00, np.uint32),
    'hf1_both': np.full(32, 0x3c003c00, np.uint32),
    'int1': np.full(32, 1, np.uint32),
    'int256': np.full(32, 256, np.uint32),
    'int0x4000': np.full(32, 0x4000, np.uint32),
    'int0x7fff': np.full(32, 0x7fff, np.uint32),
    'hi_hf1': np.full(32, 0x3c000000, np.uint32),
    'f32_1': np.full(32, 0x3f800000, np.uint32),
    'zeros': np.zeros(32, np.uint32),
}
reads = ['hf', 'ub', 'satub', 'cmub', 'poshf', 'uh21', 'uh22', 'satuh21']
a = np.full(2048, 1, np.uint8)
w = np.full(2048, 0x11, np.uint8)
jobs, meta = [], []
for tn, tw in tables.items():
    j = Job()
    set_table(j, tw)
    slots = mac_reads(j, 'ub', 'n', a, w, reads)
    jobs.append(j)
    meta.append((tn, slots))
for tn, sec in (('bias_i16_5', np.full(32, 5, np.uint32)), ('bias_hf1', np.full(32, 0x3c00, np.uint32)), ('bias_f32_1', np.full(32, 0x3f800000, np.uint32))):
    j = Job()
    set_table(j, tables['hf1_lo'], sec)
    slots = mac_reads(j, 'ub', 'n', a, w, reads)
    jobs.append(j)
    meta.append((tn, slots))
# cvt paths: cvt.X = acc(Rs) then store
cvt_jobs = []
for kind, cname in ((0, 'hf'), (1, 'ub'), (2, 'ub_sc0'), (3, 'ub_sc1'), (4, 'uh21'), (5, 'uh22')):
    for rs in (0x3c00, 0x3f800000, 1, 0x10000, 0):
        j = Job()
        mac_reads(j, 'ub', 'n', a, w, [])
        j.step(CVT, kind, rs, 0)
        for st in (0, 1, 2):
            slot = j.obytes // 4096
            j.step(STCVT, st, slot * 4096, 0)
            j.obytes = (slot + 1) * 4096
        cvt_jobs.append((cname, rs, j))


def summ(raw, rk):
    v = decode(raw, rk)
    u = np.unique(v)
    return 'nz=%d vals=%s' % (int((v != 0).sum()), u[:5].tolist())


outs = run_jobs(jobs, isolate=True)
for (tn, slots), o in zip(meta, outs):
    if o is None:
        print(tn, 'NO OUT')
        continue
    print('table %-10s' % tn, ' | '.join('%s:%s' % (rk, summ(o[1][s * 4096:(s + 1) * 4096], rk)) for rk, s in zip(reads, slots)))
outs2 = run_jobs([j for _, _, j in cvt_jobs], isolate=True)
for (cname, rs, j), o in zip(cvt_jobs, outs2):
    if o is None:
        print('cvt', cname, hex(rs), 'NO OUT')
        continue
    parts = []
    for st in range(3):
        raw = o[1][st * 4096:(st + 1) * 4096]
        for lab, dt in (('u16', np.uint16), ('hf', np.float16), ('u8', np.uint8)):
            v = np.frombuffer(raw[:2048], dt).astype(np.float32)
            u = np.unique(v)
            if (v != 0).any():
                parts.append('st%d/%s: nz=%d %s' % (st, lab, int((v != 0).sum()), u[:4].tolist()))
    print('cvt.%s Rs=%s:' % (cname, hex(rs)), ' ; '.join(parts) if parts else 'all zero')
