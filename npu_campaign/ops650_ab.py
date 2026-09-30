"""ops650: summarise untraced ABBA runs. usage: ops650_ab.py PREFIX   (e.g. ops650-ab2)  -> mean tok/s per variant and the per-round runs.
Also prints the DSP/host batch timing lines if present (GGML_HEXAGON_TIME_BATCH / LLAMA_DECODE_TIMING)."""
import re, sys, glob, os, statistics as st
from collections import defaultdict
pre = sys.argv[1]
res = defaultdict(list)
for f in sorted(glob.glob(f'mtp-speed/{pre}-*.detail.log')):
    m = re.match(rf'mtp-speed[\/]{re.escape(pre)}-(.+)-r(\d+)\.detail\.log', f)
    if not m: continue
    txt = open(f, errors='replace').read()
    t = re.findall(r'(?<!prompt )eval time =\s+([\d.]+) ms /\s+(\d+) tokens[^\n]*?([\d.]+) tokens per second', txt)
    if not t: continue
    tps = float(t[-1][2])
    dsp = re.findall(r'batch-timing n-ops (\d+) n (\d+) dsp ([\d.]+) ms host-observed ([\d.]+) ms', txt)
    dt = re.findall(r'decode-timing tgt1 n=(\d+) rebuilt=(\d+) ms/call: prep ([\d.]+) apply ([\d.]+) build ([\d.]+) setin ([\d.]+) enqueue ([\d.]+) outputs\+wait ([\d.]+) tail ([\d.]+)', txt)
    acc = re.findall(r'draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated', txt)
    res[m.group(1)].append((int(m.group(2)), tps, dsp[-1] if dsp else None, dt[-1] if dt else None, acc[-1] if acc else None))
for v, rows in sorted(res.items()):
    tps = [r[1] for r in rows]
    print(f'{v:14s} n={len(tps)} mean {st.mean(tps):6.2f} tok/s  runs(round:tps) ' + ' '.join(f'r{r[0]}:{r[1]:.2f}' for r in sorted(rows)))
    for r in sorted(rows):
        extra = []
        if r[2]: extra.append(f'batch n-ops {r[2][0]} dsp {r[2][2]} ms host-observed {r[2][3]} ms')
        if r[3]: extra.append(f'host ms/call: prep {r[3][2]} build {r[3][4]} enqueue {r[3][6]} wait {r[3][7]}')
        if r[4]: extra.append(f'accept {r[4][1]}/{r[4][2]}')
        if extra: print('     r%d: ' % r[0] + ' | '.join(extra))
