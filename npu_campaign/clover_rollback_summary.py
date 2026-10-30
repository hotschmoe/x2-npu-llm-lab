"""Summarize clover-rollback-ab.ps1 runs: per (config, prompt, width) mean over rounds of tok/s and the spec prof breakdown.

usage: python clover_rollback_summary.py TAG [TAG ...]      (reads npu_campaign/clover-logs/rb-<TAG>-*.err)
"""
import glob
import os
import re
import statistics
import sys
from collections import defaultdict

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clover-logs')


def parse(path):
    t = open(path, encoding='utf-8', errors='replace').read()
    r = {}
    m = re.search(r'decoded\s+(\d+) tokens in\s+([\d.]+) seconds, speed:\s+([\d.]+)', t)
    if not m:
        return None
    r['tok'] = int(m.group(1))
    r['tps'] = float(m.group(3))
    m = re.search(r'forwards: draft (\d+), verify (\d+), re-verify (\d+)(?:, plain (\d+))? -> ([\d.]+) forwards/token', t)
    if m:
        r['nd'], r['nv'], r['nr'] = int(m.group(1)), int(m.group(2)), int(m.group(3))
        r['np'] = int(m.group(4) or 0)
        r['fpt'] = float(m.group(5))
    m = re.search(r'draft [\d.]+ ms \(([\d.]+)/call\), verify [\d.]+ ms \(([\d.]+)/call\), re-verify [\d.]+ ms \(([\d.]+)/call\)', t)
    if m:
        r['D'], r['V'], r['R'] = float(m.group(1)), float(m.group(2)), float(m.group(3))
    m = re.search(r'plain 1-token decode [\d.]+ ms \(([\d.]+)/call\)', t)
    r['P'] = float(m.group(1)) if m else 0.0
    m = re.search(r'ckpt save ([\d.]+) ms, ckpt load ([\d.]+) ms, sample ([\d.]+) ms', t)
    if m:
        r['ckpt'] = float(m.group(1)) + float(m.group(2))
        r['smp'] = float(m.group(3))
    m = re.search(r'tokens/step ([\d.]+)', t)
    r['tps_step'] = float(m.group(1)) if m else float('nan')
    m = re.search(r'#mean acc len = ([\d.]+)', t)
    r['acc'] = float(m.group(1)) if m else float('nan')
    m = re.search(r'argmax = ([\d.]+) ms', t)
    r['argmax'] = float(m.group(1)) if m else float('nan')
    return r


def main():
    groups = defaultdict(list)
    for tag in sys.argv[1:]:
        for p in sorted(glob.glob(os.path.join(LOGS, f'rb-{tag}-*.err'))):
            name = os.path.basename(p)[:-4]
            key = re.sub(r'-r\d+$', '', name[len(f'rb-{tag}-'):])
            r = parse(p)
            if r:
                groups[(tag, key)].append(r)
    cols = ['tps', 'tps_step', 'fpt', 'acc', 'D', 'V', 'R', 'P', 'ckpt', 'smp', 'argmax']
    print(f"{'tag':6} {'config':34} {'n':>2} " + ' '.join(f'{c:>8}' for c in cols) + '  tok/s per round')
    for (tag, key), rs in sorted(groups.items()):
        vals = []
        for c in cols:
            xs = [r[c] for r in rs if c in r]
            vals.append(statistics.mean(xs) if xs else float('nan'))
        per = ' '.join(f"{r['tps']:.2f}" for r in rs)
        print(f'{tag:6} {key:34} {len(rs):2} ' + ' '.join(f'{v:8.2f}' for v in vals) + f'  [{per}]')


if __name__ == '__main__':
    main()
