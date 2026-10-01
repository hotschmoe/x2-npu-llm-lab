import re, sys, collections
rows = collections.defaultdict(list)
acc = {}
names = {'A':'plain installed','B':'plain mtp3-e','C':'MTP d1 installed','D':'MTP d1 mtp3-e','E':'MTP d1 + p-min','F':'MTP d2 gated (2nd tok p>=0.8)'}
import itertools
files = sys.argv[1:] or ['m3_p11.out']
lines = itertools.chain.from_iterable(open(f, encoding='utf-8', errors='replace') for f in files)
for l in lines:
    m = re.match(r'm3[zw]-(\w+)-r(\d)-(\w) tps=([\d.]*) acc=(\S*) power=(\w+)', l)
    if not m: continue
    pr, r, arm, t, a, p = m.groups()
    if not t or p != 'AC': continue
    rows[(pr, arm)].append(float(t)); acc[(pr, arm)] = a
for pr in ['code', 'json', 'sky', 'rep']:
    print(f'--- {pr}')
    base = None
    for arm in 'ABCDEF':
        v = rows.get((pr, arm), [])
        if not v: continue
        mean = sum(v) / len(v)
        if arm == 'A': base = mean
        print(f'{arm} {names[arm]:32s} n={len(v)} mean {mean:6.2f}  runs {" ".join("%.2f" % x for x in v)}  acc {acc.get((pr, arm), "")}' + (f'  vs installed plain {100*(mean/base-1):+.1f}%' if base else ''))
