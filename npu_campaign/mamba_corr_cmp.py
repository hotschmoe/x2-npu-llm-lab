import sys, glob, os
d, tag = sys.argv[1], sys.argv[2]
tot = 0; ok = 0
for f in sorted(glob.glob(os.path.join(d, 'p*.cpu.out'))):
    b = os.path.basename(f)[:-8]
    g = os.path.join(d, f'{b}.{tag}.out')
    if not os.path.exists(g): continue
    a = open(f, encoding='utf-8', errors='replace').read().strip()
    c = open(g, encoding='utf-8', errors='replace').read().strip()
    n = min(len(a), len(c)); i = 0
    while i < n and a[i] == c[i]: i += 1
    same = (a == c)
    tot += 1; ok += same
    print(f'{b}: {"IDENTICAL" if same else "DIVERGE at char %d of %d/%d" % (i, len(a), len(c))}')
    if not same: print('   cpu:', repr(a[max(0,i-20):i+60])); print('   npu:', repr(c[max(0,i-20):i+60]))
print(f'{ok}/{tot} identical')
