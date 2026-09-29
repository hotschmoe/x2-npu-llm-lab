import re, sys
txt = open(sys.argv[1], 'rb').read().decode('latin1')
b = {}
order = []
for m in re.finditer(r'ophash b(\d+) op(\d+) (\S+) ([0-9a-f]{8}) ([0-9a-f]{8}) ([0-9a-f]{8}) ([0-9a-f]{8})', txt):
    k = int(m.group(1))
    if k not in b: b[k] = []; order.append(k)
    b[k].append((int(m.group(2)), m.group(3), [int(m.group(i), 16) for i in (4, 5, 6, 7)]))
# names of ops with dims from profile-op lines, per batch in order
ops = {}
cur = None
for l in txt.split('\n'):
    if 'profile-op OPBATCH' in l: cur = []; continue
    if 'profile-op ' in l and cur is not None:
        p = l.split('profile-op ', 1)[1].split('|'); cur.append(p[0] + ' ' + p[1][:110] + ' ' + p[2][:60])
        if len(cur) == 1: pass
    # store when new batch begins handled below
print(len(order), 'batches')
# re-parse profile-op sequences per batch
seqs = []; cur = None
for l in txt.split('\n'):
    if 'profile-op OPBATCH' in l: cur = []; seqs.append(cur); continue
    if 'profile-op ' in l and cur is not None:
        p = l.split('profile-op ', 1)[1].split('|'); cur.append(p[0] + ' | ' + p[1][:150] + ' | ' + p[2][:70])
for bi, k in enumerate(order):
    lst = b[k]
    nz = [x for x in lst if any(x[2])]
    print(f'batch {k}: {len(lst)} ops, {len(nz)} ops with non-finite outputs')
    if nz:
        i = nz[0][0]
        print('  first bad op', i, nz[0][1], nz[0][2])
        if bi < len(seqs) and i < len(seqs[bi]):
            for j in range(max(0, i - 3), min(len(seqs[bi]), i + 2)): print('   ', j, seqs[bi][j])
