import re, sys, collections
txt = open(sys.argv[1], 'rb').read().decode('latin1')
seqs = []; cur = None
recs = collections.defaultdict(dict)
for l in txt.split('\n'):
    if 'profile-op OPBATCH' in l: cur = []; seqs.append(cur); continue
    if 'profile-op ' in l and cur is not None:
        p = l.split('profile-op ', 1)[1].split('|'); cur.append((p[0], p[1], p[2], p[3] if len(p) > 3 else ''))
    m = re.search(r'ophash b(\d+) op(\d+) (\S+) ([0-9a-f]{8}) ([0-9a-f]{8}) ([0-9a-f]{8}) ([0-9a-f]{8}) ([0-9a-f]{8}) ([0-9a-f]{8}) ([0-9a-f]{8})', l)
    if m: recs[int(m.group(1))][int(m.group(2))] = (m.group(3), int(m.group(4), 16), int(m.group(5), 16), int(m.group(6), 16), [int(m.group(j), 16) for j in (7, 8, 9, 10)])
order = sorted(recs)
print(len(order), 'batches with records;', len(seqs), 'profile batches')
agg = collections.Counter(); aggt = collections.Counter(); ex = {}
for bi, k in enumerate(order):
    for i, (nm, gap, trk, chk, ext) in sorted(recs[k].items()):
        info = seqs[bi][i] if bi < len(seqs) and i < len(seqs[bi]) else None
        srcs = info[1].split(' -> ')[0].split(' x ') if info else []
        dims = info[2].split(' -> ')[0].split(' x ') if info else []
        for s in range(10):
            if gap >> s & 1:
                key = (nm, s, dims[s] if s < len(dims) else '?', srcs[s][:40] if s < len(srcs) else '?')
                agg[key] += 1; ex.setdefault(key, (bi, i)); print(f'   GAP batch#{bi} op{i} {nm} slot{s} {key[3]} first_diff_off={ext[0]} cached={ext[1]:08x} ddr={ext[2]:08x} ndiff={ext[3]>>16} last_diff_off={(ext[3]&0xffff)*4} dims={key[2]}')
            if trk >> s & 1:
                aggt[(nm, s)] += 1
print('GAP (mismatch, tracker did NOT know) by (op, src slot, dims, name):')
for key, c in agg.most_common(60): print(f'  {c:5d} {key}  first at batch#{ex[key][0]} op{ex[key][1]}')
print('TRACKED mismatches (tracker knew, expected) by (op, slot): total', sum(aggt.values()))
for key, c in aggt.most_common(12): print(f'  {c:5d} {key}')
