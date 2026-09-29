import re, sys
log = sys.argv[1]; bidx = int(sys.argv[2]); opidx = int(sys.argv[3]); off_in = int(sys.argv[4]); span = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
txt = open(log, 'rb').read().decode('latin1').split('\n')
batches = []; cur = None
for l in txt:
    m = re.search(r'add-tensor #(\d+) (.*?) : bi (\d+) data ([0-9A-Fa-f]+) offset (\d+) size (\d+) flags (0x[0-9a-f]+) : (\S+)', l)
    if m:
        if int(m.group(1)) == 0: cur = {}; batches.append(cur)
        cur[m.group(2)] = (int(m.group(5)), int(m.group(6)), int(m.group(3)), m.group(8))
ops = []; curops = None
for l in txt:
    if 'profile-op OPBATCH' in l: curops = []; ops.append(curops); continue
    if 'profile-op ' in l and curops is not None:
        p = l.split('profile-op ', 1)[1].split('|'); curops.append((p[0], p[1], p[2]))
T = batches[bidx]; O = ops[bidx]
name, dims = O[opidx][1], O[opidx][2]
print('op', opidx, O[opidx][0], name[:150])
srcs = name.split(' -> ')[0].split(' x '); 
print('srcs:', srcs)
# find activation tensor: the src whose name contains attn_post_norm/attn_norm/ffn_moe_probs
target = None
for s in srcs:
    if s in T and ('norm' in s or 'probs' in s) and T[s][1] >= off_in: target = s
print('target', target, T.get(target))
base = T[target][0] + off_in
print(f'diff at buffer offset {base} (line {base//128}, byte-in-line {base%128})')
lo, hi = base - span, base + span
print('tensors within +-%d of that offset (bi=%d):' % (span, T[target][2]))
for n, (o, sz, bi, d) in sorted(T.items(), key=lambda kv: kv[1][0]):
    if bi == T[target][2] and o < hi and o + sz > lo:
        # ops that touch this tensor by name
        ws = [i for i, (op, nm, dm) in enumerate(O) if nm.split(' -> ')[-1].strip() == n]
        rs = [i for i, (op, nm, dm) in enumerate(O) if n in [x.strip() for x in nm.split(' -> ')[0].split(' x ')]]
        print(f'  [{o:9d},{o+sz:9d}) size {sz:7d} align128={o%128==0} {n[:60]:60s} dims {d} written_by {ws[:3]} read_by {rs[:6]}')
