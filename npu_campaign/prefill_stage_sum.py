import re,sys,collections,bisect
# usage: prefill_stage_sum.py log.err [opnames] [batch_index]
f=sys.argv[1]; want=set((sys.argv[2] if len(sys.argv)>2 else 'MUL_MAT_ID,MUL_MAT_ID_NX').split(',')); bidx=int(sys.argv[3]) if len(sys.argv)>3 else 1
W=1<<32; MHZ=2130.0
rop=re.compile(r'profile-op (\S+)\|.*?usec \d+ cycles (\d+) start (\d+)')
rev=re.compile(r'trace-evt (\S+): thread (\d+) info (-?\d+) (start|stop) (\d+)')
b=-1; ops=[]; evraw=collections.defaultdict(list)
for l in open(f,errors='replace'):
    if 'profile-op OPBATCH' in l: b+=1; continue
    if b!=bidx: continue
    if 'trace-evt' in l:
        m=rev.search(l)
        if m: evraw[m[2]].append((m[1],m[3],m[4],int(m[5])))
    elif 'profile-op' in l:
        m=rop.search(l)
        if m: ops.append([int(m[3]),int(m[2]),m[1]])
# unwrap ops
ep=0; last=None
for o in ops:
    if last is not None and o[0]+ep*W < last-W//2: ep+=1
    o[0]+=ep*W; last=o[0]
op0=ops[0][0]
ev=[]
for th,lst in evraw.items():
    ep=0; last=None; pend={}; tl=[]
    for n,i,k,c in lst:
        if last is not None and c+ep*W < last-W//2: ep+=1
        v=c+ep*W; last=v; tl.append((n,i,k,v))
    first=tl[0][3]; shift=round((op0-first)/W)*W
    if first+shift < op0-6*10**7: shift+=W
    for n,i,k,v in tl:
        key=(n,i)
        if k=='start': pend[key]=v+shift
        elif key in pend: ev.append((pend.pop(key),v+shift,n,th))
ev.sort(); starts=[e[0] for e in ev]
agg=collections.defaultdict(lambda:collections.defaultdict(lambda:[0,0])); wall=collections.Counter(); cnt=collections.Counter()
for s,c,n in ops:
    if n not in want: continue
    wall[n]+=c; cnt[n]+=1; e=s+c
    i=bisect.bisect_left(starts,s)
    while i<len(ev) and ev[i][0]<e:
        a=agg[n][ev[i][2]]; a[0]+=min(ev[i][1],e)-ev[i][0]; a[1]+=1; i+=1
for n in sorted(want):
    if not cnt[n]: continue
    w=wall[n]; print(f'{n}: {cnt[n]} ops, wall {w/MHZ/cnt[n]:.0f} us/op')
    for st,(c,k) in sorted(agg[n].items(),key=lambda x:-x[1][0])[:9]:
        print(f'   {st:14s} sum-over-threads {c/MHZ/cnt[n]:8.0f} us/op = {100*c/w:6.1f}% of wall x threads; events/op {k/cnt[n]:6.0f}, avg {c/k/MHZ*1000:7.0f} ns')
