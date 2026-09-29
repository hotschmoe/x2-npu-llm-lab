import re,sys,collections
# per-op start cycles + usec (mhz field) -> inter-op gaps within each OPBATCH
f=sys.argv[1]; batches=[]; cur=None
for l in open(f,errors='replace'):
    m=re.search(r'profile-op OPBATCH\|.*?usec (\d+) cycles (\d+) start (\d+) mhz ([\d.]+)',l)
    if m:
        cur={'usec':int(m[1]),'cyc':int(m[2]),'start':int(m[3]),'ops':[]}; batches.append(cur); continue
    m=re.search(r'profile-op (\S+)\|(.*?)\|.*?\|usec (\d+) cycles (\d+) start (\d+)',l)
    if m and cur is not None: cur['ops'].append((m[1],m[2][:60],int(m[3]),int(m[4]),int(m[5])))
for bi,b in enumerate(batches):
    ops=b['ops']; mhz=b['cyc']/b['usec']  # cycles per usec
    opsum=sum(o[3] for o in ops); gaps=[]
    for a,c in zip(ops,ops[1:]):
        gaps.append(((c[4]-(a[4]+a[3]))/mhz, a[0],c[0]))
    tg=sum(g[0] for g in gaps if g[0]>0)
    print(f'batch{bi}: ops={len(ops)} wall={b["usec"]}us opsum={opsum/mhz:.0f}us gapsum={tg:.0f}us first_gap={(ops[0][4]-b["start"])/mhz:.0f}us tail={(b["start"]+b["cyc"]-(ops[-1][4]+ops[-1][3]))/mhz:.0f}us')
    by=collections.defaultdict(lambda:[0,0])
    for g,a,c in gaps:
        if g>0: by[(a,c)][0]+=g; by[(a,c)][1]+=1
    for k,v in sorted(by.items(),key=lambda x:-x[1][0])[:8]: print(f'   gap {k[0]:>18s}->{k[1]:<18s} {v[0]:8.0f}us n={v[1]} avg={v[0]/v[1]:.1f}us')
    small=collections.Counter()
    for o in ops:
        u=o[3]/mhz
        small['<10us' if u<10 else '<50us' if u<50 else '<200us' if u<200 else '>=200us']+=1
    print('   op-duration histogram',dict(small))
