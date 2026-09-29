import re,sys,collections
f=sys.argv[1]; tot=collections.defaultdict(lambda:[0,0]); tot2=collections.defaultdict(lambda:[0,0]); batches=[]
for l in open(f,errors='replace'):
    if 'profile-op OPBATCH' in l:
        m=re.search(r'n-ops (\d+).*usec (\d+)',l); batches.append((int(m[1]),int(m[2]))); continue
    m=re.search(r'profile-op (\S+)\|(.*?)\|(.*?)\|(.*?)\|(.*?)\|(.*?)\|usec (\d+)',l)
    if not m: continue
    op,names,shp,ty,_,mode,us=m.groups(); us=int(us)
    t=tot[op]; t[0]+=us; t[1]+=1
    if op.startswith('MUL_MAT') or op in('FLASH_ATTN_EXT','GATED_DELTA_NET'):
        k=(op,names.split(' x ')[0].split('.')[-2] if 'blk.' in names.split(' x ')[0] else names.split(' x ')[0][:30],shp,mode.split(' ')[0]); t=tot2[k]; t[0]+=us; t[1]+=1
print('batches',batches)
T=sum(v[0] for v in tot.values()); print('sum op usec',T)
for k,v in sorted(tot.items(),key=lambda x:-x[1][0])[:22]: print(f'{k:22s} {v[0]/1e3:9.1f} ms {100*v[0]/T:5.1f}% n={v[1]}')
print('--- matmul-ish by shape')
for k,v in sorted(tot2.items(),key=lambda x:-x[1][0])[:25]: print(f'{v[0]/1e3:8.1f} ms n={v[1]:3d} {k}')
