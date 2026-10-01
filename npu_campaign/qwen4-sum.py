import re,collections
d=collections.defaultdict(list)
for l in open('qwen4-m8.out'):
    m=re.match(r'q4-m8-(\w+)-(\w+)-(base|s14)-r(\d) tps=([\d.]+) acc=(\S*)',l)
    if m: d[(m[1],m[2],m[3])].append((float(m[5]),m[6]))
for pr in ('Hello','CODE'):
  for cfg in ('plain','mtp','pm'):
    b=[x[0] for x in d[(pr,cfg,'base')]]; c=[x[0] for x in d[(pr,cfg,'s14')]]
    if b and c:
      mb=sum(b)/len(b); mc=sum(c)/len(c)
      print(f"{pr:5} {cfg:5} base {mb:.2f} {b} | s14 {mc:.2f} {c} | {100*(mc/mb-1):+.1f}%  acc {d[(pr,cfg,'s14')][0][1]}")
