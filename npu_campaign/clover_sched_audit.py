"""Summarize a GGML_SCHED_DEBUG=2 dump: per distinct graph, splits by backend and the ops/names placed on CPU."""
import re, sys, collections
txt = open(sys.argv[1], encoding='utf8', errors='replace').read()
graphs, cur = [], None
for line in txt.splitlines():
    m = re.search(r'## SPLIT #(\d+): (\S+) # (\d+) inputs', line)
    if m:
        if m.group(1) == '0':
            cur = {'splits': [], 'cpu': collections.Counter(), 'cpu_names': collections.Counter(), 'nodes': 0}
            graphs.append(cur)
        cur['splits'].append(m.group(2))
        continue
    m = re.search(r'node #\s*(\d+) \(\s*([^)]+?)\): +(\S+) +\(\s*[^)]*\) \[\s*(\S+)', line)
    if m and cur is not None:
        cur['nodes'] += 1
        if m.group(4) == 'CPU':
            cur['cpu'][m.group(2)] += 1
            cur['cpu_names'][re.sub(r'-\d+$', '-N', m.group(3))] += 1
seen = {}
for g in graphs:
    key = (len(g['splits']), g['nodes'])
    seen.setdefault(key, [g, 0])[1] += 1
for (ns, nn), (g, cnt) in sorted(seen.items(), key=lambda kv: -kv[0][1]):
    c = collections.Counter(g['splits'])
    print(f"graph x{cnt}: nodes={nn} splits={ns} by backend={dict(c)}")
    print("   CPU ops:", dict(g['cpu'].most_common(25)))
    print("   CPU names:", dict(g['cpu_names'].most_common(25)))
