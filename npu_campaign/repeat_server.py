# In-process repeat protocol: one llama-server process, N identical /completion requests (cache_prompt=false),
# per-request logits-row hashes. Tests whether run-to-run variation also occurs WITHIN a process.
# usage: repeat_server.py --pkg npu_campaign\pkg-decode2-base --n 16 [--npredict 3] [--env A=1 --env B=2] [--words 8]
import argparse, os, subprocess, sys, time, json, hashlib, struct, urllib.request, shutil
from collections import defaultdict
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--pkg', required=True); ap.add_argument('--n', type=int, default=16)
ap.add_argument('--npredict', type=int, default=3); ap.add_argument('--env', action='append', default=[])
ap.add_argument('--prompt', default='The quick brown fox jumps over the lazy dog near the river bank today')
ap.add_argument('--label', default='rs'); ap.add_argument('--ub', type=int, default=128)
ap.add_argument('--model', default=r'models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0-mtp-f32.gguf')
ap.add_argument('--profile', action='store_true', help='GGML_HEXAGON_PROFILE=1 + verbose log (ophash diag builds)')
ap.add_argument('--port', type=int, default=8093); ap.add_argument('--keep', action='store_true')
a = ap.parse_args()
pkg = os.path.join(root, a.pkg); out = os.path.join(root, 'npu_campaign', 'mtp-speed'); os.makedirs(out, exist_ok=True)
trace = os.path.join(out, f'{a.label}.logits.bin')
if os.path.exists(trace): os.remove(trace)
env = dict(os.environ)
for k in list(env):
    if k.startswith(('GGML_HEXAGON_', 'LLAMA_')): del env[k]
env.update({'GGML_HEXAGON_OPBATCH': '3072', 'GGML_HEXAGON_REPLAY': '1', 'GGML_HEXAGON_OPFUSION': '1', 'GGML_HEXAGON_VERBOSE': '0',
            'GGML_HEXAGON_PROFILE': '0', 'GGML_HEXAGON_MOE_TOPK_ALIAS': '0', 'GGML_HEXAGON_MM_SELECT': '2', 'GGML_HEXAGON_HEAD_PREFETCH': '0',
            'GGML_HEXAGON_FA_SELECT': '1', 'GGML_HEXAGON_GDN_SELECT': '1', 'GGML_HEXAGON_ROUTER_HVX': '1', 'LLAMA_LOGITS_TRACE': '1',
            'LLAMA_LOGITS_TRACE_FILE': trace, 'ADSP_LIBRARY_PATH': os.path.join(pkg, 'lib') + ';'})
if a.profile: env['GGML_HEXAGON_PROFILE'] = '1'
for kv in a.env:
    k, v = kv.split('=', 1); env[k] = v
lock = os.path.join(root, 'npu_campaign', '.npu.lock')
def enter():
    last = 0
    while True:
        try:
            os.mkdir(lock); open(os.path.join(lock, 'owner.txt'), 'w').write(f'decode2 repeat_server pid={os.getpid()} started={time.ctime()}'); return
        except FileExistsError:
            if time.time() - last > 30:
                last = time.time()
                try:
                    f = os.path.join(lock, 'owner.txt'); pid = int(open(f).read().split('pid=')[1].split()[0])
                    alive = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'], capture_output=True, text=True).stdout.count(str(pid)) > 0
                    if not alive and time.time() - os.path.getmtime(f) > 120: shutil.rmtree(lock, ignore_errors=True); continue
                except Exception: pass
            time.sleep(0.01)
enter()
try:
    cmd = [os.path.join(pkg, 'bin', 'llama-server.exe'), '-m', os.path.join(root, a.model), '--host', '127.0.0.1', '--port', str(a.port),
           '--device', 'HTP0', '-ngl', '99', '-fit', 'off', '--load-mode', 'none', '-c', '1024', '-b', str(a.ub), '-ub', str(a.ub), '-np', '1', '--no-warmup', '-lv', '5' if a.profile else '3']
    logf = open(os.path.join(out, f'{a.label}.server.log'), 'wb')
    p = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    t0 = time.time()
    while True:
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{a.port}/health', timeout=2).read(); break
        except Exception:
            if p.poll() is not None: sys.exit('server died')
            if time.time() - t0 > 600: p.kill(); sys.exit('timeout')
            time.sleep(1)
    print(f'server up after {time.time()-t0:.0f}s', flush=True)
    toks = []
    for i in range(a.n):
        req = urllib.request.Request(f'http://127.0.0.1:{a.port}/completion', data=json.dumps({'prompt': a.prompt, 'n_predict': a.npredict, 'temperature': 0, 'seed': 1, 'cache_prompt': False, 'ignore_eos': True}).encode(), headers={'Content-Type': 'application/json'})
        r = json.loads(urllib.request.urlopen(req, timeout=600).read())
        toks.append(r['content'])
    p.terminate()
    try: p.wait(60)
    except subprocess.TimeoutExpired:
        subprocess.run(['taskkill', '/F', '/T', '/PID', str(p.pid)], capture_output=True); p.wait(60)
finally:
    shutil.rmtree(lock, ignore_errors=True)
print(f'distinct completions: {len(set(toks))}: ' + ' || '.join(sorted(set(t.replace(chr(10),' ') for t in toks)))[:300])
r = subprocess.run([sys.executable, os.path.join(root, 'npu_campaign', 'repeat_analyze.py'), trace, str(a.n)], capture_output=True, text=True)
print(r.stdout + r.stderr[-500:])
rows = None
if a.profile:
    import re
    txt = open(os.path.join(out, f'{a.label}.server.log'), 'rb').read().decode('latin1')
    batches = {}
    order = []
    for m in re.finditer(r'ophash b(\d+) op(\d+) (\S+) ([0-9a-f]{8}) ([0-9a-f]{8}) ([0-9a-f]{8}) ([0-9a-f]{8})', txt):
        b = int(m.group(1))
        if b not in batches: batches[b] = []; order.append(b)
        batches[b].append((int(m.group(2)), m.group(3), m.group(4, 5, 6, 7)))
    print(f'ophash: {len(order)} batches, ops/batch: {[len(batches[b]) for b in order[:8]]}...')
    # batches per request
    nb = len(order) // a.n
    print(f'batches per request {nb}')
    first = None
    for j in range(nb):
        ref = batches[order[j]]
        for i in range(1, a.n):
            cur = batches[order[i * nb + j]]
            if len(cur) != len(ref): print(f'batch slot {j}: request {i} op count {len(cur)} vs {len(ref)}'); continue
            diffs = [k for k in range(len(ref)) if ref[k][2] != cur[k][2]]
            if diffs:
                k = diffs[0]
                print(f'slot {j} req {i} vs 0: {len(diffs)} ops differ; first op{ref[k][0]} {ref[k][1]} ref={ref[k][2]} cur={cur[k][2]}')
