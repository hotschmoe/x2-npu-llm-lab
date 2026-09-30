#!/usr/bin/env python
"""Sanity probe of a model config: start llama-server, send a short and a long chat prompt (temp 0), print the replies.
Used to tell 'model/NPU numerics produce garbage' apart from 'PI/template problem'.
  python probe_server.py --model gemma --cpu                     (CPU only, no NPU lock)
  python probe_server.py --model gemma --env GGML_HEXAGON_X=1 --ub 128 --b 512 [--words 1200]
"""
import argparse, json, os, sys, time, subprocess, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_suite as rs

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='gemma'); ap.add_argument('--cpu', action='store_true')
ap.add_argument('--env', action='append', default=[]); ap.add_argument('--noenv', action='store_true', help='drop the model default env')
ap.add_argument('--ub', type=int, default=1024); ap.add_argument('--b', type=int, default=2048)
ap.add_argument('--words', type=int, default=900); ap.add_argument('--port', type=int, default=8096)
ap.add_argument('--pkg', default=''); ap.add_argument('--extra', nargs=argparse.REMAINDER, default=[])
a = ap.parse_args()
if a.pkg: rs.PKG = os.path.abspath(os.path.join(rs.ROOT, a.pkg))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
m = rs.MODELS[a.model]
env = dict(os.environ)
for k in rs.CLEAR_ENV: env.pop(k, None)
if not a.noenv: env.update(m['env'])
for kv in a.env: k, v = kv.split('=', 1); env[k] = v
env['ADSP_LIBRARY_PATH'] = os.path.join(rs.PKG, 'lib') + ';'
cmd = [os.path.join(rs.PKG, 'bin', 'llama-server.exe'), '-m', os.path.join(rs.ROOT, m['gguf']), '--host', '127.0.0.1', '--port', str(a.port), '--alias', 'x',
       '-c', '8192', '-b', str(a.b), '-ub', str(a.ub), '-t', '4', '-np', '1', '--jinja', '--temp', '0'] + \
      (['--device', 'none', '-ngl', '0'] if a.cpu else ['--device', 'HTP0', '-ngl', '99', '-fit', 'off', '--load-mode', 'none']) + a.extra
held = False if a.cpu else rs.acquire_lock('pitest probe ' + a.model)
proc = None
try:
    logp = os.path.join(rs.HERE, 'logs', f'probe-{a.model}.log')
    proc = subprocess.Popen(cmd, env=env, stdout=open(logp, 'ab'), stderr=subprocess.STDOUT, creationflags=0x08000000)
    for _ in range(400):
        time.sleep(2)
        if proc.poll() is not None: sys.exit('server died')
        try:
            if json.load(urllib.request.urlopen(f'http://127.0.0.1:{a.port}/health', timeout=3)).get('status') == 'ok': break
        except Exception: pass
    words = [w for w in open(os.path.join(rs.ROOT, 'PREFILL_CAMPAIGN.md'), encoding='utf-8', errors='replace').read().split() if w.isalpha() and len(w) > 2][:a.words]
    prompts = {'short': 'Say hello in exactly three words.',
               'long': ' '.join(words) + '\n\nQuestion: In one sentence, what is the text above about?'}
    for name, p in prompts.items():
        body = json.dumps({'model': 'x', 'messages': [{'role': 'user', 'content': p}], 'max_tokens': 60, 'temperature': 0}).encode()
        t0 = time.time()
        r = json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{a.port}/v1/chat/completions', body, {'Content-Type': 'application/json'}), timeout=900))
        msg = r['choices'][0]['message']
        print(f'[{name}] prompt_tokens={r["usage"]["prompt_tokens"]} t={time.time()-t0:.1f}s content={msg.get("content")!r} reasoning={str(msg.get("reasoning_content"))[:80]!r}', flush=True)
finally:
    rs.stop_server(proc)
    rs.release_lock(held)
