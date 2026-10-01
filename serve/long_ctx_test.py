"""Needle-in-haystack + speed check at a given prompt size against a running server: python serve/long_ctx_test.py <port> <target_tokens>"""
import json, sys, time, glob, urllib.request
port, target = int(sys.argv[1]), int(sys.argv[2])
base = f'http://127.0.0.1:{port}'
def post(path, obj, timeout=1800):
    req = urllib.request.Request(base + path, json.dumps(obj).encode(), {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=timeout))
def ntok(text): return len(post('/tokenize', {'content': text})['tokens'])
files = sorted(glob.glob('*.md') + glob.glob('npu_campaign/*.md'))
chunks, total, i = [], 0, 0
needle = 'IMPORTANT FACT: the secret launch code is MANGO-4417.\n\n'
while total < target:
    t = open(files[i % len(files)], encoding='utf-8', errors='replace').read()
    seg = t[:60000]
    chunks.append(f'--- document {i}: {files[i % len(files)]} ---\n' + seg)
    total += ntok(seg); i += 1
body = needle + '\n\n'.join(chunks)
body = body[:int(len(body) * target / max(total, 1))] if total > target * 1.05 else body
prompt = body + '\n\nQuestion: what is the secret launch code mentioned at the very beginning of this text? Answer with just the code.'
n = ntok(prompt)
t0 = time.time()
r = post('/v1/chat/completions', {'model': 'qwen36-35b-a3b', 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 1500, 'temperature': 0, 'cache_prompt': False}, timeout=3000)
dt = time.time() - t0
m = r['choices'][0]['message']; tm = r['timings']
ans = (m.get('content') or '') + ' || reasoning_tail: ' + (m.get('reasoning_content') or '')[-120:].replace('\n', ' ')
print(f"prompt_tokens={tm['prompt_n']} prompt_tps={tm['prompt_per_second']:.0f} gen_n={tm['predicted_n']} gen_tps={tm['predicted_per_second']:.1f} wall={dt:.0f}s")
print('FOUND NEEDLE' if 'MANGO-4417' in ans else 'NEEDLE NOT FOUND', '::', ans[:200])
