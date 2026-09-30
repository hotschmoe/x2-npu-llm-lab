#!/usr/bin/env python
"""PI coding-agent tool-calling test suite against a local llama-server on the Hexagon NPU.

Usage (from cmd, repo root):
  python npu_campaign\\pitest\\run_suite.py --model qwen --reps 3
  python npu_campaign\\pitest\\run_suite.py --model gemma --reps 3 --cases create_file,no_tool
  python npu_campaign\\pitest\\run_suite.py --model qwen --server-url http://127.0.0.1:8080   (use an already-running server; no lock, no launch)

Takes the cross-agent NPU lock (same protocol as npu-lock.ps1 / lockrun.py) for the whole time a server it launched is alive.
Outputs (npu_campaign\\pitest\\): runs\\<tag>\\<case>-r<N>.jsonl (PI event stream), logs\\server-<tag>.log, results-<tag>.json / results-<tag>.md
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time, urllib.request, random, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
PKG = os.path.join(ROOT, 'llama.cpp-hexagon', 'pkg-wos')
LOCK = os.path.join(ROOT, 'npu_campaign', '.npu.lock')
HOME = os.path.expanduser('~')
BUN = os.path.join(HOME, '.bun', 'bin', 'bun.exe')
PI_CLI = os.path.join(HOME, '.bun', 'install', 'global', 'node_modules', '@earendil-works', 'pi-coding-agent', 'dist', 'cli.js')

MODELS = {
    'qwen': dict(
        gguf=r'models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0.gguf',
        env={'GGML_HEXAGON_OPBATCH': '3072'},
        sampling=['--temp', '0.6', '--top-p', '0.95', '--top-k', '20', '--min-p', '0.0'],
    ),
    'gemma': dict(
        gguf=r'models\gguf\gemma4-26b-a4b-qat-q4_0\gemma-4-26B_q4_0-it.gguf',
        env={'GGML_HEXAGON_OPBATCH': '3072', 'GGML_HEXAGON_ROUTER_HVX': '1'},
        sampling=['--temp', '1.0', '--top-p', '0.95', '--top-k', '64'],
    ),
    'nemotron': dict(
        gguf=r'models\gguf\nemotron35-lightning-30b-a3b-q4_0\NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf',
        env={'GGML_HEXAGON_OPBATCH': '3072', 'GGML_HEXAGON_REPLAY': '1', 'GGML_HEXAGON_OPFUSION': '1', 'GGML_HEXAGON_ROUTER_HVX': '0',
             'GGML_HEXAGON_MM_SELECT': '2', 'GGML_HEXAGON_FA_SELECT': '2', 'GGML_HEXAGON_GDN_SELECT': '2'},
        sampling=['--temp', '0.6', '--top-p', '0.95'],
    ),
}
CLEAR_ENV = ['LLAMA_LOGITS_TRACE', 'LLAMA_LOGITS_TRACE_FILE', 'LLAMA_TENSOR_TRACE_FILE', 'LLAMA_TENSOR_TRACE_FILTER', 'LLAMA_RETAIN_TRACE_FILE',
             'LLAMA_RETAIN_TRACE_FILTER', 'LLAMA_RECURRENT_SNAPSHOT_LAYER', 'LLAMA_FUSE_MTP_VERIFY', 'GGML_HEXAGON_VERBOSE', 'GGML_HEXAGON_PROFILE',
             'GGML_HEXAGON_ALLOC_FILL', 'GGML_HEXAGON_MOE_TOPK_ALIAS', 'GGML_HEXAGON_ROUTER_HVX', 'GGML_HEXAGON_MM_SELECT', 'GGML_HEXAGON_HEAD_PREFETCH',
             'GGML_HEXAGON_FA_SELECT', 'GGML_HEXAGON_GDN_SELECT', 'GGML_HEXAGON_NDEV', 'GGML_HEXAGON_REPLAY', 'GGML_HEXAGON_OPFUSION', 'GGML_HEXAGON_SSM_SCAN']

# ---------------------------------------------------------------- cases
def _w(d, name, text):
    p = os.path.join(d, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', newline='\n') as f: f.write(text)

def _r(d, name):
    try:
        with open(os.path.join(d, name), encoding='utf-8', errors='replace') as f: return f.read()
    except OSError: return None

def _norm(s): return re.sub(r'\s+', ' ', (s or '')).strip()

def ok(cond, why=''): return (bool(cond), '' if cond else why)

CASES = {}
def case(name, desc):
    def deco(fn): CASES[name] = dict(fn=fn, desc=desc, order=len(CASES) + 1); return fn
    return deco

class Ctx:
    def __init__(self, d, rng): self.d, self.rng, self.state = d, rng, {}

@case('create_file', 'single tool call: create a file with given content')
def c1(ctx):
    prompt = 'Create a file named hello.txt in the current directory containing exactly this single line: Hello from the NPU agent test'
    def check(res):
        t = _r(ctx.d, 'hello.txt')
        return ok(t is not None and t.strip() == 'Hello from the NPU agent test', f'hello.txt={t!r}')
    return prompt, check

@case('read_answer', 'read a file and answer a question about it')
def c2(ctx):
    code = 'BLUE' + ''.join(ctx.rng.choice('HERON') for _ in range(4))
    day = f'2026-{ctx.rng.randint(10, 12)}-{ctx.rng.randint(10, 28)}'
    _w(ctx.d, 'notes.txt', f'Project notes\n=============\nThe internal codename is {code}.\nThe launch deadline is {day}.\nOwner: Dana Whitfield.\nBudget: 12000 credits.\n')
    prompt = 'Read notes.txt and tell me the codename and the launch deadline. Keep the answer to one sentence.'
    def check(res):
        a = re.sub('[‐-―−]', '-', res['final_text'])   # some models emit non-breaking hyphens in dates
        import datetime
        d = datetime.date.fromisoformat(day)
        alts = [day, f'{d:%B} {d.day}, {d.year}', f'{d.day} {d:%B} {d.year}', f'{d:%b} {d.day}, {d.year}', f'{d:%B} {d.day}']
        return ok(code in a and any(x.lower() in a.lower() for x in alts), f'answer={a[:200]!r} want {code} {day}')
    return prompt, check

@case('shell_cmd', 'run a shell command and report its output')
def c3(ctx):
    tok = ''.join(ctx.rng.choice('abcdefghjkmnpqrstuvwxyz') for _ in range(8))
    _w(ctx.d, 'secret.txt', tok + '\n')
    prompt = 'Use the bash tool to run: cat secret.txt | tr a-z A-Z   and then tell me exactly what it printed.'
    def check(res):
        a = res['final_text']
        return ok(tok.upper() in a and res['tool_calls'] >= 1, f'answer={a[:200]!r} want {tok.upper()} tool_calls={res["tool_calls"]}')
    return prompt, check

_HIDDEN_DUR = '''
import solution
cases = {"1h30m": 5400, "45s": 45, "2m": 120, "1h": 3600, "1h1m1s": 3661, "90s": 90, "0s": 0, "10m10s": 610}
for k, v in cases.items():
    assert solution.parse_duration(k) == v, (k, solution.parse_duration(k), v)
for bad in ("", "abc", "5", "1x"):
    try:
        solution.parse_duration(bad)
    except ValueError:
        continue
    raise AssertionError("no ValueError for %r" % bad)
print("HIDDEN_OK")
'''
@case('py_func_test', 'multi-step: write function + test, run test, fix if failing')
def c4(ctx):
    prompt = ('Create solution.py with a function parse_duration(s) that converts strings like "1h30m", "45s", "2m", "1h1m1s" into a total number of seconds (int). '
              'It must raise ValueError for empty or malformed strings (e.g. "", "abc", "5", "1x"). Then create test_solution.py containing plain assert-based tests '
              '(no pytest needed) that print "ALL TESTS PASSED" when they succeed. Run it with `python test_solution.py` and fix any failure until it passes.')
    def check(res):
        if _r(ctx.d, 'solution.py') is None or _r(ctx.d, 'test_solution.py') is None: return False, 'missing solution.py or test_solution.py'
        p = subprocess.run([sys.executable, 'test_solution.py'], cwd=ctx.d, capture_output=True, text=True, timeout=60)
        if p.returncode != 0: return False, 'own test fails: ' + (p.stdout + p.stderr)[-200:]
        h = subprocess.run([sys.executable, '-c', _HIDDEN_DUR], cwd=ctx.d, capture_output=True, text=True, timeout=60)
        return ok('HIDDEN_OK' in h.stdout, 'hidden tests: ' + (h.stdout + h.stderr)[-250:])
    return prompt, check

@case('edit_file', 'edit an existing file at a specific location')
def c5(ctx):
    _w(ctx.d, 'config.ini', '[client]\nhost = localhost\ntimeout = 30\nretries = 3\n\n[server]\nport = 8080\ntimeout = 30\nworkers = 4\n\n[logging]\nlevel = info\n')
    prompt = 'In config.ini change the timeout in the [server] section from 30 to 90. Do not change anything else in the file (the [client] timeout must stay 30).'
    def check(res):
        want = '[client]\nhost = localhost\ntimeout = 30\nretries = 3\n\n[server]\nport = 8080\ntimeout = 90\nworkers = 4\n\n[logging]\nlevel = info\n'
        t = _r(ctx.d, 'config.ini') or ''
        return ok(t.replace('\r\n', '\n') == want, f'config.ini={t!r}')
    return prompt, check

@case('find_count', '3-4 step terminal-bench-like: find files, count lines, write result')
def c6(ctx):
    n_files, total = 0, 0
    for i in range(ctx.rng.randint(3, 5)):
        n = ctx.rng.randint(5, 40); total += n; n_files += 1
        _w(ctx.d, f'logs/{"app" if i % 2 else "svc"}/run{i}.log', ''.join(f'line {j} of run{i}\n' for j in range(n)))
    for i in range(3):   # distractors
        _w(ctx.d, f'logs/notes{i}.txt', 'not a log\n' * 50)
    _w(ctx.d, 'readme.md', '# demo\n')
    prompt = ('Under the logs/ directory (recursively) find all files ending in .log, count the TOTAL number of lines across all of them, '
              'and write the result to result.txt in exactly this format on one line: files=<number of .log files> lines=<total lines>')
    def check(res):
        t = (_r(ctx.d, 'result.txt') or '').strip()
        return ok(t == f'files={n_files} lines={total}', f'result.txt={t!r} want files={n_files} lines={total}')
    return prompt, check

@case('parallel_read', 'several independent reads (parallel tool calls if the model does so) then write sum')
def c7(ctx):
    vals = [ctx.rng.randint(100, 999) for _ in range(4)]
    for n, v in zip('abcd', vals): _w(ctx.d, f'{n}.txt', f'value: {v}\n')
    prompt = 'Read the four files a.txt, b.txt, c.txt and d.txt (they are independent, read them all), add up the four numbers, and write only the sum to sum.txt.'
    def check(res):
        t = (_r(ctx.d, 'sum.txt') or '').strip()
        return ok(t == str(sum(vals)), f'sum.txt={t!r} want {sum(vals)}')
    return prompt, check

@case('long_result', 'long tool result (~20 KB file) to exercise prefill')
def c8(ctx):
    rng = ctx.rng
    words = 'alpha beta gamma delta epsilon zeta theta kappa lambda sigma omega orbit vector matrix kernel buffer socket thread packet queue cache'.split()
    lines = []
    for i in range(380):
        lines.append(f'{i:04d} ' + ' '.join(rng.choice(words) for _ in range(rng.randint(5, 9))) + '.')
    code = ''.join(rng.choice('0123456789') for _ in range(6))
    lines[rng.randint(120, 260)] = f'AUDIT NOTE: the release access code is {code}.'
    _w(ctx.d, 'report.txt', '\n'.join(lines) + '\n')
    ctx.state['bytes'] = os.path.getsize(os.path.join(ctx.d, 'report.txt'))
    prompt = 'Read the file report.txt in full. It contains one line with an "AUDIT NOTE" giving a release access code. Tell me that code.'
    def check(res):
        return ok(code in res['final_text'], f'answer={res["final_text"][:200]!r} want {code}')
    return prompt, check

@case('no_tool', 'no-tool question (must NOT call tools)')
def c9(ctx):
    prompt = 'Without using any tools: what is the capital of France, and what is 12 times 12? Answer in one short sentence.'
    def check(res):
        a = res['final_text'].lower()
        return ok(res['tool_calls'] == 0 and 'paris' in a and '144' in a, f'tool_calls={res["tool_calls"]} answer={a[:150]!r}')
    return prompt, check

@case('fix_bug', 'hard: run failing tests, find the bug in the module, fix it (tests must not be edited)')
def c10(ctx):
    _w(ctx.d, 'inventory.py', '''"""Tiny inventory helpers."""

def total_value(items):
    """items: list of (name, qty, unit_price). Returns total value rounded to 2 decimals."""
    total = 0
    for name, qty, price in items:
        total += qty + price
    return round(total, 2)

def low_stock(items, threshold):
    """Names of items whose qty is strictly below threshold, sorted alphabetically."""
    return sorted(name for name, qty, price in items if qty <= threshold)

def most_valuable(items):
    """Name of the item with the highest qty*price; None for an empty list."""
    if not items:
        return None
    best = max(items, key=lambda it: it[1] * it[2])
    return best[0]
''')
    tests = '''from inventory import total_value, low_stock, most_valuable
ITEMS = [("bolt", 100, 0.25), ("nut", 5, 0.1), ("gear", 3, 12.5), ("axle", 10, 4.0)]
assert total_value(ITEMS) == 103.0, total_value(ITEMS)
assert total_value([]) == 0
assert low_stock(ITEMS, 10) == ["gear", "nut"], low_stock(ITEMS, 10)
assert most_valuable(ITEMS) == "axle", most_valuable(ITEMS)
assert most_valuable([]) is None
print("ALL TESTS PASSED")
'''
    _w(ctx.d, 'test_inventory.py', tests)
    ctx.state['tests'] = tests
    prompt = 'Run `python test_inventory.py`. It fails. Find the bugs in inventory.py and fix them (do not edit test_inventory.py). Re-run the test until it prints ALL TESTS PASSED.'
    def check(res):
        if _r(ctx.d, 'test_inventory.py') != ctx.state['tests']: return False, 'test file was modified'
        p = subprocess.run([sys.executable, 'test_inventory.py'], cwd=ctx.d, capture_output=True, text=True, timeout=60)
        return ok(p.returncode == 0 and 'ALL TESTS PASSED' in p.stdout, (p.stdout + p.stderr)[-200:])
    return prompt, check

@case('rename_multi', 'hard: rename a function across several files, keep tests passing')
def c11(ctx):
    _w(ctx.d, 'pricing.py', 'def calc_total(items, tax=0.0):\n    return round(sum(items) * (1 + tax), 2)\n')
    _w(ctx.d, 'cart.py', 'from pricing import calc_total\n\ndef cart_total(items):\n    return calc_total(items, tax=0.1)\n')
    _w(ctx.d, 'report.py', 'import pricing\n\ndef report(items):\n    return "Total: %s" % pricing.calc_total(items)\n')
    _w(ctx.d, 'docs/notes.md', 'Use `calc_total(items)` for raw totals and cart.cart_total for taxed totals.\n')
    _w(ctx.d, 'test_all.py', 'import pricing, cart, report\nassert pricing.compute_total([1, 2, 3]) == 6\nassert cart.cart_total([10, 10]) == 22.0\nassert report.report([1, 1]) == "Total: 2.0"\nprint("ALL TESTS PASSED")\n')
    ctx.state['test'] = _r(ctx.d, 'test_all.py')
    prompt = 'Rename the function calc_total to compute_total everywhere it is defined, imported, called or mentioned in this project (python files and docs). Then run `python test_all.py` to verify.'
    def check(res):
        for f in ('pricing.py', 'cart.py', 'report.py', 'docs/notes.md'):
            if 'calc_total' in (_r(ctx.d, f) or 'calc_total'): return False, f'{f} still has calc_total'
        if _r(ctx.d, 'test_all.py') != ctx.state['test']: return False, 'test_all.py was modified'
        p = subprocess.run([sys.executable, 'test_all.py'], cwd=ctx.d, capture_output=True, text=True, timeout=60)
        return ok(p.returncode == 0, (p.stdout + p.stderr)[-200:])
    return prompt, check

# ---------------------------------------------------------------- lock + server
def acquire_lock(owner, timeout=4 * 3600):
    if os.environ.get('NPU_LOCK_HELD') == '1': return False
    t0 = last_check = time.time()
    while True:
        try:
            os.mkdir(LOCK)
            open(os.path.join(LOCK, 'owner.txt'), 'w').write(f'{owner} pid={os.getpid()} started={time.ctime()}')
            return True
        except FileExistsError:
            if time.time() - last_check > 5:   # stale-lock recovery (same rule as npu-lock.ps1), checked rarely; polling itself is 20 ms like lockrun.py
                last_check = time.time()
                f = os.path.join(LOCK, 'owner.txt')
                try:
                    pid = int(open(f).read().split('pid=')[1].split()[0])
                    alive = str(pid) in subprocess.run(['tasklist', '/FI', f'PID eq {pid}'], capture_output=True, text=True).stdout
                    if not alive and time.time() - os.path.getmtime(f) > 120: shutil.rmtree(LOCK, ignore_errors=True); continue
                except Exception: pass
            if time.time() - t0 > timeout: raise RuntimeError('NPU lock timeout')
            time.sleep(0.004)

def release_lock(held):
    if held: shutil.rmtree(LOCK, ignore_errors=True)

def start_server(args, tag):
    m = MODELS[args.model]
    env = dict(os.environ)
    for k in CLEAR_ENV: env.pop(k, None)
    env.update(m['env'])
    env['ADSP_LIBRARY_PATH'] = os.path.join(PKG, 'lib') + ';'
    logp = os.path.join(HERE, 'logs', f'server-{tag}.log')
    os.makedirs(os.path.dirname(logp), exist_ok=True)
    cmd = [os.path.join(PKG, 'bin', 'llama-server.exe'), '-m', os.path.join(ROOT, m['gguf']), '--host', '127.0.0.1', '--port', str(args.port),
           '--alias', 'local-model'] + (['--device', 'none', '-ngl', '0'] if args.cpu else ['--device', 'HTP0', '-ngl', '99', '-fit', 'off', '--load-mode', 'none']) + ['-c', str(args.ctx),
           '-b', '2048', '-ub', '1024', '-t', '8' if args.cpu else '4', '-np', '1', '--cache-prompt', '--metrics', '--jinja', '--reasoning', args.reasoning] + m['sampling'] + args.server_extra
    logf = open(logp, 'ab')
    logf.write(('\n=== ' + ' '.join(cmd) + '\n').encode())
    logf.flush()
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, creationflags=0x08000000)
    for _ in range(300):
        time.sleep(2)
        if proc.poll() is not None: raise RuntimeError(f'server exited early rc={proc.returncode}; see {logp}')
        try:
            h = json.load(urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health', timeout=3))
            if h.get('status') == 'ok': return proc, logp
        except Exception: pass
    proc.kill(); raise RuntimeError('server not ready')

def stop_server(proc):
    """Kill only the llama-server PID we launched (taskkill /T fails on this box; TerminateProcess works) and wait until it is gone."""
    if proc is None: return
    for _ in range(3):
        if proc.poll() is not None: break
        proc.kill()
        try: proc.wait(30)
        except Exception: pass
    time.sleep(3)

# ---------------------------------------------------------------- PI run + parsing
LEAK = re.compile(r'</?think(ing)?>|<tool_call>|</tool_call>|<function=|<\|tool_call|\[TOOL_CALLS\]|<\|channel\|>|<\|"\|>|call:\w+\{|<start_function_call>|<\|im_(start|end)\|>')
SRVERR = re.compile(r'(parse|Failed to|failed to|error|invalid|exception|unexpected|empty)', re.I)
SRVERR_IGNORE = re.compile(r'(ggml_hexagon|HTP|htp|cache_prompt|n_cache_reuse|prompt cache|checkpoint|eval time|print_timing|sampler)', re.I)

def run_pi(prompt, cwd, args, jsonl_path, timeout):
    env = dict(os.environ)
    env['PATH'] = os.path.dirname(BUN) + os.pathsep + env.get('PATH', '')
    env['PI_CODING_AGENT_DIR'] = os.path.join(HERE, 'agentdir')
    env['PI_OFFLINE'] = '1'; env['PI_SKIP_VERSION_CHECK'] = '1'; env['PI_TELEMETRY'] = '0'
    cmd = [BUN, PI_CLI, '--mode', 'json', '-p', '--no-session', '--no-extensions', '--no-skills', '--no-prompt-templates', '--no-context-files',
           '--provider', 'local', '--model', 'local-model', '--thinking', 'off', prompt]
    t0 = time.time()
    timed_out = False
    with open(jsonl_path, 'wb') as out, open(jsonl_path + '.err', 'wb') as err:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=out, stderr=err, stdin=subprocess.DEVNULL, creationflags=0x08000000)
        try: p.wait(timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            subprocess.run(['taskkill', '/PID', str(p.pid), '/T', '/F'], capture_output=True); p.kill()
    return time.time() - t0, timed_out

def parse_events(path):
    r = dict(tool_calls=0, tool_errors=0, tools=[], max_parallel=0, assistant_msgs=0, final_text='', stop_reasons=[], leak=[], api_errors=[],
             in_tokens=0, out_tokens=0, empty_turns=0, thinking_chars=0)
    try: lines = open(path, encoding='utf-8', errors='replace').read().splitlines()
    except OSError: return r
    for ln in lines:
        try: ev = json.loads(ln)
        except Exception: continue
        t = ev.get('type')
        if t == 'tool_execution_start':
            r['tool_calls'] += 1; r['tools'].append(ev.get('toolName'))
        elif t == 'tool_execution_end' and ev.get('isError'):
            r['tool_errors'] += 1
            res = json.dumps(ev.get('result'))[:200]
            r['api_errors'].append('tool_error:' + str(ev.get('toolName')) + ':' + res)
        elif t == 'message_end':
            m = ev.get('message') or {}
            if m.get('role') != 'assistant': continue
            r['assistant_msgs'] += 1
            content = m.get('content') or []
            if isinstance(content, str): content = [dict(type='text', text=content)]
            txt = ''.join(b.get('text', '') for b in content if b.get('type') == 'text')
            n_calls = sum(1 for b in content if b.get('type') == 'toolCall')
            r['max_parallel'] = max(r['max_parallel'], n_calls)
            r['thinking_chars'] += sum(len(b.get('thinking', '')) for b in content if b.get('type') == 'thinking')
            sr = m.get('stopReason'); r['stop_reasons'].append(sr)
            if sr in ('error', 'aborted'): r['api_errors'].append(f'stop:{sr}:{m.get("errorMessage")}')
            if not txt.strip() and n_calls == 0: r['empty_turns'] += 1
            for b in content:
                for fld in ('text', 'thinking'):
                    mm = LEAK.search(b.get(fld, '') or '')
                    if mm: r['leak'].append(f'{b.get("type")}:{mm.group(0)}')
            if n_calls == 0: r['final_text'] = txt
            u = m.get('usage') or {}
            r['in_tokens'] += int(u.get('input', 0) or 0) + int(u.get('cacheRead', 0) or 0); r['out_tokens'] += int(u.get('output', 0) or 0)
    return r

TIM = re.compile(r'(prompt eval time|eval time)\s*=\s*([\d.]+) ms\s*/\s*(\d+) (?:tokens|runs)')
def parse_server_segment(text):
    s = dict(prompt_tokens=0, prompt_ms=0.0, gen_tokens=0, gen_ms=0.0, requests=0, cached_tokens=0, ctx_max=0, warn=[])
    for ln in text.splitlines():
        m = TIM.search(ln)
        if m:
            kind, ms, n = m.group(1), float(m.group(2)), int(m.group(3))
            if kind == 'prompt eval time': s['prompt_ms'] += ms; s['prompt_tokens'] += n; s['requests'] += 1
            else: s['gen_ms'] += ms; s['gen_tokens'] += n
            continue
        m = re.search(r'stop processing: n_tokens = (\d+)', ln)
        if m: s['ctx_max'] = max(s['ctx_max'], int(m.group(1)))
        if re.search(r'(Failed to parse|failed to parse|parse error|PEG|Unexpected|unexpected|exception|tool call|tool_call|invalid)', ln, re.I) and not SRVERR_IGNORE.search(ln):
            s['warn'].append(ln.strip()[:300])
    return s

def write_models_json(args):
    mj = json.load(open(os.path.join(HERE, 'agentdir', 'models.json')))
    prov = mj['providers']['local']
    prov['baseUrl'] = f'{args.server_url or "http://127.0.0.1:" + str(args.port)}/v1'
    prov['models'][0]['contextWindow'] = args.ctx
    prov['models'][0]['maxTokens'] = args.max_tokens
    json.dump(mj, open(os.path.join(HERE, 'agentdir', 'models.json'), 'w'), indent=2)

# ---------------------------------------------------------------- report
def fmt(x, nd=1): return '-' if x is None else f'{x:.{nd}f}'

def bashfail(r): return sum(1 for e in r['api_errors'] if e.startswith('tool_error:bash:'))

def write_report(tag, args, results, srv_note):
    json.dump(results, open(os.path.join(HERE, f'results-{tag}.json'), 'w'), indent=1)
    names = sorted({r['case'] for r in results}, key=lambda n: CASES[n]['order'])
    L = [f'# PI tool-calling suite: {tag}', '',
         f'Model: {args.model}  |  server: {srv_note}', '',
         f'PI 0.80.2 (`--mode json -p --no-session`, tools read/bash/edit/write, thinking level off, maxTokens {args.max_tokens}); sampler is server-side (see server cmd in logs/server-{tag}.log).', '',
         '| case | pass | tool calls (avg) | bash exit!=0 | bad tool calls | leaks | ctx tokens (max) | prefill s (sum) | prefill tok/s | decode tok/s | wall s (avg) |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    tot_p = tot_n = 0
    for n in names:
        rs = [r for r in results if r['case'] == n]
        p = sum(r['pass'] for r in rs); tot_p += p; tot_n += len(rs)
        avg = lambda k: statistics.mean(r[k] for r in rs)
        pre_ms = sum(r['srv']['prompt_ms'] for r in rs); pre_tok = sum(r['srv']['prompt_tokens'] for r in rs)
        gen_ms = sum(r['srv']['gen_ms'] for r in rs); gen_tok = sum(r['srv']['gen_tokens'] for r in rs)
        L.append(f'| {n} | {p}/{len(rs)} | {avg("tool_calls"):.1f} | {sum(bashfail(r) for r in rs)} | {sum(r["tool_errors"] - bashfail(r) for r in rs)} | {sum(len(r["leak"]) for r in rs)} | '
                 f'{max(r['srv'].get('ctx_max',0) for r in rs)} | {fmt(pre_ms/1000)} | {fmt(pre_tok/(pre_ms/1000) if pre_ms else None,0)} | {fmt(gen_tok/(gen_ms/1000) if gen_ms else None)} | {avg("wall_s"):.0f} |')
    L += ['', f'**Overall: {tot_p}/{tot_n} passed.**', '', '## Runs', '', '| run | result | tools | calls | tool errs (bash nonzero + bad) | parallel | out tok | prefill tok/ms | gen tok/ms | wall s | note |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in results:
        note = r['why'] if not r['pass'] else ''
        if r['timed_out']: note = 'TIMEOUT ' + note
        if r['leak']: note += ' LEAK=' + ','.join(sorted(set(r['leak'])))
        if r['api_errors']: note += ' ERR=' + '; '.join(r['api_errors'])[:200]
        if r['srv']['warn']: note += ' SRV=' + r['srv']['warn'][0][:160]
        L.append(f'| {r["case"]}-r{r["rep"]} | {"PASS" if r["pass"] else "FAIL"} | {",".join(r["tools"])} | {r["tool_calls"]} | {r["tool_errors"]} ({bashfail(r)}+{r["tool_errors"] - bashfail(r)}) | {r["max_parallel"]} | {r["out_tokens"]} | '
                 f'{r["srv"]["prompt_tokens"]}/{r["srv"]["prompt_ms"]:.0f} | {r["srv"]["gen_tokens"]}/{r["srv"]["gen_ms"]:.0f} | {r["wall_s"]:.0f} | {note.replace("|", "/")[:400]} |')
    open(os.path.join(HERE, f'results-{tag}.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')

# ---------------------------------------------------------------- main
def main():
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=list(MODELS), default='qwen')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--cases', default='all')
    ap.add_argument('--port', type=int, default=8095)
    ap.add_argument('--ctx', type=int, default=16384)
    ap.add_argument('--max-tokens', type=int, default=4096)
    ap.add_argument('--timeout', type=int, default=900)
    ap.add_argument('--reasoning', default='auto', help='llama-server --reasoning value (auto|on|off)')
    ap.add_argument('--cpu', action='store_true', help='run the server on CPU only (still takes the NPU lock so it cannot disturb other agents timings)')
    ap.add_argument('--tag', default='')
    ap.add_argument('--server-url', default='', help='use an existing server instead of launching one (no NPU lock taken)')
    ap.add_argument('--server-extra', nargs=argparse.REMAINDER, default=[], help='extra llama-server args (must be last)')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--merge', default='', help='base,extra: replace the cases of extra inside results-base.json and regenerate')
    ap.add_argument('--regen', default='', help='regenerate results-<tag>.md from results-<tag>.json and exit')
    args = ap.parse_args()
    if args.merge:
        base, extra = args.merge.split(',')
        b = json.load(open(os.path.join(HERE, f'results-{base}.json'))); e = json.load(open(os.path.join(HERE, f'results-{extra}.json')))
        cs = {x['case'] for x in e}
        b = [x for x in b if x['case'] not in cs] + e
        b.sort(key=lambda x: (x['rep'], CASES[x['case']]['order']))
        json.dump(b, open(os.path.join(HERE, f'results-{base}.json'), 'w'), indent=1)
        write_report(base, args, b, 'see logs'); return
    if args.regen:
        res = json.load(open(os.path.join(HERE, f'results-{args.regen}.json')))
        args.model = args.model if args.model in MODELS else 'qwen'
        write_report(args.regen, args, res, 'see logs')
        return
    tag = args.tag or f'{args.model}'
    names = list(CASES) if args.cases == 'all' else args.cases.split(',')
    write_models_json(args)
    os.makedirs(os.path.join(HERE, 'runs', tag), exist_ok=True)
    held = False; proc = None
    results = []
    try:
        if args.server_url:
            logp = None; srv_note = f'external {args.server_url}'
        else:
            held = acquire_lock(f'pitest {tag}')
            proc, logp = start_server(args, tag)
            srv_note = f'llama-server pkg-wos -c {args.ctx} -b 2048 -ub 1024 --jinja --reasoning {args.reasoning} {" ".join(MODELS[args.model]["sampling"])}'
        print('server up', flush=True)
        for rep in range(1, args.reps + 1):
            for name in names:
                rng = random.Random(args.seed * 100 + rep * 10 + CASES[name]['order'])
                d = tempfile.mkdtemp(prefix=f'pitest-{name}-')
                ctx = Ctx(d, rng)
                prompt, check = CASES[name]['fn'](ctx)
                off = os.path.getsize(logp) if logp else 0
                jp = os.path.join(HERE, 'runs', tag, f'{name}-r{rep}.jsonl')
                wall, to = run_pi(prompt, d, args, jp, args.timeout)
                res = parse_events(jp)
                srv = dict(prompt_tokens=0, prompt_ms=0.0, gen_tokens=0, gen_ms=0.0, requests=0, warn=[])
                if logp:
                    time.sleep(0.5)
                    with open(logp, 'rb') as f: f.seek(off); seg = f.read().decode('utf-8', 'replace')
                    srv = parse_server_segment(seg)
                    open(jp + '.server.txt', 'w', encoding='utf-8').write(seg)
                try: passed, why = check(res)
                except Exception as e: passed, why = False, f'checker exception {e!r}'
                if to: passed = False
                rec = dict(case=name, rep=rep, **{k: res[k] for k in ('tool_calls', 'tool_errors', 'tools', 'max_parallel', 'leak', 'api_errors', 'out_tokens', 'empty_turns', 'thinking_chars', 'stop_reasons')},
                           in_tokens_last=res['in_tokens'] / max(1, res['assistant_msgs']), **{'pass': passed}, why=why, timed_out=to, wall_s=wall, srv=srv,
                           final_text=res['final_text'][:300], prompt=prompt)
                results.append(rec)
                print(f'{name} r{rep}: {"PASS" if passed else "FAIL"} calls={res["tool_calls"]} err={res["tool_errors"]} wall={wall:.0f}s '
                      f'prefill={srv["prompt_tokens"]}t/{srv["prompt_ms"]:.0f}ms gen={srv["gen_tokens"]}t/{srv["gen_ms"]:.0f}ms {why[:120]}', flush=True)
                shutil.rmtree(d, ignore_errors=True)
                write_report(tag, args, results, srv_note)
    finally:
        stop_server(proc)
        release_lock(held)
    print('done ->', os.path.join(HERE, f'results-{tag}.md'))

if __name__ == '__main__':
    main()
