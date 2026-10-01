#!/usr/bin/env python
"""decisions_shim.py - SGLang-style /v1/decisions in front of a llama-server (stdlib only).

  python serve\\decisions_shim.py [--listen-port 8090] [--upstream http://127.0.0.1:8080]

* POST /v1/decisions : typed questions (choice / score / yes_no) about one input; every answer is the next-token distribution over the
  answer labels after ONE prefill (llama-server /completion, n_predict=1, n_probs, cache_prompt=true); no text is generated.
  Schema follows https://docs.sglang.io/docs/supported-models/decision_models (SGLang PR #40992):
    request : {input, questions:[{id,type,question, options:[{name,description}] | levels:[..] | yes,no}], temperature, chat_template_kwargs,
               prompt_format_version, return_prompt_token_ids, model}
    response: {object:"decisions", model, prompt_format_version, answers:{id:{type, probabilities:{label:p}, choice, score, label_mass,
               [prompt_token_ids,label_token_ids]}}, usage:{prompt_tokens, completion_tokens:0}}
  choice labels A..Z (2-26 options), score labels 0..9 (2-10 levels), yes_no labels yes/no.
  Extras (not in SGLang): answers[id].option_probabilities / level_probabilities (named), answers[id].latency_ms, response.timings.
  Question prompts share the prefix [system + input] and differ only in the suffix, so llama-server's prompt cache only prefills the new part.
* every other path (/v1/chat/completions, /v1/models, /health, /props, /completion, ...) is passed through verbatim (streaming included),
  so PI / any OpenAI client can use this one port.
"""
import argparse, json, math, sys, threading, time, urllib.request, urllib.error, http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

PROMPT_FORMAT_VERSION = 1
SYSTEM = ("You are a precise decision engine. You are given an input and one question about it. "
          "Answer with only the label of your chosen answer, nothing else.")
UP = {'host': '127.0.0.1', 'port': 8080}
_tok_cache = {}
_tok_lock = threading.Lock()
_gen_lock = threading.Lock()      # decisions are serialised (the server has one slot; keeps the prompt cache prefix warm between questions)


def up_json(path, body=None, timeout=600):
    conn = http.client.HTTPConnection(UP['host'], UP['port'], timeout=timeout)
    try:
        if body is None: conn.request('GET', path)
        else: conn.request('POST', path, json.dumps(body), {'Content-Type': 'application/json'})
        r = conn.getresponse(); data = r.read()
        if r.status >= 400: raise RuntimeError(f'upstream {path} -> {r.status}: {data[:300]!r}')
        return json.loads(data)
    finally:
        conn.close()


def label_ids(text):
    """first-token id of label text (cached); the label must be a single token or at least have a distinct first token."""
    with _tok_lock:
        if text in _tok_cache: return _tok_cache[text]
    toks = up_json('/tokenize', {'content': text, 'add_special': False})['tokens']
    if not toks: raise ValueError(f'label {text!r} tokenizes to nothing')
    with _tok_lock: _tok_cache[text] = toks[0]
    return toks[0]


class BadRequest(Exception): pass


def spec_for(q):
    """-> (labels [(label, [variants])], instruction text, names)"""
    t = q.get('type')
    qt = str(q.get('question', '')).strip()
    if not qt: raise BadRequest(f"question {q.get('id')!r}: empty question")
    if t == 'choice':
        opts = q.get('options') or []
        if not 2 <= len(opts) <= 26: raise BadRequest('choice needs 2-26 options')
        labels = [chr(65 + i) for i in range(len(opts))]
        lines = [f"{labels[i]}. {o['name']}" + (f" - {o['description']}" if o.get('description') else '') for i, o in enumerate(opts)]
        txt = f"Question: {qt}\nOptions:\n" + '\n'.join(lines) + f"\nAnswer with the single letter ({labels[0]}-{labels[-1]}) of the best option."
        return [(l, [l]) for l in labels], txt, [o['name'] for o in opts]
    if t == 'score':
        lv = q.get('levels') or []
        if not 2 <= len(lv) <= 10: raise BadRequest('score needs 2-10 levels')
        labels = [str(i) for i in range(len(lv))]
        txt = f"Question: {qt}\nScale:\n" + '\n'.join(f"{labels[i]} = {lv[i]}" for i in range(len(lv))) + f"\nAnswer with the single digit ({labels[0]}-{labels[-1]}) that best matches."
        return [(l, [l]) for l in labels], txt, list(lv)
    if t == 'yes_no':
        y, n = q.get('yes'), q.get('no')
        txt = f"Question: {qt}\n" + (f"yes means: {y}\n" if y else '') + (f"no means: {n}\n" if n else '') + "Answer with exactly one word: yes or no."
        return [('yes', ['yes', 'Yes', 'YES']), ('no', ['no', 'No', 'NO'])], txt, ['yes', 'no']
    raise BadRequest(f"question {q.get('id')!r}: unknown type {t!r} (choice | score | yes_no)")


def render_input(inp):
    if isinstance(inp, str):
        if not inp.strip(): raise BadRequest('input is blank')
        return inp
    if inp is None or inp == '' or inp == [] or inp == {}: raise BadRequest('input is blank')
    return json.dumps(inp, separators=(',', ':'), ensure_ascii=False)


def build_prompt(inp_text, q_text, kwargs):
    kw = dict(kwargs or {}); kw['enable_thinking'] = False       # thinking stays disabled
    body = {'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': f"Input:\n{inp_text}\n\n---\n{q_text}"}],
            'chat_template_kwargs': kw}
    p = up_json('/apply-template', body)['prompt']
    return p


def answer_one(q, inp_text, temperature, kwargs, want_ids):
    labels, qtxt, names = spec_for(q)
    t0 = time.time()
    prompt = build_prompt(inp_text, qtxt, kwargs)
    ids = {l: [label_ids(v) for v in vs] for l, vs in labels}
    flat = {i for v in ids.values() for i in v}
    if len(flat) < sum(len(v) for v in ids.values()) and q['type'] != 'yes_no':
        raise BadRequest('labels do not have distinct first tokens')
    r = up_json('/completion', {'prompt': prompt, 'n_predict': 1, 'temperature': 0, 'n_probs': 256, 'cache_prompt': True,
                                'post_sampling_probs': False, 'return_tokens': want_ids})
    cp = r.get('completion_probabilities') or []
    if not cp: raise RuntimeError('upstream returned no probabilities (is n_probs supported?)')
    tops = cp[0].get('top_logprobs') or cp[0].get('top_probs') or []
    byid = {}
    for e in tops:
        lp = e.get('logprob')
        if lp is None and 'prob' in e: lp = math.log(max(e['prob'], 1e-30))
        byid[e.get('id')] = lp
    # label log-prob = logsumexp over its first-token variants; labels outside the returned top-256 get a floor (their mass is negligible)
    floor = min([v for v in byid.values() if v is not None] or [-30.0]) - 1.0
    raw = {}
    for l, _ in labels:
        vals = [byid.get(i, floor) for i in ids[l]]
        m = max(vals); raw[l] = m + math.log(sum(math.exp(v - m) for v in vals))
    label_mass = sum(math.exp(v) for v in raw.values())
    T = float(temperature or 1.0)
    if T <= 0: raise BadRequest('temperature must be > 0')
    mx = max(raw.values()); ex = {l: math.exp((v - mx) / T) for l, v in raw.items()}; z = sum(ex.values())
    probs = {l: ex[l] / z for l in ex}
    best = max(probs, key=probs.get)
    order = [l for l, _ in labels]
    out = {'type': q['type'], 'probabilities': {l: round(probs[l], 6) for l in order}, 'label_mass': round(label_mass, 6)}
    if q['type'] == 'choice':
        out['choice'] = names[order.index(best)]
        out['option_probabilities'] = {names[i]: round(probs[l], 6) for i, l in enumerate(order)}
    elif q['type'] == 'score':
        out['score'] = round(sum(int(l) * probs[l] for l in order), 6)        # probability-weighted mean level index
        out['choice'] = names[order.index(best)]
        out['level_probabilities'] = {names[i]: round(probs[l], 6) for i, l in enumerate(order)}
    else:
        out['choice'] = best
    if want_ids:
        out['prompt_token_ids'] = r.get('prompt_tokens_ids') or r.get('prompt_token_ids') or None
        out['label_token_ids'] = {l: ids[l][0] for l in order}
    tm = r.get('timings') or {}
    out['latency_ms'] = round((time.time() - t0) * 1000, 1)
    return out, int(tm.get('prompt_n', 0)) + int(tm.get('cache_n', 0)), tm


def handle_decisions(req):
    if not isinstance(req, dict): raise BadRequest('body must be a JSON object')
    inp_text = render_input(req.get('input'))
    qs = req.get('questions')
    if not isinstance(qs, list) or not qs: raise BadRequest('questions must be a non-empty list')
    seen = set()
    for q in qs:
        if not isinstance(q, dict) or not q.get('id') or q['id'] in seen: raise BadRequest('each question needs a unique id')
        seen.add(q['id'])
    answers, total, tms = {}, 0, []
    with _gen_lock:
        for q in qs:
            a, n, tm = answer_one(q, inp_text, req.get('temperature', 1), req.get('chat_template_kwargs'), bool(req.get('return_prompt_token_ids')))
            answers[q['id']] = a; total = max(total, n); tms.append({'id': q['id'], 'prompt_n': tm.get('prompt_n'), 'cache_n': tm.get('cache_n'),
                                                                     'prompt_ms': round(tm.get('prompt_ms', 0), 1)})
    return {'object': 'decisions', 'model': req.get('model') or 'qwen36-35b-a3b', 'prompt_format_version': PROMPT_FORMAT_VERSION,
            'answers': answers, 'usage': {'prompt_tokens': total, 'completion_tokens': 0}, 'timings': tms}


HOP = {'connection', 'keep-alive', 'transfer-encoding', 'te', 'trailer', 'upgrade', 'proxy-authorization', 'proxy-authenticate', 'content-length', 'host'}


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, fmt, *a): sys.stderr.write('%s %s\n' % (self.log_date_time_string(), fmt % a))

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)

    def _proxy(self):
        n = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(n) if n else None
        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        conn = http.client.HTTPConnection(UP['host'], UP['port'], timeout=3600)
        try:
            conn.request(self.command, self.path, body, hdrs)
            r = conn.getresponse()
        except Exception as e:
            self._json(502, {'error': {'message': f'upstream llama-server unreachable: {e}', 'type': 'bad_gateway'}}); return
        try:
            self.send_response(r.status)
            for k, v in r.getheaders():
                if k.lower() not in HOP: self.send_header(k, v)
            # stream (SSE or not) with connection close framing: simple and correct for chunked/unknown-length upstream bodies
            self.send_header('Connection', 'close'); self.close_connection = True
            self.end_headers()
            while True:
                chunk = r.read1(65536) if hasattr(r, 'read1') else r.read(65536)
                if not chunk: break
                self.wfile.write(chunk); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError): pass
        finally: conn.close()

    def do_POST(self):
        if urlparse(self.path).path.rstrip('/') == '/v1/decisions':
            try:
                n = int(self.headers.get('Content-Length') or 0)
                req = json.loads(self.rfile.read(n) or b'{}')
                self._json(200, handle_decisions(req))
            except (BadRequest, ValueError, KeyError) as e: self._json(400, {'error': {'message': str(e), 'type': 'invalid_request_error'}})
            except Exception as e: self._json(502, {'error': {'message': str(e), 'type': 'upstream_error'}})
        else: self._proxy()
    do_GET = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = _proxy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--listen-host', default='127.0.0.1'); ap.add_argument('--listen-port', type=int, default=8090)
    ap.add_argument('--upstream', default='http://127.0.0.1:8080')
    a = ap.parse_args()
    u = urlparse(a.upstream); UP['host'], UP['port'] = u.hostname, u.port or 80
    srv = ThreadingHTTPServer((a.listen_host, a.listen_port), H); srv.daemon_threads = True
    print(f'decisions shim on http://{a.listen_host}:{a.listen_port} -> {a.upstream}', flush=True)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass


if __name__ == '__main__': main()
