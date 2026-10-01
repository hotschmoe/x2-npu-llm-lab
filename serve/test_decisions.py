#!/usr/bin/env python
"""Sanity tests for decisions_shim against a running server: python serve\\test_decisions.py [--url http://127.0.0.1:8090]
Checks: probabilities sum to 1, option-order permutation does not flip the answer, latency, prefix-cache reuse (timings), error handling."""
import argparse, json, time, urllib.request, urllib.error

ap = argparse.ArgumentParser(); ap.add_argument('--url', default='http://127.0.0.1:8090'); a = ap.parse_args()

def post(body):
    t = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request(a.url + '/v1/decisions', json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=600)
        return json.load(r), time.time() - t
    except urllib.error.HTTPError as e: return {'http': e.code, 'body': e.read().decode()}, time.time() - t

TICKET = "Hi, our payment integration has been returning 500 errors for three days and our checkout is down. We lose orders every hour. Please fix ASAP."
OPTS = [{'name': 'billing', 'description': 'invoices, refunds, payment disputes'}, {'name': 'technical', 'description': 'bugs, outages, API errors'}, {'name': 'sales', 'description': 'pricing, new contracts'}]
fails = 0
def check(name, cond, extra=''):
    global fails
    print(('PASS ' if cond else 'FAIL ') + name + (' ' + str(extra) if extra else ''))
    fails += (not cond)

r, dt = post({'input': TICKET, 'questions': [
    {'id': 'urgent', 'type': 'yes_no', 'question': 'Does the customer need help today?'},
    {'id': 'team', 'type': 'choice', 'question': 'Which team should handle this?', 'options': OPTS},
    {'id': 'sev', 'type': 'score', 'question': 'How severe is the problem?', 'levels': ['trivial', 'minor', 'moderate', 'major', 'critical']}]})
print(json.dumps(r, indent=1)); print(f'wall {dt:.2f}s')
A = r['answers']
for k, v in A.items(): check(f'{k}: probs sum to 1', abs(sum(v['probabilities'].values()) - 1) < 1e-3, v['label_mass'])
check('urgent=yes', A['urgent']['choice'] == 'yes'); check('team=technical', A['team']['choice'] == 'technical'); check('sev >= 3 (major/critical)', A['sev']['score'] >= 3, A['sev']['score'])
check('completion_tokens 0', r['usage']['completion_tokens'] == 0)

# order invariance: reversed options must pick the same named option
r2, _ = post({'input': TICKET, 'questions': [{'id': 'team', 'type': 'choice', 'question': 'Which team should handle this?', 'options': OPTS[::-1]}]})
p1, p2 = A['team']['option_probabilities'], r2['answers']['team']['option_probabilities']
check('order permutation keeps choice', A['team']['choice'] == r2['answers']['team']['choice'], {k: (p1[k], p2[k]) for k in p1})

# negative / mild example, repeated call latency (warm cache), second input
r3, dt3 = post({'input': 'Thanks, the invoice looks right now. No rush at all, whenever you have time.', 'questions': [
    {'id': 'urgent', 'type': 'yes_no', 'question': 'Does the customer need help today?'},
    {'id': 'team', 'type': 'choice', 'question': 'Which team should handle this?', 'options': OPTS}]})
check('mild: urgent=no', r3['answers']['urgent']['choice'] == 'no', r3['answers']['urgent']['probabilities'])
check('mild: team=billing', r3['answers']['team']['choice'] == 'billing', r3['answers']['team']['option_probabilities'])
print('timings', r3['timings'], f'wall {dt3:.2f}s')
r4, dt4 = post({'input': TICKET, 'questions': [{'id': 'urgent', 'type': 'yes_no', 'question': 'Does the customer need help today?'},
                                               {'id': 'angry', 'type': 'yes_no', 'question': 'Is the customer angry?'}]})
print('repeat timings', r4['timings'], f'wall {dt4:.2f}s')
# errors
e, _ = post({'input': ' ', 'questions': []}); check('blank input -> 400', e.get('http') == 400)
e, _ = post({'input': 'x', 'questions': [{'id': 'q', 'type': 'choice', 'question': 'q', 'options': [{'name': 'a'}]}]}); check('1 option -> 400', e.get('http') == 400)
print('FAILS', fails)
