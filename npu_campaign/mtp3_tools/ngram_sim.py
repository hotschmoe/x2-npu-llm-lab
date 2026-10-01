"""Offline n-gram (prompt-lookup) draft simulator on recorded agentic sessions (pitest runs).
Approximate tokens with a GPT-like regex. Cost model from MTP3_LOG: wall ms per verify of T tokens (llama-bench pp1..8).
usage: ngram_sim.py DIR [N] [M]  -> per-N/M statistics and modeled tok/s: plain, MTP-only(a), ngram+MTP fallback."""
import json, glob, re, sys, random
tokre = re.compile(r"\s?[A-Za-z]+|\s?\d|\s?[^\sA-Za-z\d]|\s+")
def toks(s): return tokre.findall(s)
V = {1:35.3, 2:53.5, 3:65.5, 4:76.0, 5:82.0, 6:88.0, 7:94.0, 8:100.5, 9:106, 10:112}
def vcost(T): return V.get(T, 100.5 + 6*(T-8))
def sessions(d):
    for f in sorted(glob.glob(d + '/*.jsonl')):
        ev = []
        for l in open(f, encoding='utf-8'):
            try: j = json.loads(l)
            except: continue
            if j.get('type') == 'message_end':
                m = j['message']; role = m['role']
                for c in m['content']:
                    if c['type'] == 'text': ev.append((role, 'text', c['text']))
                    elif c['type'] == 'thinking': ev.append((role, 'gen', c['thinking']))
                    elif c['type'] == 'toolCall': ev.append((role, 'gen', json.dumps(c['arguments'])))
                    
                if role == 'assistant':
                    ev = [(r, ('gen' if (k == 'text' and r == 'assistant') else k), t) for (r, k, t) in ev]
        yield f, ev
def simulate(ev, N, M, a_mtp, rng):
    hist = []      # token history
    steps = []     # list of (advance, cost)
    plain_cost = 0; plain_tok = 0
    mtp_cost = 0; comb_cost = 0; tok_total = 0
    ng_steps = 0; ng_acc = 0; ng_prop = 0
    for role, kind, text in ev:
        t = toks(text)
        if kind != 'gen':
            hist += t; continue
        # generated block: simulate
        p = 0
        while p < len(t):
            # n-gram lookup: last N tokens of hist (context+generated so far)
            draft = []
            if len(hist) >= N:
                key = tuple(hist[-N:])
                # most recent earlier occurrence
                for i in range(len(hist) - N - 1, -1, -1):
                    if tuple(hist[i:i+N]) == key:
                        draft = hist[i+N:i+N+M]; break
            rem = len(t) - p
            if draft:
                draft = draft[:max(0, rem-1)] if rem > 1 else []
            if draft:
                k = 0
                while k < len(draft) and draft[k] == t[p+k]: k += 1
                adv = min(k + 1, rem)
                comb_cost += vcost(1 + len(draft)) ; ng_steps += 1; ng_acc += k; ng_prop += len(draft)
            else:
                acc = 1 if (rem > 1 and rng.random() < a_mtp) else 0
                adv = 1 + acc
                comb_cost += vcost(2) + 3.3 + 1.7 - 0.0  # verify T=2 (wall) + draft + host
                
            hist += t[p:p+adv]; p += adv; tok_total += adv
    return tok_total, comb_cost, ng_steps, ng_acc, ng_prop
if __name__ == '__main__':
    d = sys.argv[1]
    a = float(sys.argv[2]) if len(sys.argv) > 2 else 0.71
    rng = random.Random(1)
    allev = list(sessions(d))
    print('sessions', len(allev))
    for N in (3, 4, 6, 8, 12):
        for M in (1, 3, 5, 8):
            tt = 0; cc = 0; ns = 0; na = 0; npr = 0
            for f, ev in allev:
                a_, b_, c_, d_, e_ = simulate(ev, N, M, a, rng)
                tt += a_; cc += b_; ns += c_; na += d_; npr += e_
            # baselines on same tokens
            plain = tt / (tt * 35.6e-3)
            print(f'N={N:2d} M={M} tokens={tt} ngram-steps={ns} ({ns/max(1,tt)*100:.0f}% of tokens/step-ish) proposed={npr} accepted={na} ({na/max(1,npr)*100:.0f}%) modeled tok/s={tt/(cc/1000):.1f} (plain 28.1, MTP-only ~{ (1+a)/(vcost(2)+5.0)*1000:.1f})')
