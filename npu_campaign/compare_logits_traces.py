import argparse
import struct
import re
from pathlib import Path
from collections import defaultdict

import numpy as np


def read(path):
    contexts = defaultdict(list)
    with open(path, 'rb') as file:
        while header := file.read(20):
            if len(header) != 20:
                raise ValueError('truncated header')
            magic, context, row, count = struct.unpack('<IQiI', header)
            if magic != 0x4C475431 or count > 1_000_000:
                raise ValueError('bad trace header')
            payload = file.read(count * 8)
            if len(payload) != count * 8:
                raise ValueError('truncated row')
            pairs = np.frombuffer(payload, dtype='<u4').reshape(count, 2)
            ids = pairs[:, 0].copy()
            values = pairs[:, 1].copy().view('<f4')
            contexts[context].append((row, ids, values))
    if not contexts:
        raise ValueError('empty trace')
    return contexts


def selected(path):
    log = Path(str(path).replace('.logits.bin', '.detail.log'))
    result = defaultdict(list)
    if log.exists():
        for context, token in re.findall(rb'logits-trace-selected ctx=([0-9A-Fa-f]+) idx=-?\d+ token=(-?\d+)', log.read_bytes()):
            result[int(context, 16)].append(int(token))
    return result


parser = argparse.ArgumentParser()
parser.add_argument('reference')
parser.add_argument('candidate')
parser.add_argument('--require-identical', action='store_true', help='Require complete aligned token logs and bit-identical logits.')
parser.add_argument('--require-tokens', action='store_true', help='Require complete matching selected-token sequences, allowing numerical differences.')
parser.add_argument('--prefix-rows', type=int, help='Explicit bounded diagnostic prefix; not full-run qualification. Requires a strict gate.')
parser.add_argument('--context-index', type=int, default=0, help='Trace context in first-seen order: 0=target, 1=draft for MTP.')
parser.add_argument('--draft-greedy', action='store_true', help='For MTP draft context, compare pre-sampler logits and their top candidate; ignore the sampled token that MTP does not use.')
args = parser.parse_args()
if args.prefix_rows is not None and (args.prefix_rows < 1 or not (args.require_identical or args.require_tokens)):
    parser.error('--prefix-rows requires a positive count and a strict gate')
if args.context_index < 0:
    parser.error('--context-index must be nonnegative')
if args.draft_greedy and args.context_index != 1:
    parser.error('--draft-greedy requires --context-index 1')
a_contexts, b_contexts = read(args.reference), read(args.candidate)
print('reference contexts/rows:', [(hex(k), len(v)) for k, v in a_contexts.items()])
print('candidate contexts/rows:', [(hex(k), len(v)) for k, v in b_contexts.items()])
if args.context_index >= min(len(a_contexts), len(b_contexts)):
    raise ValueError(f'context-index {args.context_index} is not present in both traces')
a_context, b_context = list(a_contexts)[args.context_index], list(b_contexts)[args.context_index]
a, b = a_contexts[a_context], b_contexts[b_context]
sa = selected(args.reference).get(a_context, [])
sb = selected(args.candidate).get(b_context, [])
print(f'comparison_context_index={args.context_index} reference={hex(a_context)} candidate={hex(b_context)}')
has_selected = len(sa) == len(a) and len(sb) == len(b)
if args.draft_greedy:
    # MTP calls common_sampler_sample(), but uses get_candidates(..., true)->data[0],
    # not the sampled ID. The selected trace can vary without changing the draft.
    has_selected = False
print('selected_token_alignment_available:', has_selected)
if args.prefix_rows is not None:
    if len(a_contexts) != 1 or len(b_contexts) != 1 or not has_selected:
        raise ValueError('Bounded comparison requires single contexts and complete selected-token logs')
    if min(len(a), len(b)) < args.prefix_rows:
        raise ValueError('Requested prefix is not available in both complete traces')
    a, b = a[:args.prefix_rows], b[:args.prefix_rows]
    sa, sb = sa[:args.prefix_rows], sb[:args.prefix_rows]
    print(f'comparison_scope=bounded_prefix rows={args.prefix_rows}; NOT full-run qualification')
if (args.require_identical or args.require_tokens) and (len(a) != len(b) or (not has_selected and not args.draft_greedy)):
    raise ValueError('Strict comparison requires equal row counts and complete selected-token logs')
first_bits = first_top = None
compared = 0
selected_difference = False
for step, ((a_row, a_ids, av), (b_row, b_ids, bv)) in enumerate(zip(a, b)):
    compared += 1
    if not np.array_equal(a_ids, b_ids):
        raise ValueError(f'candidate ID layout differs at step {step}')
    if not np.isfinite(av).all() or not np.isfinite(bv).all():
        raise ValueError(f'nonfinite logits at step {step}')
    bit_equal = np.array_equal(av.view('<u4'), bv.view('<u4'))
    if not bit_equal and first_bits is None:
        first_bits = step
    ai, bi = int(np.argmax(av)), int(np.argmax(bv))
    top_equal = int(a_ids[ai]) == int(b_ids[bi])
    if not top_equal and first_top is None:
        first_top = step
    delta = av.astype(np.float64) - bv.astype(np.float64)
    nmse = float(np.dot(delta, delta) / max(float(np.dot(av.astype(np.float64), av.astype(np.float64))), 1e-30))
    if step < 5 or not top_equal or (not bit_equal and first_bits == step):
        print(f'step={step} rows={a_row}/{b_row} bits_equal={bit_equal} nmse={nmse:.9g} max_abs={np.max(np.abs(delta)):.9g} '
              f'top={int(a_ids[ai])}/{int(b_ids[bi])} scores={av[ai]:.9g}/{bv[bi]:.9g} '
              f'candidate_ref_top_score={bv[ai]:.9g}')
    if has_selected and sa[step] != sb[step]:
        selected_difference = True
        print(f'first_selected_token_difference={step} selected={sa[step]}/{sb[step]}')
        ref_indices = np.flatnonzero(a_ids == sa[step])
        cand_indices = np.flatnonzero(a_ids == sb[step])
        if len(ref_indices) == 1 and len(cand_indices) == 1:
            ri, ci = int(ref_indices[0]), int(cand_indices[0])
            print(f'selected_scores reference={av[ri]:.9g}/{av[ci]:.9g} '
                  f'candidate={bv[ri]:.9g}/{bv[ci]:.9g} '
                  f'reference_margin={float(av[ri]) - float(av[ci]):.9g} '
                  f'candidate_margin={float(bv[ci]) - float(bv[ri]):.9g}')
        print('Stopping: later rows need not share the same generated prefix.')
        break
    if not has_selected and not top_equal:
        print('Stopping conservatively: no selected-token trace to establish the common prefix.')
        break
print(f'first_bit_difference={first_bits} first_raw_argmax_difference={first_top}')
print(f'compared_rows={compared} reference_rows={len(a)} candidate_rows={len(b)}')
if args.require_identical and (first_bits is not None or selected_difference or first_top is not None or compared != len(a)):
    raise SystemExit(1)
if args.require_tokens and (selected_difference or (args.draft_greedy and first_top is not None) or compared != len(a)):
    raise SystemExit(1)
