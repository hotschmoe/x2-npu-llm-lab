"""Compare committed token streams written by llama-speculative-simple (LLAMA_SPEC_TOKENS_OUT).

usage: python clover_tokens_cmp.py REF.tokens CAND.tokens [CAND2.tokens ...] [-n 256]
Prints, per candidate, IDENTICAL (first n tokens equal) or the first divergent index and the tokens around it.
Exit code 1 if any candidate differs within the first n tokens (or is shorter than n while the reference is not).
"""
import sys


def load(path):
    with open(path) as f:
        return [int(x) for x in f.read().split()]


def main():
    args = sys.argv[1:]
    n = 256
    if '-n' in args:
        i = args.index('-n')
        n = int(args[i + 1])
        del args[i:i + 2]
    ref = load(args[0])
    bad = 0
    for cand_path in args[1:]:
        cand = load(cand_path)
        m = min(n, len(ref), len(cand))
        div = next((i for i in range(m) if ref[i] != cand[i]), None)
        if div is None and m < min(n, len(ref)):
            div = m
        if div is None:
            print(f'IDENTICAL first {m} tokens (ref {len(ref)}, cand {len(cand)}): {cand_path}')
        else:
            bad = 1
            print(f'DIVERGES at {div} (ref {len(ref)}, cand {len(cand)}): {cand_path}')
            print(f'  ref : {ref[max(0, div - 3):div + 5]}')
            print(f'  cand: {cand[max(0, div - 3):div + 5]}')
    sys.exit(bad)


if __name__ == '__main__':
    main()
