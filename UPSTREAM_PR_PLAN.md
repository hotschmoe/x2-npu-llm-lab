# Upstreaming plan (ggml-org/llama.cpp, Hexagon backend)

Produced 2026-09-29 by a review agent against upstream master `139997d` (our fork base `08618ff`, 48 commits behind;
none touch matmul-ops.c / hmx-mm-kernels-tiled.h / main.c scratchpad). Nothing has been pushed or opened.

## Contribution rules that matter (CONTRIBUTING.md / AGENTS.md / PR template)

- AI-generated code is allowed but disclosed: "You are 100% responsible for every line ... Undisclosed AI usage may result
  in your account being permanently banned". PR template has a mandatory AI-usage disclosure line.
- **PR descriptions, commit messages, issue text and review replies must be written by the human.** Agents must never
  write PR text; automated commits/PRs are prohibited. If an agent commits at a human's request the trailer is
  `Assisted-by: <assistant>` (not `Co-authored-by:`). Our fork history carries `Co-Authored-By: Claude`, so the public
  branch must be re-authored by the human.
- Style: 4 spaces, snake_case, `void * ptr`, no unicode dashes/arrows, 1-2 line comments, clang-format v15+, editorconfig CI.
- Features start with an issue/discussion; new contributors: one open PR at a time; squash-merge; code owners
  (@ggml-org/ggml-hexagon: lhez, max-krasnyansky) approve. No new files under tests/ without maintainer approval.
- Evidence expected: llama-bench before/after tables (pp25..pp2048), llama-perplexity not negatively affected,
  test-backend-ops for new/changed ops. Snapdragon CI only builds Android; maintainers test on Android v73/v75/v79/v81 -
  we only have one X2E machine (state that honestly).

## Classification of our work

| Finding | Class | Evidence | Notes |
| --- | --- | --- | --- |
| Pipelined HMX MUL_MAT_ID (hmx_mm_id_pipelined, chunked hot experts) | general | pp512 385 -> 551, ub1024 pp2048 454 -> 646, PPL identical | strip diagnostic hooks first |
| ddr_spad 512 KB -> 2 MB | general | (with pipeline) +13% | needs standalone A/B on stock kernel; should grow lazily |
| Exact-range activation gather | general | +4% | replaces stock gathered worker |
| HTP_OP_MAX_BUFS 16 -> 64 | general | decode 59 -> 49 ms/tok on 35B (with other changes) | re-measure isolated |
| SSM_CONV window fusion (CONCAT+CPY+SSM_CONV) | general (GDN/Mamba) | decode 23.1 -> 25.8 tok/s | PR4 |
| ADD+N expert-sum fusion | general | +2%, single runs | needs repeats |
| Graph replay | risky | +3% | RFC first |
| ROUTER_HVX=0 (router on HMX) | do not upstream | +10% pp, PPL +0.26%, expert choice changes | opt-in only |
| *_REFERENCE diagnostics, logits tracing, MTP verify tail, probe headers | campaign-only | | never upstream |

## PR order

0. Hygiene: branch from upstream/master and port by hand (do not rebase the fork); strip every *_REFERENCE/PER_OP/MMID_PIPE
   ifdef; make the pipeline a host-decided kernel param (`htp_mm_kernel_params`) + runtime opt_ env (document in
   docs/backend/snapdragon/README.md); fix CRLF/whitespace; ASCII-only; human-authored text; open an issue first.
1. HTP_OP_MAX_BUFS -> 64. 2. exact-range gather + grown ddr_spad. 3. pipelined MUL_MAT_ID (after 2; check open PR #29626
   which touches the same files). 4. conv-window fusion. 5. ADD_N fusion (repeat measurements). 6. bug-fix candidates only
   with a repro. 7. graph replay RFC.

## Evidence commands

```
llama-bench -m Qwen3.6-35B-A3B-Q4_0.gguf -p 16,64,128,512,1024,2048 -n 128 -ub 512,1024 -b 2048 -r 3 -ngl 99 --device HTP0 -o md
llama-perplexity -m <model> -f wikitext-2-raw/wiki.test.raw -c 512 -b 512 -ub 512 --chunks 32 -ngl 99 --device HTP0 -fit off --kl-divergence-base base.kld
test-backend-ops test -b HTP0 -o MUL_MAT_ID ; -o MUL_MAT_ID_FUSION ; -o SSM_CONV
```

## Open evidence gaps

Single machine only; standalone benefit of the 2 MB scratchpad and of the exact gather on the stock kernel not measured;
some decode fusion gains are single runs; MoE-shaped test-backend-ops cases not yet run against an upstream-rebased build.
