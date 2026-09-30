# Upstream sync log (agent `upstream`), append-only

## 2026-09-29 20:40 fetch + survey
- `git fetch origin`: origin/master = 19e28a2 (2026-09-29 14:59 -0700). Merge base with npu-htp-decode/merged2 (ce75353) = 08618ff; 78 upstream commits.
- Worktree `llama.cpp-hexagon-upstream`, branch `upstream-sync` from npu-htp-decode ce75353.
- Upstream Hexagon commits (ggml/src/ggml-hexagon): 2b129cc backend sampler support (ARGMAX op, argsort rewrite into argsort-ops.h, sum-rows/top-k pieces),
  7ac59a6 tiled Q4_0/Q8_0 GET_ROWS (HMX-tile-layout kernel), e904318 FP32 GELU_ERF/GEGLU_ERF, 7fee178 concat op optimization, 19e28a2 f16 activation ops
  (act/unary/binary f16 variants, hvx-base/arith/sigmoid/erf helpers, dma-queue asm "memory" clobbers). None of them touches MOE/MUL_MAT_ID, SSM, HMX matmul, flush/cache
  code or fusion; no overlap with our decode/MTP/Mamba work except the files listed below.
- Overlaps: get-rows-ops.c (our scalar Q4_0 flat kernel + extended-index staging vs upstream tiled kernel), argsort-ops.c (upstream renamed argosrt_ramp_lut -> argsort_ramp_lut, split header;
  our MOE_TOPK block was appended), unary-ops.c/.h + main.c + htp-ops.h (our RELU_SQR vs their STEP/GELU_ERF), concat-ops.c (both touched, auto-merged), supported_get_rows in ggml-hexagon.cpp.
- Non-hexagon upstream changes that matter: f1ea206 "batch: migrate speculative, mtmd and server to batch_ext" (common_batch class, llama_process(); speculative.cpp process/draft
  API changed: `common_speculative_impl::process(const common_batch&)`) -> our MTP driver (fused catch-up etc.) had to be ported; 8019dc5/b5cf8ce ggml graph_inputs collection /
  inputs must be OP_NONE (watch graph-replay/aliasing); d280808 stop accepting draft tokens at EOG; server: typed embeddings input, RANK pooling, HF cache path fixes, jinja
  dict/sameas builtins, chat: response_format json_schema fix (139997d). No change to common/chat parsers of note for tool calls beyond that.

## 2026-09-29 21:00 merge (commit 45b7454) and conflict decisions
- htp-ops.h / main.c / unary-ops.h: kept both sides (our ADD_N..SSM_SCAN..UNARY_RELU_SQR enum entries + upstream ARGMAX/GELU_ERF/GEGLU_ERF; UNARY_STEP case next to RELU_SQR).
- unary-ops.c: kept relu_sqr_f32/tile_relu_sqr_f32 and upstream step kernels side by side.
- get-rows-ops.c: took upstream file (SAMETYPE/TILED/FLAT kernel_type). Upstream's Q4_0 support is the tiled (repacked) kernel, host `supported_get_rows` now rejects unrepacked Q4_0 in a
  hexagon buffer; this supersedes our scalar Q4_0 dequant flat kernel (dropped, not equivalent in format assumption but covers the same tensors: repacked Q4_0 token embeddings).
  Re-applied our extended-index (>4 GB address) staging into VTCM on top of it. `ne[0] % 32` check for Q4_0 dropped (upstream's `src0->type != dst->type && ne[0] < 32` guard remains).
- argsort-ops.c: took upstream file, re-added includes (hvx-arith.h, softmax-ops.h, pf_probe.h) and appended our MOE_TOPK block (ramp lut renamed).
- ggml-hexagon.cpp supported_get_rows type list: upstream form (Q4_0/Q8_0/I32/F16/F32).
- common/speculative.cpp: MTP driver ported to common_batch: `batch.add(...)+set_embd(idx,{ptr,1,n_embd})` instead of common_batch_add+memcpy into batch.embd, llama_process(DECODE)
  instead of llama_decode; fused catch-up (defer_tok/pos/h, flush_deferred, pad_rows, profile timers) preserved. batch_in.token/pos -> batch_in.tokens[k].id/.pos[0].
- Hazard re-check by grep after merge: hex_l2flush 64-byte step, SSM_SCAN, ADD_NW n_src/scale-fold (n_src==8): see below.
- Build: build-upstream.cmd / build-upstream-htp.cmd / pkg-upstream.ps1 (sed copies of the merged2 recipes); host + v81 skel built clean; package pkg-upstream-u1
  (ggml-hexagon.dll 5253C562, skel F19F2646, llama.dll 31284CB8).

## 2026-09-29 21:10 gates
- Qwen PPL 4x512: 12.1307 (exact). 16 chunks: 13.3331 (exact).
- 2026-09-30 00:00 Fusion edge check (edge-u1, OPFUSION on vs mask 8062): 6/6 identical (6,40,130,260,520,900 words).
- Lock starvation note: run scripts that take the NPU lock per run (edge check = 12 acquisitions) waited ~1 h each behind the 10 ms pollers; switched to one-hold bundles
  (`upstream-holdA/B/C.ps1` via lockrun3.py, NPU_LOCK_HELD=1). mamba-run.ps1 did not honour NPU_LOCK_HELD (would self-deadlock inside a bundle): added a 1-line guard.
  (I also wasted ~40 min of lock hold once on exactly that deadlock before killing it; sorry to the others.)
- 2026-09-30 01:40 MTP gate (mtp-gate.ps1 -Prefix up1, Ref pkg-merged2-m1, Cand pkg-upstream-u1, 128 tokens, draft-max 1): ref-plain vs cand-plain, ref-mtp vs cand-mtp
  (also --context-index 1 --draft-greedy), cand-plain vs cand-mtp, ref-plain vs ref-mtp: ALL bits_equal=True, nmse 0, 128/128 rows (74 draft rows); acceptance 52/74 in both.
  So the batch_ext port of the MTP driver (fused catch-up, deferral, pad rows) is behaviour-identical.
- Lightning PPL 4x512 (mamba std env): merged2-m1 13.4941 = upstream-u1 13.4941 (identical). Gemma PPL 4x512 and greedy: identical between ref and cand (bitwise same PPL text/greedy text);
  note Gemma with a raw non-chat prompt is degenerate text on CPU as well (pre-existing, not a regression), and the harness PPL for Gemma is ~1e29 for both builds.
- 2026-09-30 02:28 test-backend-ops -b HTP0 (upstream-u1): GET_ROWS 44/44, ARGSORT 98/98, TOP_K 525/525, CONCAT 48/48, SET_ROWS 55/55 OK. (MUL_MAT_ID/SSM_SCAN not touched by upstream;
  SSM_SCAN oracles re-run anyway via mamba-gates: exact 19/19, hmx-ssd 19/19, negative control 0/19 as required.)
- Nemotron-3-Nano gates (mamba-gates.ps1 -Pkg pkg-upstream-u1): PPL 16.1617 (identical to merged2's 16.1617), KLD vs CPU 0.005273 (merged2 0.00527), greedy vs CPU 3/5 identical
  (p3, p5 diverge from CPU as before); Qwen PPL 12.1307 / 13.3331 again exact.
- 2026-09-30 02:58 holdC self-deadlocked 25 min (npu-lock-fast.ps1 ignored NPU_LOCK_HELD); added the same guard to Enter/Exit-NpuLock there (mamba-run.ps1 earlier). Restarted.
- 2026-09-30 04:05 Interleaved A/B, pkg-merged2-m1 (ref) vs pkg-upstream-u1 (up), 3 rounds, one NPU hold (upstream-holdC.ps1):
  Qwen pp1024 ub1024 (ab-multi, 3 reps/round): ref 763.2 (760.6-765.3) vs up 778.7 (773.4-785.2), +2.0%.
  Gemma 4 pp1024: ref 978.2 vs up 981.8 (+0.4%, within noise).
  Plain decode 256 tok (mtp-abba): ref 26.72 (26.53/26.58/27.05) vs up 27.53 (27.55/27.48/27.55).
  MTP draft-max 1 (FUSED_CATCHUP=1, DRAFT_VOCAB=65536): ref 29.14 vs up 29.60, acceptance 106/149 in every run of both.
  No regression; small gain (likely dma-queue memory clobbers/compiler differences and f16/unary changes; not investigated).
- Decision: upstream-sync tip = 45b7454 (merge commit; no further code commits), package npu_campaign/pkg-upstream-u1. Not installed.
