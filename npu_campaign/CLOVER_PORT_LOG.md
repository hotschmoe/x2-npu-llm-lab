# CLOVER_PORT_LOG (append-only) - agent `port`
Worktree `llama.cpp-hexagon-clover-port`, branch `clover-port` (from `clover` d587232). Build `npu_campaign/build-clover-port.cmd` (dir `build-clover-port`),
packages `npu_campaign/pkg-clover-port-<x>` via `pkg-clover-port.ps1 -Name <x>`. Baseline package for A/B: `pkg-clover-topk-c`.

## 2026-10-30 03:40 Setup + feature 1: incremental dirty-range merge
- Cherry-pick of 8666b61 (hexagon-upstream-prep) conflicted only on the file rename (upstream renamed htp-tensor.c -> htp-dcache.c and the functions
  htp_tensor_dirty_all/htp_flush_dirty_ranges/htp_tensor_flush_all -> htp_dcache_mark_dirty_tensors/..._flush_dirty_ranges/..._flush_dirty_tensors).
  Applied the htp-tensor.c hunks to htp-dcache.c with a sed-renamed patch (all hunks applied cleanly, content of the tracker otherwise unchanged upstream).
  ctx/main.c hunks (dirty_hi field + reset next to memset) cherry-picked as is. Branch `clover-port` commit 00d7f9b. Build OK (host + htp-v81).
- Package `pkg-clover-port-dr` (host ggml-hexagon.dll 051B4B7D, skel AB865B38). Gates queued: clover-kld (label port-dr).

## 2026-10-30 04:00 Survey: what in the fork is still worth porting (read of PR_PREP_NOTES, BOLT_PR_PREP, OVERNIGHT_SUMMARY, QWEN4_LOG, OPS650_LOG, PREFILL_CAMPAIGN vs current upstream code)
Upstream (f4d8af6) has already absorbed or re-invented much of the fork's decode/prefill fusion set: htp-fusion.cpp has RMS_NORM_MUL, RMS_NORM_SCALE, MUL_MAT_ADD, MUL_MAT_NX, MUL_MAT_ID_NX,
GDN+CPY, SSM_CONV+SILU, UNARY+MUL->GLU, ADD_N (binary_nx), REPEAT+GET_ROWS; an OPBATCH queue; kparams solved on the host; dma_queue_push already collapses contiguous pushes.
=> NOT ported (already upstream): conv-window SILU fold, SILU+MUL, expert-combine ADD_N, RMS_NORM_SCALE, MoE fusions. Replay (GGML_HEXAGON_REPLAY): fork-only, +3%, risky (RFC-grade) - skipped.
Ranking for Clover (its steps are 16-row blocks + 1-row decodes with 500-1160 splits per graph, so per-op fixed cost and small-M kernels matter, not M=1 expert streaming):
 1. dirty-range tracker (fork: 4 ms of a 34 ms token; upstream Qwen decode +15.5%)          -> ported (00d7f9b)
 2. Q4_0/IQ4_NL tiles padded to 640 B in DDR (Q4C/s13; fork plain decode +13.7%)               -> ported as a 3-line change thanks to upstream's htp_tiled_row_size/tile_padded_size plumbing
 3. GDN snapshot DMA (rollback verify writes up to 16 snapshots x 2 MiB x 30 layers; rollback log: "the snapshots are the real price") -> ported
 4. bounded OPPOLL spin + watchdog (product goal: idle host CPU; spin only helps latency)       -> ported; upstream default stays blocking read
 5. xp flat expert pipeline (M=1..4 only; Clover decode steps P=102 ms are 1-row but dominated by CPU splits) -> evaluate after profile
 6. pipelined HMX MoE (hmx_mm_id_pipelined, fork pp16 71->142 with 2 MB ddr_spad, exact gather): large rewrite against upstream matmul-ops.c (+900 lines since fork base), needs standalone evidence -> documented, not ported unless profile shows MUL_MAT_ID dominates the 16-row passes.
Lock queue was 7-9 deep for the whole session, so runs below finish late; build/pkg names: dr = feature 1, pad = 1+2, spin = 1+2+4, all = 1+2+3+4 (pkg-clover-port-<x>).
Tools added: npu_campaign/clover-port/{tbo.ps1 (locked test-backend-ops), abab.ps1 (interleaved clover-run A/B), q1.ps1, snap_patch.py}; test-backend-ops perf cases (BEGIN/END port agent block).

## 2026-10-30 04:15 Code state (branch clover-port)
00d7f9b dirty-range merge | 82354b9 Q4_0/IQ4_NL tile pad 640 (htp-tensor.h, htp-repack.h only) | d1055ae bounded OPPOLL spin (+GGML_HEXAGON_OPPOLL_SPIN_MS, RSP_TIMEOUT_S; Windows dspqueue timeout is infinite so
the fallback is an infinite blocking read, the watchdog only matters off-Windows) | 826e193 GDN pp kernel snapshot DMA (new kernel type HVX_RECURRENT_SNAP, env GGML_HEXAGON_GDN_SNAPDMA=0 disables) | test-backend-ops perf cases.
All build clean (host + htp-v81). Packages: dr, pad, spin, all.

## 2026-10-30 04:50 Gates so far (all through the NPU lock; clover-kld with the OLD CPU-bin reference cpu-kld-base.bin, i.e. directly comparable to the baseline 0.047923 / 89.216% / 15.0832)
| package | KLD mean | same-top | PPL(Q) | note |
| --- | --- | --- | --- | --- |
| pkg-clover-port-dr  (dirty-range merge)            | 0.047923 +- 0.002348 | 89.216% | 15.083178 | identical to topk-c baseline |
| pkg-clover-port-pad (+ Q4_0/IQ4_NL tile pad 640 B) | 0.047923 | 89.216% | 15.083178 | identical |
| pkg-clover-port-all (+ bounded spin + GDN snapDMA) | 0.047923 | 89.216% | 15.083178 | identical (K=1 causal path does not use snapshots) |
Op-level: test-backend-ops MUL_MAT_ID on pad: 796/796 pass. GATED_DELTA_NET on all: the 16 failures are all K=1 / n_tokens >= 64 (HMX chunked kernel, NMSE 1e-6..6e-3 vs 1e-7 limit), K>1 cases (K=2,4, kda and non-kda) OK;
GATED_DELTA_NET_CACHE_FUSION (dst_cache + K>1 -> exercises the SNAP kernel) 5/5 pass. (baseline failure list on topk-c queued to confirm they are pre-existing.)
Profile (GGML_HEXAGON_PROFILE=1 -v, pkg pad, 64 tokens): DSP busy 9.3 s of 36.7 s generation window (25%), 21074 batches of 7.9 ops (mean 467 us), mean host gap between batches 1.4 ms (median 227 us; profile logging inflates it).
Draft-pass 16-row MoE: MUL_MAT_ID_NX (gate+up, hmx-tiled) 1494 us/op x 40 = 60 ms, MUL_MAT_ID down 797 us x 40 = 32 ms, dense 16-row MUL_MAT 124 us x ~160 = 20 ms => ~140 ms DSP per draft pass;
the DSP is only a quarter of wall time, so DSP-side ports are capped at a few % until CPU fallbacks/splits (bf16 agent) shrink.
A/B block ab2a (one lock hold, ABBA, 256 tokens, same prompt, exact verify, AC): topk-c 5.096 / 5.178, all 5.249 / 5.254 tok/s => +2.2% (mean 5.137 -> 5.252); output hash identical in all 4 runs; host cores 0.99 / 0.98.

## 2026-10-30 06:35 Speed A/B results (blocks of 4 runs under ONE lock hold, ABBA, draft-self n_max 15, exact verify, 256 tokens, diffusion.txt, AC; output hash identical in all runs)
| block | arm: tok/s (runs) | mean |
| --- | --- | ---: |
| ab2a/ab2b | topk-c: 5.096 5.178 5.118 5.093 | 5.121 |
|           | all (dr+pad+spin-code+snap, default env): 5.249 5.254 5.210 5.234 | 5.237 (+2.3%) |
| ab2c | dr only: 5.257 5.230 | 5.244 (+2.4% vs topk-c) |
|      | pad (dr + Q4_0 tile pad): 5.283 5.278 | 5.281 (+3.1%; pad alone +0.7% over dr) |
| ab3a | all: 5.243 5.186 | 5.215 |
|      | all + GGML_HEXAGON_OPPOLL=1 (bounded spin, 400 ms): 5.729 | +9.9% |
|      | all + OPPOLL=1 + OPPOLL_SPIN_MS=0.5: 5.514 | +5.7% |
Host CPU busy cores: 0.98-0.99 in EVERY arm, including OPPOLL=0. Per-thread census (clover-port/thr.ps1; 250 ms samples over the generation window): exactly ONE thread (main) burns 0.99 core,
all ggml worker threads 0.000 (also with -t 1/2 and --poll 0: tok/s unchanged 4.70-4.87); Qwen3.6 plain decode on HTP0 (no CPU fallbacks at all) shows the same 0.97 core on the main thread.
=> on Windows the blocking dspqueue_read (DSPQUEUE_TIMEOUT_NONE) is effectively busy-waiting in the driver; the "idle host" goal cannot be met by host flags, and OPPOLL=1 costs no extra core while removing wake-up latency (+10%).
(DSP is busy only 25% of wall in a Clover step, so the remaining main-thread time is host work: graph/sched/solve/enqueue - see the hoststat experiment below.)
llama-bench (-p 16,256 -n 32 -r 2, ABBA, one lock hold): topk-c pp16 45.75/45.32, pp256 235.57/239.97, tg32 10.78/10.91; all pp16 46.33/46.73, pp256 242.87/239.47, tg32 11.25/11.15 (+2.2% / +1.4% / +3.2%).
GDN snapshot DMA op-level (perf, 32 heads x S_v 128 x 16 tokens): K=6 577.6 -> 498.1 us (-13.8%), K=16 659.9 -> 532.6 us (-19.3%), K=1 unchanged (280 -> 278). Only used when the rollback branch (K>1) lands in clover.
GDN failure list (K=1, n_tokens >= 64, HMX chunked kernel) identical on topk-c and all: 16 fails each, pre-existing.
