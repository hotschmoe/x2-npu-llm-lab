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

## 2026-10-30 10:40 WRAP-UP (stall recovery by the coordinator at 09:50)
Branch `clover-port` tip bcce4b9 = merge of `clover` e695295 (rollback + trace + topk) with: 00d7f9b dirty-range, 82354b9 Q4_0 tile pad, d1055ae bounded OPPOLL spin, 826e193 GDN snapshot DMA,
e1b35dd test-backend-ops perf cases, 85ceeab graph cache + HOSTSTAT. Package of the tip: pkg-clover-port-final (host ggml-hexagon.dll AEE6CA2E, skel ADE7C252).

### The ab3b-3 hang (orphan, 09:16)
ab3b-3-allpoll log stops at "llama threadpool init" 12 s after start. Cause of the ORPHAN: I stopped background tasks of my own session (TaskStop / Stop-Job) at ~09:13-09:16; the old blk.ps1 launched
clover-run.ps1 in-process but the parent died, leaving the child running with no lock. Was the hang itself OPPOLL=1? Evidence says no: all 4 other OPPOLL=1 starts (ab3a #1 #3, ab3b #2, fin-c poll) came up fine, the killed run overlapped bf16's HTP0 runs for 34 min (two processes opening HTP0 / mapping ~20 GB each) and it stopped before the first HTP graph - the same signature as the earlier
two-process overlaps in this campaign. Not provable without a repeat; fin-c below repeats OPPOLL=1 on the final package (one clean run), and the new scripts kill their child on timeout.
Also: a 2 h 17 min "gc4096 0.031 tok/s" row (gca #2) was the laptop sleeping inside my lock hold (cpu.csv has ~600 samples in 8287 s) - discard it; the graph-cache A/B was redone in fin-c.
New launch discipline (coordinator item 3): clover-port/blk2.ps1, tbo.ps1, bench.ps1, clover-kld.ps1 all do Enter-NpuLock -> Start-Process + WaitForExit(timeout) -> finally kill the child -> Exit-NpuLock in ONE process.
(A killed parent can still leave a running child - Windows has no parent-death signal for Start-Process - but a stale lock is released by the 2-minute dead-owner rule and the child is bounded by RunTimeoutSec.)

### Gates on pkg-clover-port-final (clover merged with rollback)
| gate | result |
| --- | --- |
| clover-kld, new default reference (F32-dense CPU) | **KLD 0.019871, same-top 93.137%, PPL 15.083178** = bit-identical to m2/M0 baseline (0.019871 / 93.137%) -> GATE PASS |
| old reference (cpu-kld-base.bin) on dr / pad / all | 0.047923 / 89.216% / 15.083178 on all three = identical to topk-c |
| GATED_DELTA_NET_CACHE_FUSION (K>1: dst_cache + snapshots, SNAP kernel; env GDN_SNAPDMA 1 and 0) | 5/5 pass on both (all pkg) and on final (K>1 now really used by rollback) |
| GATED_DELTA_NET | 27/43 on topk-c, all and final - the same 16 K=1/n>=64 HMX-chunked failures (pre-existing) |
| MUL_MAT_ID (pad) | 796/796 |

### Feature table (Clover, HTP0, draft-self exact verify, 256 tok, ABBA in one lock hold, output hash identical where the algorithm is the same)
| feature | commit | speed vs topk-c (clean pairs) | verdict |
| --- | --- | --- | --- |
| incremental dirty-range merge | 00d7f9b | 5.121 -> 5.244 (+2.4%) | MERGE |
| Q4_0/IQ4_NL tile pad 640 B (contiguous weight DMA) | 82354b9 | dr 5.244 -> pad 5.281 (+0.7% more, +3.1% total) | MERGE (+11% weight RAM for Q4_0/IQ4_NL: Clover 18.2 -> ~20 GiB) |
| GDN snapshot DMA | 826e193 | op: K=6 -14%, K=16 -19% (590 -> 498 us / 660 -> 533 us); e2e only with rollback (see fin-b/fin-c) | MERGE |
| bounded OPPOLL spin + watchdog (code) | d1055ae | code only; the behaviour is the env: OPPOLL=1 = **+9.9%** (5.215 -> 5.729; second block below) | MERGE code; recommend OPPOLL=1 default on Windows |
| graph cache N entries + HOSTSTAT | 85ceeab | see fin-c | MERGE if fin-c shows hits (default stays 1 = old behaviour) |
| all of the above (default env, legacy loop) | - | 5.121 -> 5.237 (+2.3%); llama-bench pp16 +2.2%, pp256 +1.4%, tg32 +3.2% | |

### Host CPU / the dspqueue busy-wait finding (matters for the NPU-only goal)
* Host busy cores during generation: 0.98-0.99 in EVERY configuration (topk-c, dr, pad, all, OPPOLL 0/1, 0.5 ms hybrid).
* Per-thread census (clover-port/thr.ps1, 250 ms samples over the generation window only): ONE thread (main) burns 0.99 core, all 7 ggml worker threads 0.000;
  `-t 2 --poll 0` and `-t 1` change nothing (tok/s 4.70 vs 4.87, cores 0.99). Qwen3.6 plain decode on HTP0 - zero CPU fallback ops - shows the same 0.97 core on the main thread.
* Therefore the main thread is inside the Windows `dspqueue_read(..., DSPQUEUE_TIMEOUT_NONE)` (libcdsprpc) and that wait is a busy-wait: the HTP backend's idle-host goal cannot be reached with host flags.
  That also explains why OPPOLL=1 costs no extra core and is faster: the "blocking" call is not asleep, it just adds wake/poll latency on top; a 0.5 ms spin then blocking read (OPPOLL_SPIN_MS=0.5) recovers only half the gain (5.51-5.61).
* DSP is busy only ~25% of a Clover step (profile: 9.3 s of 36.7 s; 21k batches of 7.9 ops; ~575 splits per pass), so most of the remaining main-thread time is host work between batches (sched splits, CPU CPY, enqueue, kparams solve).
* Ideas for a real (event-based) wait, untested: (1) create the queue WITH a packet callback (dspqueue_create(..., packet_callback, error_callback, ctx) - today both are nullptr) which fires on a driver thread,
  signal a Windows event / std::condition_variable from it and let flush_pending WaitForSingleObject, then drain with dspqueue_read_noblock (already imported in htp-drv.cpp). Combine with a short spin
  (50-300 us) first so sub-ms batches keep the OPPOLL latency and long batches (MoE layers, 1-2 ms) sleep. (2) If libcdsprpc's callback thread itself spins, ask Qualcomm for a QueueEvent/semaphore
  variant or use FastRPC async (remote_handle_invoke_async + completion event). (3) Independently, cut the number of sync points: fewer CPU splits (bf16 agent) and the multi-slot graph cache (this branch)
  reduce the host work that keeps the thread running between batches.

### Final A/B on the merged tip (blk2.ps1, one lock hold per block, ABBA, AC, no overlapping model process, host cores 0.98-0.99 everywhere)
| block | config | baseline | final (clover-port tip) | note |
| --- | --- | --- | --- | --- |
| fin-a | legacy defaults (n_max 15, exact verify), baseline = pkg-clover-m2 (pre-rollback clover) | 5.165 5.117 (5.141) | 5.082 5.119 (5.101, -0.8%) | different algorithm (rollback is default in merged clover; stream hash differs: 05758780 vs 2B319F2D) - the port features roughly cancel the rollback's own -3% (rollback log: rs4 4.75 vs 4.91) |
| fin-b | rollback product config (LLAMA_SPEC_BONUS=1, --spec-draft-n-max 5), baseline = pkg-clover-rollback-rs6 (= clover tip without the port) | 7.199 7.208 (7.204) | 7.421 7.352 (7.387, **+2.5%**) | **identical output hash D9B44E9DCF in all 4 runs** (GDN snapshot DMA, tile pad, dirty merge are bit-exact end to end with K>1) |
Not completed in the time left (fin-c queued behind other agents; results are appended below only if they landed): OPPOLL=1 / GDN_SNAPDMA=0 / GRAPH_CACHE=4096 attribution inside the rollback config.

### What to merge / defaults to recommend
* Merge `clover-port` (all commits above) into `clover`: every code change is bit-exact (KLD/PPL identical, identical token streams in fin-b), self-contained (htp-dcache.c/main.c/htp-ctx.h; htp-tensor.h + htp-repack.h constants;
  gated-delta-net-ops.{c,h} + htp-kparams.cpp; ggml-hexagon.cpp flush_pending/graph_compute; test-backend-ops perf block "BEGIN/END port agent"). Expected conflicts with bf16: none in htp-fusion/kparams except one hunk in htp-kparams.cpp (GDN kernel selection) and ggml-hexagon.cpp env block.
* Env defaults: leave upstream defaults (OPPOLL=0) in the code, but for Clover runs on Windows set `GGML_HEXAGON_OPPOLL=1` (+10% on the legacy loop in ab3a: 5.729 vs 5.243/5.186; hybrid 0.5 ms spin +5.7%) - the host core is already 100% busy in the blocking wait, so it costs nothing;
  the 400 ms bound and the response watchdog make it safe against the old infinite-spin hang. One OPPOLL=1 start hung only inside the orphaned run (see verdict above); no hang in 4 other OPPOLL=1 starts.
  `GGML_HEXAGON_GDN_SNAPDMA` default 1 (needed by rollback), `GGML_HEXAGON_GRAPH_CACHE` default 1 (old behaviour; raise only after fin-c shows hits), `GGML_HEXAGON_HOSTSTAT=1` for diagnostics.
* Skipped (reasons): xp flat expert pipeline (M=1..4 only; Clover 1-row decode is ~94 ms of which the DSP is a minor part -> <0.5% of a step), GDN row-block kernel (-3.6% of one op), pipelined HMX MoE + 2 MB ddr_spad + exact gather
  (matmul-ops.c diverged by ~900 lines; 16-row MoE is 60+32 ms per draft pass of ~260 ms and near the DMA floor; the 2 MB spad only matters above 512 KB row maps, i.e. ub >= 512), fusions/replay (upstream already has the fusion set; replay is RFC-grade),
  Q8_0 contiguous kernels (not used by Clover), HTP_OP_MAX_BUFS 64 (no gain upstream; n-bufs already 64 here).
* Next ideas: event/callback wait for dspqueue (see above); fewer splits (bf16 CPY fusion) so the DSP share rises above 25%; multi-slot graph cache if fin-c shows hits; Q8_0/Q4_1 tile padding the same way (Clover down-proj is Q4_0 so not needed).

### fin-c (landed 11:25): attribution inside the rollback product config on pkg final (single runs, one lock hold, identical stream hash D9B44E9DCF in all four; HOSTSTAT=1 on all)
| arm | tok/s | note |
| --- | ---: | --- |
| snap1 (default) | 7.217 | (final without HOSTSTAT in fin-b: 7.421 / 7.352) |
| snap0 (GGML_HEXAGON_GDN_SNAPDMA=0) | 7.373 | within run-to-run noise (+-1.5%) of snap1: at K<=6 the e2e effect of the 14% op gain (~2 ms/pass) is below the noise floor; op-level gain is real (K=16 -19%) |
| **poll (GGML_HEXAGON_OPPOLL=1)** | **8.136** | **+10.1% vs final fin-b mean 7.387 (+12.7% vs the same-block snap1)**; clean start, no hang, same stream |
| gc4096 (GGML_HEXAGON_GRAPH_CACHE=4096) | 7.299 | no measurable gain; hoststat lines were not visible (backend INFO logs need -v) - a -v rerun (fin-d) was queued, see below if it landed |
=> Verdict on the ab3b-3 hang: OPPOLL=1 (or the 0.5 ms hybrid) started and ran cleanly in 4 of 4 other launches (ab3a #1, ab3a #3, ab3b #2, fin-c poll) so the orphan's startup stall was contention with the bf16 HTP0 run it overlapped, not an OPPOLL=1 bug.
Recommended product config on this machine: rollback + `LLAMA_SPEC_BONUS=1` + `--spec-draft-n-max 5` + `GGML_HEXAGON_OPPOLL=1`: 7.20 (rs6) -> 7.39 (port, +2.5%) -> 8.14 (+OPPOLL, +13% over rs6).

### fin-d (11:43, -v so the hoststat lines are visible; rollback config, 128 tokens, same stream hash 86206C17 in both): graph cache is a dead end
GRAPH_CACHE=1: 7.246 tok/s; GRAPH_CACHE=4096: 7.168 tok/s (noise). hoststat in both: **graphs hit 0 / miss 32000** (310743 ops, 9.7 ops per sub-graph), kparams solve+fusion-tag total only **152-195 ms** for the whole run,
dspqueue_read wait 16.7-16.9 s in 31991 reads (0.52 ms mean per batch); graph_compute total 56-60 s of which "enqueue" is almost all of it but it includes the one-time buffer mapping / first-batch costs at start-up, so it is not a per-pass number.
=> the sched assigns a NEW uid to every split on every re-split (ggml-backend.cpp: `sched->splits[i].graph.uid = ggml_graph_next_uid()`), so no uid-keyed cache can ever hit in Clover, and re-solving is negligible anyway.
Keep 85ceeab only for HOSTSTAT (diagnostics; the N-entry cache is harmless at default N=1); do not expect speed from it. The host-side cost between batches is in enqueue/sched/CPU splits, not in kparams.

### Closing state (11:45)
No process of mine is running and the lock is released. Packages: pkg-clover-port-{dr,pad,spin,all,gc,final} (final = tip bcce4b9). Worktree llama.cpp-hexagon-clover-port (branch clover-port). Nothing pushed.
