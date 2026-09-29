# prefill2 log (append-only)

Agent `prefill2`. Worktree `llama.cpp-hexagon-prefill2` (branch `prefill2` from `npu-htp-decode` ee4381b). DSP build: `build-prefill2-htp.cmd`
(-> `build-prefill2-htp`), package: `prefill2-cand.ps1 -Name X` (-> `prefill2-cand-X`), A/B: `ab-prefill2.ps1 -Variants base,X`
(labels `p2ab-*`/`dbl-*`), host build: `build-prefill2-host.cmd` (-> `build-prefill2-host`). `mk_prefill2_scripts.py` generated the copies.
Variant sweep helper: `build-p2-variants.sh name=value ...` (writes `htp/pf_probe.h`, builds, packages).

## 2026-09-29 09:30  0. Setup
Rebuilt the DSP skel from the branch tip without changes -> `prefill2-cand-base` (SHA 6866F5DD; the installed one is 37AD9661, different
because of build path strings/timestamps; same source). All A/B below compare against `base` (and `installed` where noted).

## 2026-09-29 09:40  1. Marginal-cost probes of the pipelined expert kernel (candidate 1)
Idea: instead of "spoofing" (removing work, which garbles the routing downstream) do the reverse and keep the result bit-exact: run ONE stage
twice per item (PF_DBL bitmask in `htp/pf_probe.h`: 1 dequant, 2 HMX, 4 weight DMA, 8 scatter, 16 gather; duplicated work is idempotent).
If doubling stage X costs +dT at the whole-pass level, X is on the critical path with weight dT; 0 = slack. Variants dbl1..dbl16 vs base,
pp1024/ub1024, interleaved.

## 2026-09-29 09:45  2. Router refine puzzle SOLVED (candidate 4)
Hypothesis: the old refine (exp 11) recomputed logits with `vec_dot_f32_f32_aa_1x1`, whose lane reduction (`hvx_vec_reduce_sum_f32`: rotate
by 4,8,..) is a DIFFERENT summation tree from the one the HVX router kernel (2x2) uses (`hvx_vec_reduce_sum_f32x2`: vshuff, then rotate
from VLEN/2 down). Same inputs, same per-lane accumulation, different last-bit rounding -> not bit-identical logits -> PPL moves by +-0.04
(the pipeline is that sensitive: the NPU-vs-CPU spread is 0.07).
Change (`patch_router_refine2.py`, now generalised to any K%32==0, N%32==0<=256 F32 router, incl. Gemma 2816x128): recompute the top-16
candidates with `vec_dot_f32_f32_aa_2x1` (the x2 tree). Self-test with PF_ROUTER_K=256 (recompute ALL logits, HMX output ignored):
**PPL 12.1307 exactly** (bit-identical to the HVX router) -> diagnosis confirmed. K=16 (production): PPL 12.1611, 0.78 s/pass
(HVX router 0.86-0.88). K=16 is not bit-identical only because the 240 non-candidate logits stay fp16-accurate and feed the softmax
denominator (cancels in the normalised top-8 weights up to fp32 rounding, i.e. ulp-level weight noise -> chaotic PPL shift of 0.03,
CPU reference 12.2015). Selection itself is exact (top-8 is inside the exact top-16 recompute).

## 2026-09-29 10:10  1b. Result of the marginal-cost probes (pp1024/ub1024, llama-bench 2 reps; r1/r2 mean; base 764 t/s = 1340 ms/pass)
| doubled stage | t/s | +ms/pass | reading |
| --- | ---: | ---: | --- |
| dequant (1) | 706 | +110 | biggest single stage on the critical path (24% of the 457 ms of experts) |
| HMX job (2) | 720 | +82 | HMX is only ~28% busy, yet doubling it costs 82 ms: VTCM/HMX contention with the HVX dequant writes, not HMX capacity |
| weight DMA (4) | 733 | +56 | DMA has slack but is not free |
| scatter (8) | 738 | +48 | |
| gather (16) | 735 | +53 | |
Sum of the marginals is ~350 of the 457 ms: the pipeline is uniformly loaded, no single bottleneck to remove (this is why the LUT
dequant, which cut instructions, could not help: the dequant cost is dominated by VTCM stores/contention, not ALU issue). (dbl8/dbl16 have
1 run each; ab process was stopped to free the NPU queue for candidates.)
Stage trace at ub512 (PROFILE=3): per expert op HVX threads are ~75% busy (dequant 317-348% of 800%, A_PREP 115-165%, O_PROC 53-160%),
HMX 28-36% busy, "HVX-active" 86% of the op wall; conv op is compute-bound (HVX_COMP 88% of wall: the SiLU sigmoid costs ~100 vector
instructions/32 lanes), so DMA tiling of the conv cannot help; GDN is 67% active.
Ideas that follow: (a) balance the gather/scatter across threads (contiguous split left threads 4..7 idle for experts with <=16 rows,
tested as `rrpair`); (b) hybrid HVX-vs-HMX routing for small-M (below).

## 2026-09-29 10:15  3. Fused multi-row router top-k (candidate 5): MOE+TOPK for prefill
The decode agent's `MOE+TOPK` (SOFT_MAX+ARGSORT+GET_ROWS+SUM_ROWS+CLAMP+DIV in one op, 1 token) generalised to T>=32 rows (rows dealt
over the worker threads, per-thread scratch) and n_exp 128 (Gemma 4). Host matcher `try_fuse_moe_topk`, DSP `moe_topk_worker`
(argsort-ops.c). Result: PPL 12.1307 exactly (bit-identical), 0.83 s/pass vs 0.86-0.88 (ub512 PPL run). `PF_TOPK_SELECT=1` (8-round max-select instead of
the 8-vector bitonic sort, writes only the first 8 indices) built as `topksel`; PPL/A-B pending.
