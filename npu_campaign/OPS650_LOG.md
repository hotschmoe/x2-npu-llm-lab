# ops650 log (append-only)

Agent ops650. Worktree `llama.cpp-hexagon-ops650` (branch `ops650` from `npu-htp-decode` ce75353). Build `build-ops650.cmd` (+ `build-ops650-htp.cmd` once after the first configure),
package `pkg-ops650.ps1 -Name <x>` -> `pkg-ops650-<x>`. Analysis script `ops650_prof.py` (profile logs). Queue scripts `ops650-q*.sh` (every run takes the NPU lock via lockrun.py).

## 2026-09-29 21:30  Setup + first look at an existing M=1 profile (old package, `mtp-speed/prof-plain-hello16.detail.log`, 14 warm batches of 1384 ops)
- Whole token is ONE dspqueue message (opbatch 3072): the per-op cost is DSP-internal, not FastRPC. Inter-op gaps by cycle stamps ~0.03 ms/batch: the op loop
  (proc_op_req prep) is INSIDE each op's measured time.
- batch 33.65 ms = sum(op cycles) 33.54 + 0.11 (start/end whole-cache flush).
- Non-matmul/FA/GDN ops: 923/batch, 8.83 ms, median 7.05 us (compute for 2048-float ops is <1 us => ~6 us is fixed cost per op).
- 651 ops < 8 us, 270 ops 8-12 us. Ops <= 12 us: 921 ops = 6.8 ms (20% of the token).
- Matmul-like ops (431/batch): fit t(us) = 19.5 + 7.12 us/MB (marginal 140 GB/s) R2 0.99; by size bucket: 2-8 MB ops (166/batch) 81 GB/s, 8-32 MB ops (94/batch) 91 GB/s,
  the single 418 MB head matmul 141 GB/s. So mid-size matmuls run well below the streaming rate: either compute-bound in the HVX dequant/dot or start/tail latency per op.
- Spoof hooks added (OFF by default): `GGML_HEXAGON_OPS650_SPOOF=<mode>` (host sets op flags; DSP `htp/ops650-spoof.h`): see ggml-hexagon.cpp comment for modes 1-10.

## 2026-09-29 22:20  FIRST MEASUREMENTS (pkg-ops650-s1/s4, Qwen3.6-35B-A3B Q4_0, plain decode, Hello, PROFILE=1 => host spins/logs, DSP 'fast' state)
- Host side (untraced, pkg s1, run ops650-h-A_base-r1, 128 tok): 27.35 tok/s = 36.57 ms/token; DSP batch 35.70 ms (ONE dspqueue message per token, 1384 ops, opbatch 3072), host-observed
  submit->response 36.18 ms => host prep+enqueue+wake ~0.85 ms/token (2.3%). FastRPC call count = 1 per token; no call-to-call gap problem on the DSP side.
- DSP per-op profile (ops650-b-plain, 31 warm batches): batch 34.62 ms; op-loop gap between op end and next op start ~0.03 ms/batch, i.e. the per-op framework prep is inside the op time.
- SPOOF cycle run (ops650-cyc-r1; modes cycled every 3 tokens, garbage outputs, research only). DSP batch ms per token (1384 ops):
  | mode | what runs | batch ms |
  | 0 | real | 34.6 (profile-fast state; first 6 batches slower) |
  | 1 | every op skipped (prep + profile record only) | **10.43** => 7.5 us per op of pure framework cost |
  | 6 | every op = empty work-queue fork/join | 10.95 => fork/join costs only 0.38 us per op |
  | 2 | matmul/FA/GDN real, everything else skipped | 32.69 |
  | 4 | ... non-big ops = empty fork/join | 33.02 (+0.33) |
  | 5 | ... non-big ops = 16 KB DDR->VTCM->DDR DMA round trip | 33.26 (+0.57 => ~0.6 us per DMA round trip) |
  | 3 | matmul/FA/GDN skipped, everything else real | 12.37 |
  Conclusion so far: ~10.4 ms of the 34.6 ms token (30%) is per-op framework overhead in proc_op_req/process_opbatch (7.5 us per op incl. cache maintenance and profile timers),
  NOT kernel setup, NOT fork/join, NOT DMA latency, NOT FastRPC. Mode 7 (non-big = fork/join + DMA) crashed the DSP on its 3rd batch (garbage ids -> OOB expert), expected for spoofed data.
- Next: per-op segment timing (prep / src flush / dst flush+dirty / barrier / execute) added under profile=1 (tag HTP_PROF_SEG_TAG, `seg a,b,c,d,e,total` in the profile-op line) and
  spoof modes 11-13 (no per-op cache maintenance) to attribute the 7.5 us.
- Matmul roofline notes (b-plain): Q6_K head 417 MB @141 GB/s; Q8_0 ssm_out @125; Q4_0 qkv/gate @88-98; Q4_0 experts NX @82, Q4_1 down @69 GB/s (Q4 dequant-bound, int4u8 agent's area).

## 2026-09-29 22:35  Per-op SEGMENT timing + no-flush spoof (pkg-ops650-s5, run ops650-cycB-r1; profile=1 DSP state; decode batches of 1384 ops)
Segments of proc_op_req (us per op, means): prep = param memcpy + src loop; src flush = htp_tensor_flush_all(src); dst flush+dirty = htp_tensor_flush_all(dst)+htp_tensor_dirty_all(dst); execute = kernel.
| class (per batch) | prep | src flush | dst flush+dirty | execute | total |
| non-matmul/FA/GDN ops, real (923) | 0.42 | 0.94 | **3.60** | 5.62 | 10.60 |
| matmul/FA/GDN ops, real (461) | 0.38 | 1.60 | **4.80** | 54.5 | 61.3 |
- Cache-maintenance framework cost = 0.94+3.60 = 4.5 us per small op, 1.6+4.8 = 6.4 us per big op => 923*4.5 + 461*6.4 = **7.1 ms of the 34.6 ms token (20%)**.
- Spoof 13 (real kernels, per-op flush/dirty bookkeeping removed, outputs unchecked/garbage) = 29.2 ms vs 34.6 => upper bound of the cache-maintenance saving 5.4 ms/token (15.6%). 
- Spoof 11 (skip kernels + no flush): 4.03 ms per token for 1384 ops => bare per-op loop cost 0.37 us on big / ~0.6 us small (prep + profile timers). Fork/join 0.38 us/op, DMA round trip ~0.6 us/op.
- Interpretation: the dst side (flush_all(dst) + dirty_all) dominates. dirty_all is O(64) scans + O(A*64) merge_dirty_ranges twice per op, and once the 64 slots fill (steady state: every DMA-written dst is conservatively marked dirty) every new dst
  EVICTS an old range with a flush (WQ fork/join when > 4 KB). Next: split dst flush_all / dirty_all / merge inside proc_op_req (pkg s6) and count active ranges.
- Mode 8 (matmul weight streaming spoof) crashed the DSP in the cycle run (mode 7 did earlier); not needed for the conclusions above, deprioritised.

## 2026-09-29 22:55  ROOT CAUSE of the framework cost + fix 1 (incremental dirty-range merge)
Split of the dst side (pkg s6, ops650-seg-r1, small/big ops): dst flush_all 0.40/1.14 us, **dirty_all 3.45/3.92 us of which merge_dirty_ranges 2.82/3.21 us (called 1.8-2.0x per op)**.
htp_tensor_dirty_all re-ran an O(64*64) full merge_dirty_ranges over the 64-slot table twice per op. That is 2.8-3.2 us of EVERY op = 1384*3.0 = ~4.1 ms of the 34.6 ms token.
Fix (htp-tensor.c, commit aedf830 on branch ops650): merge_into(k) absorbs only the neighbours of the slot that was just inserted/grown; scans bounded by a new ctx->dirty_hi
(1 + highest slot used since the last reset; reset next to every memset of dirty_ranges). Same set of dirty ranges (union semantics identical; only slot positions differ), eviction path untouched.
Result (pkg s7, ops650-seg7, profile mode): dirty_all 3.45/3.92 -> 0.19/0.24 us; batch 34.62 ms -> **29.9 ms (-4.7 ms, -13.6%)**; skip-all batch 10.8 -> 5.74 ms.
Remaining framework cost per op now: prep 0.4, src flush 0.56-1.4, dst flush 0.29-1.0, dirty 0.2 (small/big) ~ 1.7/3.0 us => ~3 ms/token.
Gate + interleaved untraced ABBA queued (ops650-q8: g2 gate vs ref1 = pkg-merged2-m1 plain logits; ab1 base vs s7).
Also built (pkg s8): SSM_CONV+WIN decode window rebuilt with vgather/vscatter instead of the 32-channel scalar loops (48 us/op x 30 = 1.4 ms/token before); same values/MAC order.

## 2026-09-29 23:00  Profile of pkg s8 (= tracker fix + vgather/vscatter SSM_CONV+WIN), ops650-prof8, profile mode, 23 warm batches of 1384 ops
- batch 34.62 (s1/base) -> **28.81 ms** (-5.8 ms, -16.8%). Non-matmul small ops: median 7.34 -> 4.11 us, 9.30 -> 5.25 ms/batch. SSM_CONV+WIN 48.8 -> 10.5 us/op (30 ops: 1.46 -> 0.32 ms).
- matmul fit t = 15.7 + 7.09 us/MB (was 19.5 + 7.11): marginal 141 GB/s; big-op remaining cost is kernel pipeline, not framework.
- NOTE gate line "GATE g4 ref1 vs s8+ALL ... FileNotFoundError" in ops650-gates.out is INVALID (queue killed/restarted before its run existed); the real g4 gate is re-run on pkg s9.
- Correctness gates (bit-identical full-vocab plain logits vs pkg-merged2-m1, Hello 128 rows): base hooks build s3 with all opt-ins OFF: identical (g1 'off'); ref vs ref repeat: identical. s7 (tracker) / s8 (+conv) / s9 (+conv silu opt-in) / s10 (+ flush WQ threshold 32 KB): queued.

## 2026-09-29 23:15  GATES (Hello, 128 rows, full-vocab plain logits, compare_logits_traces --require-identical vs pkg-merged2-m1 = ref1; ref1==ref2 repeat identical)
| candidate | result |
| s3 hooks, all opt-ins OFF | 128/128 bit-identical |
| s8 = tracker fix + vgather/vscatter SSM_CONV+WIN | **128/128 bit-identical** (g3) |
| s9 + ALL opt-in fusions ON (GDN_STATE, RMS_NORM_SCALE, SHEXP_EP, SILU+MUL->SWIGLU, SILU folded into SSM_CONV+WIN, MOE_TOPK alias=1) | **128/128 bit-identical** (g4; the log line says 's8+ALL' but the package was s9: label typo in the queue script) |
The four matchers that were disabled on 2026-09-26 because 'greedy text changed' (GDN_STATE, RMS_NORM_SCALE, SHEXP_EP, MOE_TOPK alias) are bit-exact once the plain path is deterministic (decode2's 64 B flush fix).
Still to run on the final package: 3 prompts, MTP arms, PPL 12.1307/13.3331, edge check 6/6, interleaved ABBA.

## 2026-09-29 23:40  Clean warm decomposition of one decode token (profile mode = 'spin/fast' DSP state; only warm batches; small = non-matmul/FA/GDN ops, big = matmul/FA/GDN ops)
| | baseline (s6, old tracker) | s7 (incremental dirty merge) |
| batch (ms) | 35.08 | 29.89 |
| framework (prep + src flush + dst flush + dirty), small ops (923) | 4.78 | 1.51 |
| framework, big ops (461) | 3.25 | 1.41 |
| kernels, small ops | 4.73 | 4.75 |
| kernels, big ops | 22.08 | 21.97 |
Framework 8.03 -> 2.92 ms per token. Later steps: vgather conv window (-1.15 ms of small kernels), fusions (-390 ops), see below.
Other notes: (1) RMS_NORM_SCALE fused kernel ended memcpy'ing through uncached VTCM (13.3 us/op vs 3.5+3.2 unfused): fixed with hvx_copy_f32_aa (s11). (2) the historical 'text changed' fusions are exact.
(3) SSM_CONV+WINK (MTP verify) was 66 us/op x 30 = 2.0 ms: same scalar-loop structure as WIN, vectorised in s12 and WINK SILU fold + MOE+TOPK for 2..4 rows in s14 (opt-in bits 18/19).
(4) NEGATIVE CONTROL for the vgather conv kernel: GGML_HEXAGON_OPS650_NEGCTRL=1 (pkg s13+) swaps a tap plane; the plain-logit gate must FAIL (queue q16).

## 2026-09-30 00:05  NEGATIVE CONTROL for the vgather conv-window kernel: PASS (gate fires)
GGML_HEXAGON_OPS650_NEGCTRL=1 (pkg s13: swaps the b/d tap plane gather offsets in SSM_CONV+WIN) vs the same reference (pkg-merged2-m1, ref1): compare_logits_traces exit 1, first_bit_difference=1
(row 1: the first decoded token after the prompt row). So the bit-exact logits gate does detect a wrong conv-window kernel; the correct kernel (s8/s9/s12) passes it.
