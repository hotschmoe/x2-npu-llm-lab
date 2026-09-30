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

## 2026-09-30 01:35  FIRST UNTRACED PAIR (plain decode, Hello, 128 tokens; NOT yet interleaved - base r1 ran 00:17, s14F r1 ran 01:29; ABBA rounds 2/3 queued in ops650-q18)
ops650-ab2-base-r1 (installed pkg-merged2-m1): 28.00 tok/s (35.72 ms/token).  ops650-ab2-s14F-r1 (s14 + all opt-in fusions): **35.69 tok/s (28.02 ms/token), +27.5%**.
Do not quote as a result until the interleaved rounds finish.

## 2026-09-30 01:36  MTP GATE (g8): PASS. pkg s14 + every opt-in fusion (GDN_STATE, RMS_NORM_SCALE, SHEXP_EP, SILU+MUL, conv-window SILU fold incl. WINK, MOE+TOPK for 2..4 rows, MOE_TOPK alias) vs pkg-merged2-m1,
`--spec-type draft-mtp --spec-draft-n-max 1`, LLAMA_MTP_FUSED_CATCHUP=1, LLAMA_MTP_DRAFT_VOCAB=65536, Hello 128 tokens: target rows 128/128 bit-identical (compare_logits_traces --require-identical),
draft candidate rows 72/72 bit-identical (--context-index 1 --draft-greedy).

## 2026-09-30 02:08  g9: pkg s15 = s14 with all gated fusions ON BY DEFAULT (no env; MOE_TOPK alias default on): plain logits vs pkg-merged2-m1 (ref1), Hello 128 rows: bit-identical.
Untraced plain decode so far: base r1 28.00; s14F r1 35.69, r2 35.47 (base rounds 2/3 pending).

## 2026-09-30 04:06  UNTRACED PLAIN-DECODE A/B (Hello, 128 tokens, --spec-type none, seed 1234, temp 0; runs spread over 3.5 h because of NPU contention, so order is base r1, fused r1, fused r2, base r2, base r3, fused r3)
| variant | runs (tok/s) | mean |
| installed pkg-merged2-m1 (base) | 28.00, 26.56, 26.49 | **27.02** |
| pkg-ops650-s14 + all gated fusions (s14F; == s15 defaults) | 35.69, 35.47, 34.02 | **35.06** |
Gain +29.8% on the means (+21% vs the best base run, +37% vs the worst); per-run session drift on base is +-3% (the upstream agent measured base plain at 26.5-26.6 in the same window).
ms/token: 37.0 -> 28.5. All comparisons that count are same-package-family runs in the same window; log files mtp-speed/ops650-ab2-*.

## 2026-09-30 04:25  PPL gate 4x512 (run-prefill-ppl.ps1, pkg s14 + all gated fusions incl. MOE_TOPK alias, tracker fix, vgather kernels): **12.1307 +/- 1.11278 (exact)**

## 2026-09-30 04:40  PPL 16 chunks: 13.3337 +/- 0.63454 (expected 13.3331 / 0.63451) with s14 + all gated fusions + MOE_TOPK alias env = 1 => NOT exact (4x512 was exact)
Suspects on the prefill path: MOE_TOPK alias (prefill rows >= 32 now fused with the alias rule), RMS_NORM_SCALE (prefill rows), tracker/flush-threshold. Bisect queued (ops650-q20: off / fusions without alias / alias only, 16 chunks).
Pre-emptive change (s16): alias default ON only for <= 4 rows (decode / MTP verify), prefill keeps the old overlap rule; explicit GGML_HEXAGON_MOE_TOPK_ALIAS=1 still allows it everywhere. q21 runs s16 defaults for 16 chunks.

## 2026-09-30 05:19  fusion edge check (run-fusion-edge-check.ps1, pkg s14, all gated fusions on): 6/6 greedy transcripts identical (6/40/130/260/520/900 words, OPFUSION=1 vs 8062).

## 2026-09-30 05:23  PPL 16-chunk bisect (expected 13.3331 +/- 0.63451)
| package / config | PPL |
| s14, all fusions OFF, alias 0 (tracker + vgather kernels + WQ 32 KB only) | **13.3331 +/- 0.63451 (exact)** |
| s14, all gated fusions ON + GGML_HEXAGON_MOE_TOPK_ALIAS=1 for every row count | 13.3337 +/- 0.63454 (5e-5 relative: NOT exact) |
| s16 defaults (all gated fusions ON, router top-k alias only for <= 4 rows, no env) | **13.3331 +/- 0.63451 (exact)** |
=> the tracker fix and the conv kernels are exact on the prefill path; the alias router top-k fusion is exact for decode/MTP rows (logits gates) but perturbs prefill rows (>= 32) slightly, so the default now
restricts the alias to <= 4 rows. (The alias-only run was not needed and was cancelled.) 4x512 with s14+alias was 12.1307 exact; a 4x512 run on s16 defaults is queued.
s16 defaults 4x512 (no env): PPL 12.1307 (exact, run ops650-ppl4-s16def). So s16 defaults are exact on BOTH prefill PPL gates (12.1307 / 13.3331).
g6: second prompt ('Explain how a hash table handles collisions, with a concrete example.') plain full-vocab logits, 128 rows: s14 + all fusions vs pkg-merged2-m1: bit-identical.
Gate summary so far (all vs pkg-merged2-m1): plain logits Hello x (s3/s8/s9/s15) + hash prompt (s14+ALL) bit-identical; MTP target 128/128 + draft 72/72 (s14+ALL); PPL 12.1307 / 13.3331 (s16 defaults);
fusion edge check 6/6 (s14+ALL); negative control fires; ref-vs-ref repeat identical.

## 2026-09-30 12:05  STATUS AT HAND-OFF (the NPU has been held by other agents for ~6 h; the tail of the queue is still running unattended)
Done and gated (all vs pkg-merged2-m1 = installed): see gate list above. FINAL PACKAGE = `npu_campaign/pkg-ops650-s16` (branch `ops650` of llama.cpp-hexagon, worktree llama.cpp-hexagon-ops650, head b7f0f4e;
branch point npu-htp-decode ce75353, which has not moved): tracker fix + vgather conv windows + flush threshold + RMS_NORM_SCALE copy fix + all gated fusions ON by default
(`GGML_HEXAGON_OPS650_FUSE=0` turns the five opt-in matchers off, `GGML_HEXAGON_MOE_TOPK_ALIAS=0` the router alias).
Still queued/running (ops650-q18.sh, ops650-q24.sh; results will land in mtp-speed/ops650-abm-*, ops650-h2-*, ops650-gates.out; ops650-finisher.sh appends a summary to this log when q18 ends):
 - MTP interleaved ABBA base vs s14F (draft-max 1, fused catch-up, dv 65536), 3 rounds; host knobs (OPBATCH 1280, OPPOLL=1) one round; g10 = plain gate of s16 defaults.
 - delete npu_campaign/mtp-speed/ops650-g1-ref1.logits.bin (254 MB) after g10.
Not done: MTP speed numbers (only bit-exact gates + profile-mode op times, which are confounded by session drift), per-layer MTP fusions for GDN state (T=2), GDN-gates chain fusion, 3+ prompt gates on the
final package (2 prompts + Hello done), prefill speed A/B of the tracker fix.
Not worth doing (measured): (1) persistent-kernel / mega-fused layer for DISPATCH reasons - empty fork/join is 0.38 us/op and the DMA round trip 0.6 us/op; after the tracker fix the whole remaining framework
is ~1.7 us/op (small) / 3 us/op (big) = 2.9 ms of 29 ms; the recoverable upper bound from removing every per-op fixed cost incl. kernel prologues is ~6 ms (22%), ~9 ms (33%) if matmul pipeline fill/drain
(15-18 us fixed per matmul op, 6.5-7 ms) were also hidden by cross-op weight prefetch, which needs kernel cooperation in the int4 matmul code (other agents' area) - realistic 3-4 ms.
(2) DMA-only exemption from dirty tracking: only ~0.5 ms left to gain and it reopens the stale-cache-line hazard class decode2 fixed.
(3) more small-op fusions: each removes only ~4-5 us/op now; GDN gates chain (5 ops x 30 layers) would be ~0.6 ms.
