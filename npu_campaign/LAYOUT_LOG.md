# LAYOUT log (append-only) - agent layout
Worktree llama.cpp-hexagon-layout (branch layout from npu-htp-decode 56a66d1), build-layout, pkg-layout-*.

## 2026-09-30 23:45 Algebra (paper)
- Layout A (vec s byte 4*row+b: lo=q[row][8s+b], hi=q[row][8s+4+b]): decode and/lsr give natural vrmpy groups, 0 permutes (baseline 5/128B); prefill vdeal(-2)+2 vunpack = 3 permutes (same as baseline vshuff+2 vunpack).
- Layout L1 (vec i halfword 2row+e nibble m = q[row][2(4i+m)+e]): prefill = halfword shift/and only (0 permutes, +3 ALU/128B); decode 0 permutes but needs act groups [k0,k4,k1,k5]/[k2,k6,k3,k7].
- Pair packaging (2 tiles = 9 aligned vectors, scale vec halfword 2row=tile e,2row+1=tile o): contiguous 576 B/tile, combined scales via one hf*hf->sf vmpy pair + vmux (no permute).
- Microbench: htp/layout-q4.h, layout-bench.h, env GGML_HEXAGON_LAYOUT_BENCH=1 + GGML_HEXAGON_PROFILE=1 (results as seg words of following ops). Build was killed once by RAM exhaustion (not mine), rebuilt 23:43.

## 2026-10-01 00:20-01:30 Microbenchmark results (DSP, VTCM resident, K=2048 tile row = 64 tiles, 64 reps x min of 7 trials, pcycles; run lb1..lb7 in mtp-speed/, every variant checked bit-exact against the baseline kernel output)
Dot kernels, cycles/tile (packets/tile): single HVX thread, 32 rows x 32 k Q4_0 tile.
| kernel | 32x1 | 32x2 |
| baseline (tiled_vec_dot_q4_0_32x1/2, 640 stride, 5 permutes per 128 B) | 65.0 (26.5) | 109.1 |
| layout A, 640 stride (and/lsr only, natural act) | 59-74 (alignment noise, same code differs 20% by buffer placement) | 76.4 |
| layout A or L1/L1x, PAIR packaging (contiguous 576 B/tile) | 48.4 (16.2) = -26% | 60.6 = -44% |
| pair, no vsub (diagnostic) / int-sum only / both | 44.4 / 33.5 / 27.6 | |
| L1x pair (nibbles stored n^8, xor+sub sign extension, swizzled act) | 47.4 | |
=> removing the permutes gives -26% (M=1) / -44% (M=2) per tile; of the remaining 48 c/tile, ~15 are the float scale math (int->f32, 2 mul, add), ~4 the vsub, the rest vrmpy/and/lsr/loads. The pair packaging also fixes the odd-tile misalignment of contiguous 576 B/tile DMA (qwen4's q4_0c pair kernel measured ~90 c/tile).
Prefill HMX dequant (nibbles -> fp16 tile), cycles/tile (packets/tile), must reproduce the baseline fp16 tiles bit-exactly (all variants below do):
| baseline | 90.2 (46) |
| A (vdeal(-2) + 2 vunpack, natural act) | 130-132 (63): worse, same permute count but more packets |
| L1 (halfword shifts, 0 permutes), 640 / pair | 105-110 |
| L1 batched arrays | 284 (spills) |
| L1x (n^8 stored, vasl+vasr sign extension), 640 / pair | 102 / 101 |
| L1x pair kernel, 2 tiles per call (also with vmpyi instead of vasl) | 97 (45 packets) = +7% vs baseline |
| native hf multiply (Q6_Vhf_vmpy_VhfVhf) | no gain (111 / 103) |
| A' = byte layout + Q6_Whf_vcvt_Vb (no vunpack) | 104, 51 packets, output WRONG (vcvt lane semantics differ); no gain either way |
=> prefill dequant is instruction-count bound (4 slots/packet): the permutes are not its limiter, so removing them does not help; best layout (L1x pair2) costs +7% dequant cycles. Decode and prefill cannot BOTH improve: the decode-optimal layouts cost >= +7% on prefill dequant.

## 2026-10-01 01:20 Prefill SPOOF A/B (GGML_HEXAGON_LAYOUT_SPOOF_DQ=1, pkg-layout-b8, garbage outputs, timing only): Q4_0 HMX dequant runs the L1x pair2 kernel (97 c/tile) on baseline-layout bytes
Round 1 (layab, one pair only, other rounds were killed by ram-guard because a foreign llama-cli started in the same seconds): pp1024 ub1024: off 776.2 / on 746.7 (-3.8%), consistent with +7% dequant cycles. More rounds: see below (lab2).

## Decision (01:30)
Step (3) criterion not met: decode dot -26% (M=1) / -44% (M=2) per tile, but prefill dequant is not "not worse" for ANY layout that helps decode: best +7% dequant cycles = -3.8% pp1024 (spoof A/B). End-to-end implementation NOT done. Decode gain is also only realisable together with contiguous (1D) weight DMA (qwen4 Q4C), because the baseline 640-stride 2D-DMA path is DMA-bound already (compute-only 65 c/tile < DMA time 71 c/tile at 136 GB/s/8 threads).
Projection if combined with qwen4's contiguous DMA (their numbers, ms/token): dense MUL_MAT 8.77 -> ~7.7, ID_NX 3.78 -> 2.83, MUL_MAT+ADD 2.42 -> 2.03, NX 1.23 -> 0.96 = about -2.7 ms of 24.8 (~ +11% tok/s over qwen4 s12), at a cost of about -4% prefill.
Handoff spec (ready kernels in llama.cpp-hexagon-layout/ggml/src/ggml-hexagon/htp/layout-q4.h, branch layout c9d9a2e): layout L1x pair = per 2 k-tiles a 1152 B block: vectors 0-3 / 4-7 nibbles of tile 2j / 2j+1, vector i halfword (2*row+e) nibble m (bit 4m) = q[row][2*(4i+m)] ^ 8 stored as (q^8); vector 8 halfword 2*row = scale(2j,row), 2*row+1 = scale(2j+1,row). Decode act group bytes for source vector i: lo-nibble group [k8i, k8i+4, k8i+1, k8i+5], hi-nibble group [k8i+2, k8i+6, k8i+3, k8i+7] (byte permutation of the quantised activation vector within each 8-byte chunk, done once per activation). Needs n_k_tiles even (K % 64 == 0).
Wild-hair items, reasoned from the numbers (not built): HMX for the M=1..4 decode matvec needs the HVX dequant first (90 c/tile) which is more than the whole current dot (65 c/tile): no. vlut for the nibble expansion: removes only the vsub (4 c/tile of 48): no. vcvt Vb->hf (no vunpack): more packets, wrong lane order, no gain. native hf multiply: no gain (2 uops).

## 2026-10-01 03:50 Prefill SPOOF A/B result (2 pairs; heavy NPU contention and ram-guard kills prevented more rounds)
pp1024 ub1024 reps 3, pkg-layout-b8: pair 1 off 776.2 / on 746.7 (-3.8%); pair 2 (off lab2-off-r3, on lab3-on, not adjacent) 777.7 / 751.0 (-3.4%). Spoof = outputs garbage, timing only. Conclusion unchanged: the decode-optimal layout costs ~3.5% prefill.
Branch layout c9d9a2e (llama.cpp-hexagon-layout), packages npu_campaign/pkg-layout-b1..b8 (b8 = final). Nothing installed/merged; all code default OFF (env GGML_HEXAGON_LAYOUT_BENCH, GGML_HEXAGON_LAYOUT_SPOOF_DQ).
