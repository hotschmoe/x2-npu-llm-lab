# Prefill campaign (Hexagon HTP0, Qwen3.6-35B-A3B Q4_0)

Started 2026-09-28. Separate from the decode/correctness campaign in `npu_campaign/STATUS.md`.
Harness: `npu_campaign/run-prefill-bench.ps1` (llama-bench, production env, no diagnostic HMX builds),
per-op summary: `npu_campaign/prefill_prof_sum.py` on a `PF_PROFILE=1 ... -Extra:--verbose` log.

## Baseline (installed pkg-wos, build 08618ff, HTP0, ub512, 1 rep)

| test | t/s |
| --- | ---: |
| pp16 | 71 |
| pp64 | 158 |
| pp128 | 244 |
| pp512 | 381 (345 with profiling on) |

Context: Llama-3.2-1B Q4_0 on the same stack does pp512 = 5081. specula CPU/OpenCL on this 35B: ~190/~210.
Qualcomm QAIRT Qwen3-4B w4a16: pp 2229 (dense, whole-graph, weight-resident; not the same model).
The "awful" prefill numbers in the decode campaign STATUS come from diagnostic HMX-correctness builds
(217 s for 13 tokens) and 4-17 token prompts, not from the production path.

## Where pp512 time goes (one 2024-op batch = ~1.4 s)

| op | share |
| --- | ---: |
| MUL_MAT_ID_NX (expert gate+up, hmx-tiled) | 36% |
| MUL_MAT_ID (expert down, hmx-tiled) | 24% |
| MUL_MAT dense (qkv/ssm_out/gate/attn) | ~7% |
| MUL_MAT ffn_gate_inp router (hvx-tiled, N=256) | 6% |
| GDN+CPY, ADD, CONCAT, MUL, SSM_CONV, SILU, ... | ~25% combined |
| flash attn | 0.6% |

Effective rate: dense HMX projections ~17 TFLOPS; expert matmuls ~1.0-1.4 TFLOPS (12-17x lower).
Weight traffic floor: 256 experts x 3 x 2048x512 x 0.5625 B = 453 MB/layer, ~18 GB/batch (~0.15 s at ~120 GB/s).

## Latency / stall check (npu_campaign/prefill_gap_sum.py)

Inside one 2024-op OPBATCH the summed inter-op idle gaps are ~0.5 ms of ~1400 ms (typical gap 0.1-1.2 us),
and op time sums to within 0.2% of batch wall. So at pp512 there is no dispatch/dependency stall on the DSP:
graph replay + opbatch already hide FastRPC cost. Op-duration histogram: 712 ops >=200 us, ~1000 in 50-200 us,
~300 in 10-50 us, almost none <10 us. Latency still matters elsewhere: small-M (pp16 = 71 t/s, ~225 ms per
16 tokens) where per-expert setup/dequant cannot amortize, and host<->DSP batch boundaries. Next: per-op
latency table at pp16/pp64 and per-stage (DMA wait / dequant / HMX / scatter) traces inside the expert loop.

## ub sweep (llama-bench, 1 rep)

| ub | pp512 | pp1024 | pp2048 |
| --- | ---: | ---: | ---: |
| 512 | 381 | - | - |
| 1024 | - | 460 | 456 |
| 2048 | - | 436 | 472 |

Bigger ub recovers ~24% then flattens near ~470: per-expert weight-dequant amortization is not the whole story.

## Stage trace inside expert ops (GGML_HEXAGON_PROFILE=3, `prefill_stage_sum.py`, measured batch)

Trace timestamps are 32-bit and wrap; script unwraps per thread and aligns to op clock. Per-op averages,
"sum over threads" against 8 HVX threads (800% = fully saturated):

| op (wall/op) | stage | thread-time/op | notes |
| --- | --- | ---: | --- |
| MUL_MAT_ID down (8.3 ms) | HVX_W_DEQUANT | 6.9 ms | |
| | HVX_A_PREP | 4.7 ms | activation gather+F32->F16 |
| | HVX_O_PROC | 3.8 ms | scatter |
| | INIT (thread 0, serial) | 3.7 ms = 44% of wall | per-expert setup |
| | DMA wait | 1.0 ms | |
| | **HMX_COMP** | **0.9 ms = 11% of wall** | |
| MUL_MAT_ID_NX gate+up (12.7 ms) | HVX_A_PREP | 25.6 ms | 2400 events/op for 305 chunks: ~8x redundant |
| | HVX_W_DEQUANT | 13.6 ms | |
| | INIT | 2.2 ms (17%) | |
| | DMA wait | 2.1 ms (17%) | |
| | HVX_O_PROC | 2.4 ms | |
| | **HMX_COMP** | **1.05 ms = 8% of wall** | |

Reading: HMX is idle ~90% of the expert ops. Phases (INIT, A_PREP, W_DEQUANT, HMX, O_PROC, DMA wait) run
back to back with thread barriers, so wall ~= sum of phase walls; HVX threads are only ~25-40% busy.
The win is removing/overlapping phases, not a faster multiplier:
- hoist activation F32->F16 conversion out of the per-expert loop (convert the 512x2048 rows once, 2 MB;
  experts then only gather F16 rows). Targets A_PREP (biggest NX phase).
- kill serial per-expert INIT (thread 0): precompute chunk plans/row maps once per op.
- pipeline dequant(i+1) || HMX(i) || scatter(i-1), prefetch next expert's weight DMA across expert boundary.

## Why experts are slow (matmul-ops.c `hmx_mm_id_2d_f32`)

- ~16 rows/expert at ub512 (512 x 8 / 256), padded to 32; each weight tile is dequantized (Q4_0 -> fp16, HVX)
  and used for only ~16 rows, vs 512 rows for dense projections.
- The per-chunk loop is fully serial: dequant (threaded HVX) -> wait DMA -> HMX push+pop (blocking) ->
  scatter output. `pipeline=false` for the id path; dense path overlaps HMX with dequant.
- Activation gather + F32->F16 convert is redone per expert.

## Goals

- G1: pp512 >= 600 t/s, pp2048 >= 800 t/s (from 381) without changing numerics of the production path.
- G2: expert matmul >= 4 TFLOPS.
- G3: learn best shapes: ub size, expert row batching, fused gate+up+SwiGLU+down, router on HMX, W4A8 vs dequant.
- Stretch: >= 1000 t/s pp512 (beats CPU 5x; approaches QAIRT-class).

## Experiments (ordered by expected value / cost)

1. ub sweep (512/1024/2048) - amortizes per-expert dequant. No code. Running.
2. Software-pipeline the MoE id path (dequant chunk i+1 on HVX while HMX runs chunk i; double-buffer outputs).
3. Fuse gate+up (already NX) with SwiGLU and feed down directly, skip DDR round trip of the 512x8x512 intermediate.
4. Router `ffn_gate_inp` (2048x256) onto HMX (6% for free).
5. Sort/bucket tokens so hot experts get full 32-row tiles; skip padding waste for tiny experts (HVX for cnt<=~4).
6. Prefetch next expert's weights across expert boundary (DMA today restarts per expert).
7. W4A8 HMX integer MAC (specula: int4xu8 confirmed on v81) to remove dequant entirely.
8. Pre-dequantized fp8/fp16 weight cache in spare memory for hot experts (bandwidth vs dequant tradeoff).
9. Small stuff: GDN/CONCAT/ADD/MUL fusion (~25%): CONCAT of conv state, per-layer ADD residuals.
10. Other runtimes/formats as controls: MXFP4 GGUF on HTP, OpenCL, QAIRT for a fixed expert-shaped GEMM.

## Log

- 2026-09-28: baseline + per-op profile above.
