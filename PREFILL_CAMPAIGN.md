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

## Dev loop (no system install needed)

- `npu_campaign/build-prefill-htp.cmd` (DSP-only v81 build; env `PF_BLD` = build dir, `PF_CMAKE_EXTRA` = cmake flags)
- `npu_campaign/prefill-cand.ps1 -Name X -Build <dir>` packages + signs a candidate lib dir (`prefill-cand-X`)
- `run-prefill-bench.ps1 -LibDir npu_campaign\prefill-cand-X` uses it via ADSP_LIBRARY_PATH; installed pkg-wos untouched
- `run-prefill-ppl.ps1 -LibDir ...` correctness gate: perplexity, 4x512-token chunks on `ppl_text.txt`
- Clean rebuild of unmodified source (`base`) reproduces the installed baseline: pp512 385 (vs 381).

## Experiment 1: cross-expert software pipeline (`GGML_HEXAGON_MMID_PIPE`, htp/matmul-ops.c `hmx_mm_id_pipelined`)

Flat (expert, weight) work list; raw-weight DMA for item i+1, HMX for item i, output scatter for item i-1 all
overlap dequant of item i; activation F32->F16 gather done once per expert and shared by gate+up (NX);
double-buffered raw/fp16-weight/act/out in VTCM; experts with > m_max (<=256) rows fall back to the old path.
Off by default in CMake; single-device (mdev.count == 1) only.

| | base | pipe |
| --- | ---: | ---: |
| pp128 | 228 | 348 |
| pp512 | 385 | 485 |
| PPL (4x512) | 12.1307 | 12.1307 (bit-identical output stat) |
| perplexity s/pass | 1.44 | 1.08 |

## Experiments 2-4 (same `GGML_HEXAGON_MMID_PIPE` option), interleaved A/B, pp512, 3 rounds x 3 reps

| variant | what | pp512 (mean) | vs base |
| --- | --- | ---: | ---: |
| base | clean rebuild, option OFF | 348-385 (machine drift between sessions) | - |
| pipe | cross-expert pipeline (Exp 1) | 436 | +25% |
| pipe2 | weight DMA sliced over per-thread queues | 450 | +29% |
| pipe3 | one worker dispatch per item (scatter(i-2)+gather+DMA+dequant fused) | 453 / 488 | +28% |
| pipe4 | + l2fetch of gather rows | 489 (no change vs pipe3) - reverted | 0 |
| pipe5 | pipe2 code (NOT the fused dispatch; see note) + ddr_spad 512 KB -> 2 MB (removes per-op memalign of the ~1 MB row-map table) | 551 | +43% |

Findings:
- Same-skel run-to-run and session-to-session drift is 5-10%; only interleaved multi-round A/B is trustworthy.
- DMA trace events span push->completion, not wait time; weight streaming was not the limiter (~24 GB/s used).
- Serial INIT on thread 0 (expert-ID scan/mapping table) was 20-44% of expert-op wall; the mapping table is
  n_experts * max_rows * 8 B (up to ~1 MB) but ddr_spad was 512 KB, so every op did memalign+free. 2 MB removed it.
- Gather (A_PREP) is 13 us/thread for ~2 rows but l2fetch did not help end to end.
- PPL (4x512 tokens) stays 12.1307 +/- 1.11278 on pipe, pipe2, pipe3.

Correction: pipe5 was built after a `git checkout` that discarded the uncommitted pipe3 (fused dispatch) patch, so
pipe5 = pipe2 (per-thread DMA slices, separate gather/scatter dispatches) + 2 MB scratchpad. pipe3's fused dispatch
alone was +/-0 vs pipe2 in the interleaved A/B; `patch_mmid_fuse.py` re-applies it (tested as pipe6, below).

## Extra results (pipe5)

- ub sweep pp2048 (base -> pipe5): ub512 371 -> 545; ub1024 454 -> 588; ub2048 503 -> 546.
  At ub2048 more experts exceed m_max=256 rows and fall back to the old serial path, so pipe5 regresses;
  fix is to chunk hot experts into row-chunks inside the pipeline instead of falling back.
- `GGML_HEXAGON_ROUTER_HVX=0` (router matmul 2048x256 on HMX, FP16 operand rounding): pp512 552 -> 605 (+9.7%);
  PPL 12.1307 -> 12.1627 (+0.26%, inside the +/-1.1 stat error) - expert selection can differ; opt-in only.
- Per-op profile of pipe5 (per pass ~930 ms): experts 373 (40%), router 87, dense mm ~92, linear-attn group
  (GDN+CPY 60, CONCAT 37, SSM_CONV 32, SILU 24) ~150-170, ADD/MUL 81, rest small.

## Experiments 5-6: fused dispatch re-test and row-chunked experts

- pipe6 (= pipe5 + fused per-item dispatch, `patch_mmid_fuse.py`): pp512 550.7 vs 551.5 - neutral (kept in source).
- pipe7 (= pipe6 + `patch_mmid_chunk.py`): experts with > m_max (256) rows are split into row-chunk items inside
  the pipeline instead of falling back to the legacy serial loop.
  pp2048: ub1024 585 -> 614 (+5%), ub2048 555 -> 578 (+4%). PPL ctx2048 (2 chunks) base 8.2042 = pipe7 8.2042;
  time/pass 4.37 s -> 3.64 s.
- Best config so far: pipe7, ub1024, ~614 t/s (base at ub1024: 454, +35%); with ROUTER_HVX=0 (opt-in) add ~+10%.

## Log

- 2026-09-28: baseline + per-op profile above.
- 2026-09-28: Experiment 1 (pipelined MoE expert kernel): pp512 385 -> 485 (+26%), pp128 +53%, PPL identical.
- 2026-09-28: pipe2/3/4/5: fused dispatch, l2fetch (no gain, reverted), ddr_spad 2 MB (+13%). pp512 ~385 -> ~551.
- 2026-09-28: pipe6 (fused dispatch, neutral), pipe7 (row-chunked hot experts): pp2048 ub1024 614 t/s, PPL identical.
