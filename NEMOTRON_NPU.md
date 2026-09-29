# Nemotron 3 (Mamba2 + MoE hybrid) on the Hexagon NPU - `mamba` agent report (2026-09-29)

Code: `llama.cpp-hexagon` worktree `llama.cpp-hexagon-mamba`, branch `mamba` (from `npu-htp-decode` ee4381b). Log: `npu_campaign/MAMBA_LOG.md`.
Models: bartowski Nemotron-3-Nano-30B-A3B Q4_0 (18.6 GB, `models/gguf/nemotron3-nano-30b-a3b-q4_0/`) and the retarget
ggml-org NVIDIA-Nemotron-3.5-Lightning-30B-A3B Q4_0 (18.9 GB, `models/gguf/nemotron35-lightning-30b-a3b-q4_0/`, plus its 1.16 GB MTP GGUF).
Both are arch `nemotron_h_moe` (52 blocks = 23 Mamba2 + 23 MoE + 6 attention); the fork already had the model code. Config diff Nano vs 3.5-Lightning:
identical dimensions and layer pattern, only +1 MTP (nextn) layer; every kernel assumption below holds for both.

## Results (HTP0, llama-bench `-b 2048 -ub 1024`, default install path + `GGML_HEXAGON_ROUTER_HVX=0`)

| | CPU (8 thr) | first NPU run (SSM_SCAN on CPU) | + HVX SSM_SCAN | + HMX SSD | + Q5_1->Q8_0 | + router on HMX, fusions (final) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Nano pp512 / pp1024 (t/s) | 137 / - | (CPU scan, 71 splits) | 391 / 440 | 496 / 554 | 756 / 889 | ~800 / ~950 |
| Nano tg64 (tok/s) | 34.5 | 18 (12-token run) | 24.0 | 23.3 | 31.7 | ~33 |
| 3.5-Lightning pp512 / pp1024 / tg64 | 172 / - / 32.8 | | | | | 816 / 970 / 27.5 |

Same-session cross-model (one package `pkg-mamba-dskip`, llama-bench ub1024, round 1 of `mamba-xmodel.ps1`; Nemotron with router-HMX, Qwen/Gemma with the campaign default):

| model | pp512 | pp1024 | tg64 |
| --- | ---: | ---: | ---: |
| Nemotron-3-Nano-30B-A3B Q4_0 (this work) | 810 | 954 | 31.7 |
| Qwen3.6-35B-A3B Q4_0 | 682 | 757 | 28.3 |
| Gemma-4-26B-A4B QAT Q4_0 | 796 | 885 | 29.0 |
| Nemotron-3.5-Lightning Q4_0 (ggml-org, Q8_0-heavy) | 816 | 970 | 27.5 |

(Only one round: the second was cut for time; Gemma/Qwen numbers are lower than the overnight summary's because this session is under heavy NPU/RAM contention from the other agents.)

The ONLY unsupported ops were `SSM_SCAN` and `MUL_MAT` with Q5_1 weights (shared experts of 12 of 23 MoE layers in the bartowski GGUF).
Everything else (ReLU^2 as RELU+SQR, sigmoid router + bias, ADD_ID-free routing, grouped RMS norm, SSM_CONV, MUL_MAT_ID q4_0/q4_1) already ran on HTP0.

## What was built
1. `htp/ssm-scan-ops.c`, `GGML_OP_SSM_SCAN` (Mamba-2, scalar decay per head; Mamba-1 stays on CPU):
   * exact HVX kernel (`GGML_HEXAGON_SSM_SCAN=1`): job = (seq, head), 64x128 fp32 state DMA'd into VTCM (double buffered across jobs), sequential token loop,
     B/C rows in registers, 4-row batched lane reduction. Supports K>1 rollback snapshots. NMSE vs CPU 1e-14.
   * chunked SSD on HMX (`=2`, default, T >= 64, d_state 128, head_dim 64): per chunk of 64 tokens and <= 8 heads of a group: HVX prep (softplus/cumsum/decay vectors,
     fp16 tile packing, fp32 state kept transposed in VTCM), HMX `G = C B^T` (shared per group), `B^T` via identity multiply, `Y_intra = M X`, `Y_inter = C S^T`,
     `dS = B^T (w x)`, HVX finalize. Prefill scan 24.2 ms -> 4.9 ms per layer at T=1024 (4.9x). fp16 operands, fp32 accumulate and fp32 master state.
   * fusions: SSM_SCAN + state CPY (scan writes the cache slot), SSM_SCAN + skip connection (`y + x*D`), RELU+SQR unary, prefill expert-combine for 6 experts.
2. `Q5_1` weights: converted to Q8_0 tiles at upload (`GGML_HEXAGON_Q5_1=0` disables). Found late: it hid ~0.7 s/pass of CPU matmuls (+70% prefill when fixed).
3. Test cases: Nemotron shapes in `tests/test-backend-ops.cpp` (decode, PP-64/512/1024, odd, multi-seq, K=3) + perf cases; `TBO_PRINT_ERR=1` prints NMSE; `TBO_SSM_HMX_TOL=1` relaxes HTP to 5e-7 for the HMX path.
4. Spoofed/negative controls (OFF by default): `GGML_HEXAGON_SSM_SCAN_SPOIL=1` (decay*1.01, both paths), `=0x100|mask` (skip SSD phases, timing only).

## Correctness
* Oracle `test-backend-ops -b HTP0 -o SSM_SCAN` (17 cases): exact path 17/17 at the strict 1e-7 (NMSE 6e-15..3e-14); HMX SSD 17/17 at 5e-7 (NMSE 2.2e-7..2.3e-7, the fp16 operand floor;
  the existing CPU/CUDA SSD test tolerance is 2e-7); NEGATIVE CONTROL 0/17 in both modes.
* Nano, 4x512 tokens of ppl_text.txt: CPU PPL 16.0833; NPU exact-scan 16.1429, +SSD 16.1546, +Q5_1->Q8_0 16.1680, +router-HMX 16.1310 (unchanged by the SCAN+CPY, skip and RELU+SQR fusions).
  KL vs frozen CPU logits: KLD 0.0050-0.0053 for every config, same-top 96.4-96.8%: the SSD path, the Q8_0 repack and the HMX router are invisible against the NPU-vs-CPU floor.
* State carry across ubatches (c=2048, ub 512/1024): CPU 11.3601, NPU 11.3557..11.3657.
* Greedy CPU vs NPU, 5 prompts x 48 tokens: identical for the short/medium prompts, first divergence at near-ties for the others (see MAMBA_LOG.md); logit-level agreement is the KLD above.
* Qwen3.6 gates on the mamba build: PPL 12.1307 (4 chunks) and 13.3331 (16 chunks) exact.

## Where the time goes (Nano, pp1024, ~1.06 s/pass, everything on HTP0)
MUL_MAT_ID experts 472 ms (44%, HVX dequant limited: ~6 TFLOP/s vs ~18 for dense HMX matmuls), other MUL_MAT ~180, SSM_SCAN 113, ADD/MUL/SWIGLU/RELU/SQR/SILU/CONV ~250 (elementwise).
Decode: ~2.1 GB (Nano) / ~2.65 GB (3.5-Lightning, mostly Q8_0) of weights per token at ~100 GB/s; 29 ms DSP time vs 20 ms bandwidth floor, the rest ~650 small ops.
MTP: the 3.5-Lightning MTP GGUF (attn + MoE nextn block) loads in the fork (`-md ... --spec-type draft-mtp`, CPU smoke ok); acceptance/speed not measured on the NPU.

## Known limits
fp16 range in the SSD path (|C.B| dt < 65504, |state| < 65504; not observed); SSD needs d_state 128 / head_dim 64 / K=1 (else exact path); Mamba-1 on CPU; Q5_1 requant is approximate.
