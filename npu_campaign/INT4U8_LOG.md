# INT4U8_LOG.md (agent `int4u8`; append-only)

Question: can an int4-weight x u8/int8-activation path (or any low-bit path that avoids the fp16 dequant of Q4_0 weights) speed up MoE prefill expert
matmuls and/or decode on Hexagon v81? Worktree `llama.cpp-hexagon-int4u8`, branch `int4u8` (from `npu-htp-decode` ce75353), build dir `build-int4u8`,
packages `npu_campaign/pkg-int4u8-*`. Probe code is DSP-side `htp/hmx-int-probe.h` (compiled only with `GGML_HEXAGON_DIAG_DEFINES=GGML_HEXAGON_INT8_PROBE=1`,
OFF by default) + host driver `tests/test-hmx-int-probe.cpp`; python drivers `npu_campaign/hip_*.py`.

## 2026-09-29 21:00 Step 1: what the notes claimed vs what was measured

Claims (exact sources):
- `PREFILL_CAMPAIGN.md:148`: "7. W4A8 HMX integer MAC (specula: int4xu8 confirmed on v81) to remove dequant entirely."
- `DEEPGEMM_ASCEND_LEARNINGS.md:219`: "... our own logs (`PREFILL2_LOG.md:148` "W4A8 HMX integer MAC (specula: int4xu8 confirmed on v81)") suggest v81 HMX *does* support int4xu8 MAC in principle, untested in this codebase"
  and `:326`: "Whether Hexagon v81 HMX can do native low-bit MAC without HVX dequant: unknown, referenced as "specula: int4xu8 confirmed on v81" ... but not verified in `llama.cpp-hexagon`."
- Source of the claim, `../github/specula/docs/hmx_single_tile_probe_findings.md` (2026-06-16): header "## Result 1 - w4a8 integer MAC: **WORKS**" but the same document later says
  "## Systolic structure - the remaining wall (probe v5) ... the HMX integer matmul does **not** compute the naive `[M][K]->[M][N]` I assumed ... A single activation+weight pass contracts only ~4 deep,
  not 32 ... The int4 weight tile format is NOT flat `[n][k]` ... **Conclusion:** the tile *geometry* is solved ..., but the *systolic semantics* ... are not, and the weight mxmem tile format is unknown ...
  A guessed kernel would be wrong and unvalidatable, so the kernel stays `-1`-guarded". Also "Native w4a16 (int4 x fp16) is disproven".
  So "confirmed" only means: a structure-preserving integer MAC exists (uniform in -> uniform out, monotone ramps); no correct int4 x u8 product was ever obtained, and no throughput was measured.

Key new observation from the SDK header (`hmx_hexagon_protos.h`, Hexagon SDK 6.6.0.0 / tools 19.0.07): the specula probe used `Q6_weight_ubit_mxmem_RR` as "the int4 loader".
Its builtin is `__builtin_HEXAGON_M8_mxmem_wei_b1` and its assembly syntax is `weight.ubit=mxmem(Rs32,Rt32)`: a 1-BIT weight loader (sbit = `wei_sb1`). The loader family in the header is
`weight.{b,ubit,sbit,c,sc,sm,n,hf,f8}` (+ `:2x` for n, `:after/:single/:deep/:drop/:dilate` variants); by the naming the 4-bit one is `weight.n` (`Q6_weight_n_mxmem_RR`, plus `_2x`).
specula only ever paired `weight.n` with an fp16 activation (mixed path, "tiny values"), never with `activation.ub`. So the earlier "int4 x u8" experiments were 1-bit x u8, which explains "only ~4 of
32 lanes contract" style oddities. => the true int4 x u8 (weight.n x activation.ub) and int8 (weight.b x activation.ub) modes were UNTESTED. This is what this log tests.

Other HMX facts read off the header: activation loaders are `activation.{ub,hf,f8}` (f8 needs `__HMX_ARCH__ >= 81`? see below), readouts `mxmem(...):after[.retain][.sat].{uh:2x1,uh:2x2,ub,hf,ub:cm,hf:pos}`,
`cvt.{hf,ub,uh,f8}=acc`, per-output-column scale/bias tables via `bias=mxmem2(Rs)` (used by the fork with scale 1.0), `acc=mxshl(acc,#16)` (does NOT select on v81 in our toolchain:
LLVM ERROR "Cannot select: intrinsic llvm.hexagon.M8.mxaccshl").
Compiler constraint found: an activation load and a weight load must be in the same packet (LTO error "expected activate and weight pair"); the probe issues every MAC as one inline-asm packet.
Assembler check: all of `activation.{ub,hf,f8}` x `weight.{b,ubit,c,hf,n,sbit,sc,sm,f8,n:2x}` (+ modifiers) assemble for v81 (necessary, not sufficient: hardware must still execute them).

Existing fork code paths for reference (measured/grepped, not new): HVX Q4_0 x Q8_0 integer path already exists (`hvx-mm-kernels-tiled.h`, `Q6_Vw_vrmpyacc_VwVbVb`), used for decode / small M
(`opt_hmx_min_rows = 4`, `opt_id_hvx_max_t = 4` in `ggml-hexagon.cpp:102-103`, env `GGML_HEXAGON_HMX_MIN_ROWS`, `GGML_HEXAGON_ID_HVX_MAX_T`). Tiled Q4_0 weights are 576 B per 32x32 tile:
512 B nibbles + 32 fp16 column scales (`matmul-ops.h:23`), i.e. one Q4_0 scale per (32-deep K tile, output column).


## 2026-09-29 21:30-23:00 Step 2: on-device HMX probe (all numbers measured on this machine, X2 Elite, v81, NPU lock held)
Harness: `htp/hmx-int-probe.h` (+ `hmx-int-epi.h`), built only with `-DGGML_HEXAGON_DIAG_DEFINES=GGML_HEXAGON_INT8_PROBE=1` (default OFF); a MUL_MAT node with F16 src0 [1024x256],
F32 src1 [1024x256] and `op_params[7]==0x50495848` is not a matmul: src1 carries a tiny program + activation bytes, src0 carries raw weight bytes, dst gets raw VTCM readout + per-step
pcycle deltas; everything runs on the HMX queue thread. Host: `tests/test-hmx-int-probe.cpp`; drivers `npu_campaign/hip_*.py` (`hip_lib.py` job builder). Package `pkg-int4u8-probe6`.
Calibration: fp16 x fp16 all-ones tile reads 32.0 and an activation impulse at (m=3,k=5) appears in row 3 at halfword index (r/2)*64+2c+r%2 (activation tile = row-major [m][k], output and weight
tiles = the (r/2)*64+2c+r%2 layout used by the fork's "eye" tile). Core clock from HVX bench: 2.112 GHz (pcycles/us).

### 2a. Which HMX modes exist (hip_e1, act ones / weight pattern bytes, readouts)
- `activation.ub` x `weight.{b,n,n:2x,c,sc,sm,ubit,sbit}` all EXECUTE on v81 and give a structure-preserving integer MAC; the only readout that returns the integer accumulator is
  `after.uh:2x1` (`Q6_mxmem_AR_after_uh_2x1`): 1024 x u16 = (acc + bias) >> 1 (mod 2^16). `after.uh:2x2` writes half the positions; `after.hf/ub/pos.hf/cm.ub` read the FP accumulator (they return
  the bias table only, acc=0); v81 `cvt.hf=acc(Rs)`+`mxmem(..)=cvt` on an INT accumulator returns zeros for every Rs tried (0x3c00, 1.0f, 1, 0x10000, 0); `cvt.ub`/`cvt.uh` work but are 8/16-bit.
  => there is NO int32-accumulator -> fp16/fp32 hardware conversion; the per-column scale table (`bias=mxmem2`) applies to fp readouts only.
- Any readout that is not `:retain` clears the accumulator (second readout of the same accumulation returns bias only). uh21 adds the second 128 B vector of the mxmem2 table as an integer bias per column before the >>1.
- `activation.hf` x `weight.{n,b,c,sc,sm,ubit,f8}`: the weight loader kind is IGNORED, the tile is read as fp16 (identical outputs 0.00049/0.0198/2730/nan for bytes 01/11/55/FF for every kind).
  => native int4-weight x fp16-activation does not exist (confirms the earlier specula negative, now with the right loader `weight.n`).
- `weight.ubit/sbit` are 1-bit weights (ubit: value 0/1; sbit: 0 -> +1, 1 -> -1), `weight.c/sc` 2-bit, `weight.n` signed 4-bit two's complement, `weight.b` signed 8-bit, `weight.sm` behaves like an 8-bit sign-magnitude byte (0x01->16, 0x11->273, 0xFF->-2056). Not needed further.
- `activation.f8 x weight.f8` runs (39 cycles/pair at Rt=2047) but its hf readout is constant (2^-11) for all patterns: not investigated (irrelevant: a Q4_0 scale needs >3 mantissa bits).
- `acc = mxshl(acc,#16)` does not select in the v81 toolchain (LLVM "Cannot select" M8.mxaccshl).

### 2b. Exact semantics of activation.ub x weight.n (validated: 20/20 random 32x32x32 products bit-exact vs numpy, hip_s5)
out[r][c] = floor( sum_{k<32} A[r][k] * Q[k][c] / 2 )  as s16 (mod 2^16), where A = unsigned 8-bit, Q = signed 4-bit ([-8,7]).
- Activation tile: 2 KB in the fp16-tile layout (halfword index (r/2)*64 + 2k + r%2), the u8 value in the HIGH byte of each halfword (odd bytes); the low byte is a 1/256 fraction (constant fill v gives 257v/256:
  that was the "256/255" artifact of an earlier constant test; with low bytes = 0 the product is exact).
- Weight tile (n): 512 B = 4 HVX vectors; vector g holds k = 8g..8g+7, 32-bit word c = output column c, byte b of the word holds (low nibble: k=8g+b, high nibble: k=8g+b+4)
  (`hip_data/s5_kmap.npy`: e%8 -> k offset [0,4,1,5,2,6,3,7]); weight impulse response is 32 rows x 1 column.
- Negative sums read as one's complement of the truncated half (floor((acc-1)/2) for acc<0). Range: |acc| <= 32*255*8 -> |out| <= 32640 fits s16 for ONE k-tile; accumulating two k-tiles wraps mod 2^16.
- Output tile: u16 in the (r/2)*64+2c+r%2 layout; second mxmem2 vector = int bias per column (added before the >>1).
Pair timing depends on the Rt (byte range) argument, not on the weight width: see 2c.

### 2c. Rates (cycles per 32x32x32 pair; unrolled asm loops, distinct/same tiles, `hip_s2a`, `hip_s3a`)
| pair | Rt act/wei | cycles | ns | equiv |
| --- | --- | ---: | ---: | --- |
| fp16 hf x hf | 2047/2047 | 6.8-7.2 | 3.3 | 20 TFLOP/s |
| ub x b (int8) | 2047/1023 | 7.06 | 3.34 | same as fp16 |
| ub x n (int4) | 2047/511 | 7.08 | 3.35 | same as fp16 |
| ub x n / b / c ... | 2047/2047 (2 KB range) | 32.3 | 15.3 | 4.7x SLOWER: the wrong (too big) weight range makes the array load 2-4 weight tiles |
| ub x n:deep 2047/2047 | | 39 | | |
| ub:above x any | | 118-149 | | |
| f8 x f8 | 2047/2047 | 39 | 18.5 | (range not tuned) |
| clracc + int4 pair + uh readout | | 55 | 26 | mxclracc costs ~24 cycles; use the clearing readout instead |
| int4 pair + uh readout (no clracc) | 2047/511 | 6.7 | 3.2 | readout is free in the pipeline |
So the integer modes run at the fp16 MAC rate (not 2x), with 4x (int4) / 2x (int8) less weight data through the array.
(Correction of my own first result: the first sweep with Rt=2047 for every weight kind said "integer path is 4.7x slower" - that was the oversized range, not the array.)

### 2d. HVX peak (hip_hvx, 1..8 threads, per-thread 2.0 cyc per vrmpy, scales perfectly to 8 threads)
- vrmpyacc u8 x i8 -> i32 (128 MAC/instr): 2 cycles/instr/thread = 135 GMAC/s/thread, 8 threads 1.08 TMAC/s = 2.16 TOP/s (identical for i8 x i8).
- vmpyacc u8 x i8 -> i16 pair: same 2.16 TOP/s. qf16 vmpy+vadd: 1 cycle/instr (2 instr per cycle) 1.08 TMAC/s; native hf x hf -> sf pair (Q6_Wsf_vmpyacc_WsfVhfVhf): 0.54 TMAC/s.
- HMX fp16 = 32768 MAC / 6.8 cyc = 20 TFLOP/s = 10 TMAC/s: 9x the HVX int8 peak, before any unpack/scale/convert work.

## 2026-09-29/30 Step 2 (cont.): the cost that decides it - dequant vs int epilogue for ONE Qwen expert matmul (M=32 rows, N=512, K=2048 = 1024 weight tiles), 8 HVX threads, best of 7
(`hip_s4a.py`, `htp/hmx-int-epi.h`; the dequant is the fork's own `dequantize_tiled_weight_to_fp16_task_q4_0`, 640 B aligned Q4_0 tiles in VTCM)
| stage | time | note |
| --- | ---: | --- |
| Q4_0 -> fp16 tile dequant (HVX, 8 thr) | 5.8 us | 95 cyc/tile/thread; independent of the number of rows M |
| HMX fp16 chain (1024 pairs + 16 readouts) | 3.5 us | 7.2 cyc/pair |
| HMX int4 chain, MAC + clearing readout per (n,k) tile | 3.2 us | 6.7 cyc/pair (Rt 2047/511); with explicit mxclracc 26.7 us (55 cyc/pair) |
| HMX int4 chain, 64 k-tiles accumulated, 1 readout/n-tile | 3.2 us | only valid without per-k-tile scale |
| epilogue v1: hf accumulate, column scale | 10.6 us | 16 live accumulators spill |
| epilogue v2: v1 + per-(k-tile,row) activation scale | 10.8 us | |
| epilogue v3: fp32 (qf32) accumulate, column scale only | 5.3 us | = dequant cost, no saving |
| epilogue v4: fp32 accumulate, column + per-(k-tile,row) scale (Q8_0-style blocks) | 21.3 us | 3.7x the dequant |
| epilogue v5: fixed-point s16 scale x s16 readout -> s32 `vmpyacc` (2 vector ops per output vector) | 3.4 us | best case, single per-token activation scale, s32 overflow risk |
The epilogue is unavoidable: Q4_0 has one fp16 scale per (32-deep k tile, output column); the int accumulator can only be read as u16 (no hardware int->fp16 conversion, 2a),
so every k tile must be read out (free on the HMX side) and scaled/accumulated on HVX. Epilogue cost is proportional to the number of row pairs present (M), dequant cost is not:
epilogue(M) = E32 * ceil(M/2)/16 (best case: only the used row-pair vectors are processed).

### Break-even (per expert matmul, 8 threads; E32 = epilogue at 32 rows)
| epilogue variant | E32 | rows M* where epilogue == dequant (5.8 us) |
| --- | ---: | ---: |
| v3 fp32, col scale | 5.3 us | 35 |
| v4 fp32, col + row (Q8_0 block) scale | 21.3 us | 8.7 |
| v5 fixed-point (best case) | 3.4 us | 55 |
Rows per expert: sum over experts of M = 8 x T (T tokens in the ubatch, top-8 routing), 256 experts => mean M = T/32. Summed over a layer/matrix: dequant = 256 x 5.8 = 1485 us (all experts active),
v5 epilogue = 3.4 x 8T/32 = 0.85 T us => 15% / 30% / 60% / 120% of dequant for T = 256 / 512 / 1024 / 2048.
Upper bound on the wall-time gain: prefill2's marginal-cost probe (`PREFILL2_LOG.md` section 1b: doubling the dequant stage costs +110 ms of a 1340 ms pp1024/ub1024 pass, 8.2%; HMX
doubling +82 ms) => removing ALL dequant would gain at most 8% at ub1024. The int path only removes (1 - epilogue/dequant) of it: v5 best case 40% x 8.2% = ~3% at ub1024,
~10% at ub512 (pass 770 ms), ~120% cost at ub2048 (loss); v3 ~0-1%, v4 a loss for every M >= 9. Costs NOT included in these bounds: per-token activation quantization (absmax + u8 pack
into the halfword-high-byte layout), the HMX readout write traffic (the doubled-HMX probe shows the pipeline is VTCM-contention-sensitive and the int path moves the same 2 KB per tile-step
through VTCM in the other direction), s32 accumulator overflow handling in v5, the extra weight repack (nibble layout above) and weight scale tables.

### Decode (T = 1..4)
Expert/dense matmuls with M <= 4 already run the HVX Q4_0 x Q8_0 int8 path (vrmpy, `hvx-mm-kernels-tiled.h`, thresholds `opt_hmx_min_rows`/`opt_id_hvx_max_t` = 4). One weight tile (1024 weights)
per token is 8 vrmpy = 16 cycles per thread; 8 threads consume a 576 B tile in ~1 ns = ~600 GB/s of compute-side appetite vs the ~120 GB/s DDR link, so decode is bandwidth-bound; at M=4
compute ~ DMA (which is why the threshold is 4). HMX int4 cannot help: a 32-row tile is 8x wasted at M<=4 and the k-tile epilogue makes it worse than HVX. Nothing to gain.

## 2026-09-30 Step 3/4 and DECISION
Measured HVX-only alternative (real model, installed-line build `pkg-merged2-m1`, Qwen3.6-35B-A3B Q4_0, pp256 ub256, `hip_hvxab.py`, interleaved, one llama-bench rep per point):
MoE expert matmuls on the HMX dequant pipeline (default, `GGML_HEXAGON_ID_HVX_MAX_T=4`) vs forced onto the existing HVX Q4_0 x Q8_0 int8 vrmpy path (`ID_HVX_MAX_T=4096`):
round 1: 542.4 vs 28.8 t/s (first HVX run after other agents' work: outlier, not repeated), round 2: 565.3 vs 287.4, round 3: 561.3 vs (not obtained: the NPU lock was held by another agent for >40 min; run abandoned). HMX mean 556 t/s (3 rounds), forced-HVX 287 t/s (1 valid round).
=> the HVX int8 path reaches ~0.29 TMAC/s effective on the experts (~27% of the 1.08 TMAC/s vrmpy peak measured in 2d) and is ~2x slower than the HMX pipeline at 8-16 rows/expert; at the
prefill row counts of interest (32+ rows/expert) the gap only widens (HVX cost scales with rows, HMX does not).

DECISION: NO-GO for a prefill/decode kernel (no int4 x u8 path implemented; nothing installed, no package with behaviour change). Evidence:
1. The hardware does NOT lack the mode: v81 HMX has an exact int4 x u8 (and int8/int2/int1 x u8) 32x32x32 MAC at the fp16 rate (7 cycles/pair), with a 4x smaller weight tile (512 B) and
   a fully decoded layout (2b). The earlier "confirmed on v81" was a 1-bit loader mix-up; the real mode works. It is NOT faster than fp16 (no 2x), and there is no int->fp16 accumulator conversion.
2. What kills it is Q4_0's per-(32-deep k tile, column) fp16 scale: every k tile must be read out (u16, 2 KB) and scaled/accumulated on HVX. That epilogue costs 3.4 us (fixed-point best case),
   5.3 us (fp32, one activation scale per token) or 21 us (fp32, Q8_0-style per-block activation scales) per 32-row expert matmul, versus 5.8 us for the fork's Q4_0->fp16 dequant that it
   replaces, and the epilogue grows linearly with the rows M while the dequant does not. Break-even rows M* = 55 / 35 / 9. Ceiling on end-to-end prefill gain (dequant is 8.2% of a
   pp1024/ub1024 pass, prefill2 marginal probe): <= ~3% at ub1024 in the best variant (which needs single per-token activation scales = worse numerics than the CPU Q8_0 reference, plus an
   s32 overflow hazard), ~10% at ub512, negative at ub2048; realistic value after activation quantization and VTCM contention is ~0.
3. HVX-only (vrmpy) alternatives lose outright: 2.16 TOP/s peak (8 threads) vs 20 TFLOP/s HMX; measured 2x slower end to end even at 8-16 rows per expert.
4. Decode is bandwidth-bound and already uses the HVX Q4_0 x Q8_0 path for M <= 4; HMX int4 has nothing to add there.
Not done because the go/no-go says no: activation-quantization numerics gate (would be judged against the CPU reference PPL 12.2015, whose MUL_MAT_ID with Q4_0 weights already quantizes activations to
Q8_0 blocks), negative controls, A/B on pp1024. If someone retries: the only lever left is a fused epilogue that halves the HVX cost again (v5 is 2 vector ops per output vector; the floor is the
vector load + vmpyacc), which would still cap at ~3% at ub1024. Do not retry unless the dequant share of a pass grows well beyond 8%.

Artifacts: worktree branch `int4u8` (llama.cpp-hexagon-int4u8) commit 7426053 (probe code only, OFF by default, tests/test-hmx-int-probe.cpp); drivers npu_campaign/hip_*.py; raw scan data
npu_campaign/hip_data/s3b.npz + s5_kmap.npy (s2b.npz used act value 1 and was deleted); packages pkg-int4u8-probe6 (probe skel, diagnostic only). Nothing was installed and pkg-wos/npu-htp-decode were not touched.
