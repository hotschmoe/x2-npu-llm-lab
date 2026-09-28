# Hexagon-specific harness: format and experiment backlog

Research/to-do snapshot, 2026-09-27. This is documentation, not an implementation decision or permission to resume the paused testing campaign. The harness would complement llama.cpp as an independent, narrow measurement/correctness lane, not silently replace the model or serving path. See [EARLY_ENGINE_REUSE.md](EARLY_ENGINE_REUSE.md) for Specula/Trident salvage and the proposed host/transport design, [GEMMA_MOE_HEXAGON_FIT.md](GEMMA_MOE_HEXAGON_FIT.md) for the alternate-model shape audit, and [STATUS.md](STATUS.md) for exact run artifacts and installed-binary provenance.

## Why build it; what is known now

- Qwen3.6-35B-A3B Q4_0 has a fast historical Hexagon path near 28.4-29 tok/s single-stream decode (`htp-qwen-replay.log.err`, `htp-qwen-replay2.log.err`, `topk-alias-control.log.err`), but output/coherence and exact-logit qualification remain incomplete. A newer arithmetic-diagnostic path is slower: an OFF-default exact-byte Q8_0 activation conversion improved its untraced A-B-B-A decode from 76.37/76.81 to 70.26/70.32 ms/token (13.02/13.09 to 14.22/14.23 tok/s; `q8_0-untraced-abba-a1-a.detail.log` through `q8_0-untraced-abba-a2-a.detail.log`). Both 128-token candidate logit traces match its scalar control bit-for-bit, and a separate 64-token prompt also matches. These paths have different controls and cannot be ranked as one clean A/B.
- Short-prompt prefill evidence is not a qualified long-prompt rate: a 17-token code prompt reported 72.04 tok/s on the historical fast path (`router-gate-1-code-candidate.log.err`); a one-repetition `llama-bench` pp4 measured 51.90 tok/s (`bench-batch.log`). The newer diagnostic path spends about 131 seconds on a 13-token prompt, with HMX preparation identified as a large cost (`q8_0-untraced-abba-b1-a.detail.log`, scalar-hoist stage profile in STATUS). Do not call that model-load time or conflate it with decode.
- One-token routed expert gate/up is a true shared K2048 F32 activation row and eight selected expert IDs, not eight redundant activation rows. The traced F32-to-Q8_0 conversion was about 40-42 us per worker before the exact-byte vector fix. Q8_0 is primarily the activation operand for Q4_0-weight HVX byte-dot decode (W4A8); it is not another copy of the entire model. Q4_1 expert-down uses Q8_1, and other mixed weight types may use still other activation contracts.
- The public v81 HMX reference describes FP16 32x32 cells/2 KB tiles. Current backend eligibility sends one-token GEMV mainly through HVX and multirow prefill toward HMX. The public document's FP16 example is **not** proof that every X2 path lacks native INT4; inspect SDK/QNN capabilities and measure. Current HVX source unpacks nibbles into bytes and uses `vrmpyacc` byte-dot instructions.
- Corrected sizing: GGUF Q4_0 is 18 bytes/32 weights = 4.5 bits/weight. The *packed* DSP Q4_0 tile is 576 bytes/32x32 weights = also 4.5 bits/weight. The 640-byte number is the Q4_1 packed tile **and** the Q4_0 aligned destination/stride, not the Q4_0 packed payload. Thus no demonstrated 11% Q4_0 model-storage saving is available by merely retitling it INT4. Investigate whether alignment affects actual DMA/VTCM traffic, but the source says packed bytes are copied into a padded destination; use counters before claiming a traffic saving. Local source: `llama.cpp-hexagon/ggml/src/ggml-hexagon/htp/matmul-ops.h` tile-size constants and `matmul-ops.c` Q4_0/Q4_1 repacked dispatch.

## Format decision to test, not assume

Start with **Q4_0-equivalent symmetric INT4 values and FP16 block-32 scales** in a DSP-native tile layout for a one-layer harness. Keep the decoded weight values and activation contract fixed initially: this is a lossless relayout experiment, so expect exact packed-value agreement and, for the same arithmetic order, bit-identical output. Explore signed two's-complement nibble encoding, scale placement, expert interleaving, DMA alignment and on-demand unpack as separate factors. Signed nibble encoding might remove the present subtract-8 after unpack; it could also cost more sign-extension/shuffle work. The answer comes from a kernel microbenchmark, not from the word INT4.

Treat operand precision separately from the weight file:

| Lane | Candidate | Why test | Main caveat |
| --- | --- | --- | --- |
| One-token decode | Q4_0-equivalent W4A8, HVX byte dot | Existing fast path and exact Q8_0 quantizer evidence | Unpack, activation quantization, scales and DMA still cost time |
| Multirow prefill/verification | W4A16-like unpack-to-FP16, HMX | Public v81 HMX FP16 tile path and larger effective M | Extra unpack/VTCM work; one-token M does not become 8 merely because top-8 experts are selected |
| Alternate INT4 | symmetric or asymmetric blocks 32/64/128, per-channel or per-group scales | Trade scale traffic/accuracy against SIMD-friendly layout | Requantization changes the model; zero-point correction and outliers can erase savings |
| AutoRound INT4 | offline learned rounding/scales with a chosen INT4 export | Potential quality-per-byte improvement on the same source checkpoint | AutoRound is **not** a Hexagon instruction or automatic runtime rounding; export and kernels must still be implemented/measured |
| FP4/codebook | MXFP4, NVFP4, IQ4_NL or custom 16-entry LUT | Different quality/scale tradeoffs; existing local MXFP4 HVX LUT is a useful control | LUT/scale conversion may outweigh nominal bytes on this chip |

AutoRound specifically is Intel's post-training quantization toolkit: it optimizes quantization parameters/rounding offline, supports W4A16 and multiple exports, and can be evaluated with block-size and mixed-precision choices. Ask "does an AutoRound-quantized INT4 checkpoint preserve more quality at the same *measured* Hexagon bytes/time?", not "does INT4 autoround in hardware?". It will not be bit-identical to the existing GGUF Q4_0 model unless the exported quantized values and all arithmetic happen to coincide. Its official repository says the method uses sign-gradient descent and distinguishes optimized/pure round-to-nearest recipes.

NVFP4 is NVIDIA E2M1 codes, FP8 block scale per 16 weights, and a global FP32 scale: nominal 9 bytes/16 = 4.5 bits/weight before padding/global amortization, equal to Q4_0's nominal density. It does not inherit a Hexagon-native fast path by being called FP4. Local MXFP4 is *not* interchangeable with NVFP4: the current packed MXFP4 tile constant is 544 bytes/1024 weights with an aligned 640-byte destination, and its decoder uses a LUT. Compare actual payload, alignment, scale arithmetic, output error and device cycles separately.

Preserve the Qwen GGUF's mixed tensors where needed: the filename Q4_0 does not mean every tensor is Q4_0 (sampled expert-down is Q4_1, and Q5_K/Q6_K paths were measured). Do not force the router, output head, sensitive norms, or all experts into one new format merely to simplify the harness. Evaluate a per-tensor or per-layer mixed recipe against a common F16/F32 source checkpoint and a fixed quality corpus.

## Research track A: Qualcomm's own runtime and documented operand formats

Before committing to an INT4 or FP4 kernel family, audit what Qualcomm's own current engine/toolchain actually documents and executes on this X2/v81 machine. Treat QNN/QAIRT/AI Engine Direct, installed SDK headers/examples, operator-package APIs, and any on-device profiling as separate evidence sources; a converter accepting a datatype does not prove native execution of that datatype in HMX/HVX.

- Inventory official tensor/quantization data types and encodings: signed/unsigned INT4, INT8, INT16, FP4/FP8, FP16/BF16/F32, block/group/per-axis/per-channel scales, zero points, accumulation/output types, and supported shape/alignment restrictions. Record exact SDK/runtime version and document or header location for each claim. If INT4 is exposed, identify whether it is stored only, unpacked to INT8/FP16, or has a measurable native dot/matmul path.
- For representative GEMV and GEMM shapes, compare the official engine's source tensor bytes, compiled/context bytes, padded transfer bytes, activation representation, intermediate VTCM footprint, and output dtype/rounding. Use public documentation first, then a tiny documented QNN graph and profiler/disassembly where permitted. Do not infer undocumented hardware opcodes from benchmark rate alone.
- Ask whether official kernels choose different lanes by M/K/N, expert count, sparsity, or quant encoding. If a datatype is unsupported or emulated, measure the conversion and launch cost instead of crossing it off by name. A useful deliverable is a version-pinned capability matrix: documented, compiler accepted, runtime executed, measured representation, and still unknown.
- Compare those results with the existing llama.cpp HVX byte-dot and HMX FP16 paths under the same frozen matrices. The official engine can be an oracle or a source of layout ideas without requiring the harness to adopt its graph format.

## Research track B: our own quantization and DSP-native reformat

The harness is free to choose an encoding, but there are two different experiments. A **lossless relayout** preserves the exact quantized values/scales and should pass byte/value plus same-arithmetic output gates. A **new quantization** changes those values and must be judged against the same higher-precision source model, fixed prompts/tasks, first-divergent-tensor diagnostics, and measured end-to-end cost.

- Explore offline calibration or training-aware rounding (including AutoRound), symmetric versus asymmetric INT4, signed versus offset nibbles, block/group sizes 16/32/64/128/256, per-output-channel scales, clipped outliers, and selective Q5/Q6/F16 residuals. Try co-designing weight codes with the chosen activation quantizer (W4A8 for decode, W4A16-like for prefill); do not assume one recipe serves both equally well.
- Repack independently for HVX GEMV and HMX multirow work: code order, scale/zero-point sideband, expert-major versus K-major tiling, 128-byte alignment, padded columns, hot-expert locality, and DMA double buffering. Account for any duplicate resident layouts or per-prompt conversion. Try a single on-demand packed source plus VTCM materialization before paying for two full model copies.
- Compare formats on *effective* bytes per useful selected expert and token: GGUF/source bytes, HTP-resident bytes, aligned DMA bytes, VTCM scratch, unpacked operands, scale traffic, output correction, and host mapping time. A nominal 4-bit format can lose to 4.5-bit Q4_0 if it needs extra gathers, scales, or repeated conversion.
- Preserve an untouched source checkpoint and manifests for every derived model. Record quantization seed/calibration set/export code and per-tensor exceptions. A smaller/faster model that changes greedy tokens is not automatically wrong, but quality must be evaluated as a new model rather than misreported as an output-preserving kernel optimization.

## Research track C: Qwen versus Gemma versus Nemotron MoE fit

Compare **specific pinned text-model variants**, not brand names or parameter-count headlines. Qwen3.6-35B-A3B and Gemma4-26B-A4B already have partial local shape audits ([GEMMA_MOE_HEXAGON_FIT.md](GEMMA_MOE_HEXAGON_FIT.md)); choose an exact Nemotron MoE checkpoint/config/revision before filling its column. Nemotron architecture, quantization and MTP/draft support must be established from that checkpoint, not borrowed from another Nemotron release. No model download or switch is implied by this to-do.

| Comparison dimension | Why it matters on this Hexagon |
| --- | --- |
| Layer count; hidden/intermediate widths; experts total/top-k; shared-expert width | Active weight bytes, GEMV length, selected expert reads, router/fusion count |
| Matrix K/N modulo 32/64/128, scale groups and physical tile strides | HVX lane utilization, HMX 32x32 packing, VTCM padding and DMA alignment |
| One-token versus prompt-M=2/4/8/9/32/128 shapes | HVX GEMV versus HMX GEMM eligibility and verification/prefill efficiency |
| GDN/recurrent/conv versus local/global/sliding attention; KV state | Cache traffic, state ownership/rollback, attention kernel fit and long-context cost |
| Router math, expert output weighting, shared expert, activation function and norms | Fusion opportunities, extra launches, exact-arithmetic boundaries |
| Vocabulary/output head and embedding tying | One large serial GEMV, model residency, output-copy/sampling cost |
| Actual per-tensor quant types, bytes, calibration quality and available source checkpoint | A model named Q4_0 may still mix Q4_1/Q5_K/Q6_K/F16; quality and placement are not implied by filename |

Produce a spreadsheet or machine-readable shape manifest with derived active *and resident* bytes, expected selected-expert transfer per token, padding factor, per-layer state/KV bytes at 128/1K/4K contexts, and a list of currently unsupported ops. Then run the same frozen kernel fixtures and full-HTP placement check for each model. Rank by measured correct-output decode, prefill and long-context behavior at comparable quality; do not infer a winner from fewer layers, total parameters, or idealized bytes alone. Existing Gemma arithmetic estimates suggest its active FFN bytes may exceed Qwen's despite fewer layers; Nemotron remains unmeasured.

## Small, decisive experiment ladder

1. Pin source checkpoint/GGUF hashes and tensor-type histogram. Export one frozen layer's exact weights, F32 activations, expert IDs, state and expected outputs, including prompt M=1/2/4/9 and one-token decode M=1. Record each physical layout's payload bytes, aligned bytes and actual transferred bytes.
2. Build isolated kernels for packed Q4_0-equivalent values: current offset nibble versus signed nibble, block32/64/128, scale vector placement, one/two output rows, expert slot ordering. Keep the same arithmetic contract and compare every output bit before benchmarking. Test alias/reused buffers, 128-byte alignment offsets, tails and deliberately poisoned padding.
3. Break out quantization, cache clean/invalidate, DMA queue/residence, unpack, dot, scale/reduction, HMX materialization, export and host round trip. Distinguish DMA interval from DMA-engine busy or exposed stall; use uninstrumented end-to-end A-B-B-A as the rate gate.
4. Requantize the **same** high-precision weights into symmetric/asymmetric INT4, AutoRound INT4, Q4_K/IQ4_NL, MXFP4 and NVFP4. For these *different models*, require deterministic repeats and compare output/error/task quality to the same high-precision reference. Do not require different quantizations to be token-identical to Q4_0; do not waive quality because a kernel is fast.
5. Test decode GEMV and prefill GEMM independently, then selected-expert fusion, output head and a whole layer. Only after first-divergent-tensor and model-quality gates should a format become a full-model candidate. Keep llama.cpp as the reference/placement lane until a new harness actually proves a better result.

## Wild ideas worth a bounded experiment

These are hypotheses, not hardware features or accepted optimizations. Each should have an exact/quality gate and a byte/cycle accounting before becoming a model change.

- **Expert-batched HMX by construction:** pack selected experts as a block-diagonal or grouped output problem, so HMX sees useful rows/channels without pretending eight different weights share one input matrix. Measure replicated activation bytes, zero/mask work and VTCM fit against the existing fused `MUL_MAT_ID_NX`; reject if fake M costs more than HVX GEMV.
- **Dual-format or just-in-time expert tiles:** keep compact immutable INT4 in DRAM, materialize byte-dot or FP16 tiles in VTCM for the chosen experts, perhaps prefetching predicted hot experts. Test residency/eviction under changing routes and account for any second cached copy; never count VTCM-only speed while hiding repack time.
- **Bit-sliced or LUT INT4 arithmetic:** try nibble-to-int8 lookup, sign-extension-free encoding, or two bitplanes with HVX byte-dot. Existing local MXFP4 LUT and Q4_0 unpack are controls. A four-bit code is not useful unless its decode plus dot beats the current byte path.
- **Scale/zero-point folding:** precompute a correction term or combine weight/activation block scales in a way that reduces per-block multiplies; for asymmetric INT4, use activation sums to fold zero-point corrections. Exactness may change with F16/F32 rounding order, so use frozen edge cases and full-logit checks.
- **Sparse high-precision outlier sidecar:** INT4 core plus a sparse F16/int8 residual for sensitive channels/experts or output-head rows. Measure sidecar bytes, random gathers, routing misses and quality; a smaller nominal core can still be slower.
- **Cross-token microbatching with honest state semantics:** accumulate independent prompt rows or speculative verification rows to reach HMX's useful M, but never combine unknown future tokens or commit rejected draft KV/recurrent state. Compare pp1/2/4/8, acceptance patterns and rollback cost, not just GEMM throughput.
- **Persistent graph with mutable descriptor table:** reuse the op graph while changing expert IDs, KV slots and VTCM addresses through a checked parameter block. Test captured-graph replay with 32 changing inputs, aliases, poison and rejection. A stale pointer can be fast and wrong.
- **Pipeline DMA and compute by expert tile:** double-buffer selected expert weights, overlap transfer with HVX compute, and optionally reorder experts for locality while restoring exact output order. Thread-trace DMA residence is not a stall measure; quantify actual overlap and exposed wait.
- **Precision split for HMX:** represent INT4 codes exactly as FP16 values for multirow matrix work, then apply block scales in a controlled order; test high/low residual decomposition only where it beats direct HVX and passes numeric gates. Public FP16 HMX support makes this plausible, not proven fast.
- **Kernel-shape spoofing with a negative oracle:** pad or permute K/N/M to hit a more favorable primitive, but validate every logical row and expert map, including wrong-map/poison controls. Count all padding bytes and launch savings. No model semantics may be silently discarded.

## Revisit queue before inventing another quantizer

1. GDN state-gather fusion: removes 30 `GET_ROWS` state copies (~0.96 ms in an older fast-path profile); early 28.59 tok/s run changed text. Later plain output varied even with fusion/replay off, so causality is unproven. Freeze state indices/strides and compare pre/post state plus first differing decode tensor under a matched allocation.
2. Shared-expert `SHEXP+EP` fusion: removes about 120 ops; early 28.93 versus 28.62 tok/s single runs changed late text. Re-run exact logits and frozen SwiGLU/down/gate intermediates with the same binaries and negative weight maps before changing arithmetic.
3. Router top-k aliasing: 117 fewer ops; later changing-input alias/replay oracle and three prompt classes passed, but the first all-layer trial changed one transcript and A-B-B-A speed ranges overlapped. Test first-divergent router outputs/logits and ownership; do not call it a speed win yet.
4. MTP: BF16-to-F32 draft gate expansion reduced scheduler splits and improved a clean 64-token pair from 19.59 to 26.39 tok/s, but still lost to plain decode and changed text. Separate target logits, ordered router IDs, draft KV cell metadata, accepted/rejected-prefix rollback, and recurrent state. Evaluate verified tokens per target forward, not acceptance alone. KV-only and fused verification probes also need this semantic gate.
5. CPU-style Q8_1 expert-down: sampled local NMSE improved about 1.4e-5 to 4.6e-14, but decode slowed and the CPU greedy prefix shortened from 93 to 60. Keep as an arithmetic-factorial control around the first divergence, not a near-term speed candidate. The initially inexact Q8_0 conversion *was* revisited and fixed; it is the current exact-output speed candidate, not an unresolved dead end.

## Sources and scope limits

- Local: `FINDINGS.md`, `STATUS.md`, `EARLY_ENGINE_REUSE.md`, `GEMMA_MOE_HEXAGON_FIT.md`, `llama.cpp-hexagon/ggml/src/ggml-hexagon/htp/{matmul-ops.h,matmul-ops.c,hvx-mm-kernels-tiled.h}`.
- Qualcomm v81 HMX programmer reference (FP16 cells/layout): https://docs.qualcomm.com/doc/80-N2040-62/80-N2040-62_REV_AA_Qualcomm_Hexagon_V81_HMX_Programmers_Reference_Manual.pdf
- ggml quantization type sizes: https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py
- Intel AutoRound method, recipes and exports: https://github.com/intel/auto-round
- NVIDIA NVFP4 encoding/scales: https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html

No new kernel, build, model download, quantization job or inference run was performed for this document. The goal remains paused pending the user's choice about restarting the campaign versus revisiting these leads, especially MTP.
