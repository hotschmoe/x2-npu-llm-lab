# Specula and Trident harness reuse audit

The expanded format, hardware, AutoRound, wild-kernel and MTP experiment backlog is in [HEXAGON_HARNESS_RESEARCH.md](HEXAGON_HARNESS_RESEARCH.md). It supersedes any shorthand format recommendation here. In particular, Q4_0 has a 576-byte packed 32x32 DSP tile and a separate 640-byte aligned destination; 640 bytes is **not** the Q4_0 packed payload. The paused campaign has not begun a custom harness implementation.

User-requested agent review, 2026-09-27. Cloned via gh to C:/Users/hotschmoe/Documents/github; clean worktrees. No builds/inference/commits/pushes. Findings are source/history review, not performance qualification.

## Branches

- Trident master 6a5262d, only remote branch.
- Specula master 0e982bb.
- Specula p1-int8-kv-16x8: 120 commits behind, one unique commit0cb72c3 (AIMET int8 KV/16x8 attention/int8 head); quantization research, not output-preserving Q4_0 implementation.
- Specula spec-decoder-example-rtx3090: 386 commits behind, no unique commits.
- Trident is Zig; Specula Python/PowerShell. No tracked Rust/Dart implementation found.

## Salvage priorities

1. Trident src/model/gguf.zig:139 mmap lifecycle, :218 tensor lookup, :421 metadata handling. Windows-specific; differential validation against authoritative GGUF parsing required.
2. Specula npu_engine/bench_qwen3_4b_ortqnn.py:139 persistent output binding, :171 borrowed-view lifetime contract, :204 state copying before buffer reuse. ORT CPU IOBinding is not established CPU-DSP zero-copy.
3. Specula npu_engine/qualcomm_qwen3_4b_oracle.py:335 fixed KV layouts, quantized zero initialization and AR1/AR128 master/mirror ownership. npu_engine/engine_14b_q.py:69 binary seam/state metadata. No current Qwen GDN/MTP rollback implementation.
4. Specula scripts/npu_spec_outer_loop.py:90 snapshot indexing and :133 all-accepted lazy snapshot. Borrow rejection/bonus test inventory, not external-draft implementation as real MTP.
5. Trident src/npu/gemm.zig:349 quantization/context creation/execution/free/output conversion timing. Specula scripts/parse_qnn_profile.py cycle decomposition, end-to-end/lib/bundle.py:26 SHA256 provenance, scripts/analyze_hmx_probe.py raw tile/layout analysis.

## Dead ends and constraints

- Trident NPU GEMM converts weights/activations/output to u8 with127*K output scaling; quality unvalidated and not current Q4_0 arithmetic.
- Trident npu_current_status.md historical QNN context churn: approximately32.2s create,7s destroy,1.4s execute; prefill4.4 versus36.3tok/s. Not rerun tonight.
- qnn_types.zig:463 registration descriptors remain opaque; shared rpcmem allocation/registration proposed, not implemented. Existing GEMM RAW client buffers are not custom FastRPC transport.
- Trident fixed dense seven-projection forward lacks MoE/GDN/MTP.
- Specula scripts/npu_vs_cpu_correctness.py:590 cosine>.99 and greedy-match>=50% are too loose for present promotion gates; identical-state single-step controls remain useful.
- Specula HMX analyzer references sibling llama-int4a16 branch npu-int4a16-hmx; device implementation is not in these reviewed repositories. Historical primitive conclusions are probe-specific.

## Proposed architecture, not implemented

Thin Zig or Rust host runner; reuse file/lifecycle utilities and fixed-output/state/manifest contracts, small C ABI around existing FastRPC transport and signed C DSP kernels. Start one frozen subgraph, disjoint/reused buffers, deterministic poison, identical input/state repeats, byte comparisons, per-stage timing. Keep llama.cpp as reference/model lane. Dart can serve orchestration/UI. Do not port whole engines or replace arithmetic based on this review.

## Try later: Hexagon-specific harness experiments

User explicitly requested preserving this backlog. This is a proposed research lane, not authorization to replace the current engine or change its arithmetic.

- Host options: Zig for explicit allocator/lifecycle plus existing Trident utilities; Rust for typed ownership and safe manifest/state handling; C++ for minimal direct reuse of current ggml/transport code. Keep signed C/HVX/HMX DSP kernels and expose a narrow C ABI first. Language choice is not a kernel optimization by itself.
- First executable: replay one frozen Qwen layer or subgraph with exact tensor bytes, dimensions, strides, quant layouts, initial recurrent/KV state, and expected outputs exported from the reference. Preserve model and artifact hashes; do not silently requantize.
- Explicit memory plan: immutable weights, persistent state, transient activation slots, borrowed output lifetime, scratch ownership, buffer alias table, cache maintenance and completion contracts. Compare disjoint buffers with deliberately reused slots, clean-read versus dirty-write transitions, poison patterns, tail widths and alignment offsets.
- Execution controls: persistent DSP graph/descriptor plan, one versus multiple submissions, replay on/off, selective fusions, worker count, and fixed versus dynamic shape plans. Avoid per-token/per-shape QNN context construction.
- Correctness gates: repeated identical inputs/state must repeat byte-identically under identical arithmetic; compare first differing intermediate; CPU numerical oracle where appropriate; require selected-token identity for output-preserving optimization. Test MTP reject-first/reject-later/all-accepted/bonus cases with recurrent state and KV restoration, not only text or acceptance rate.
- Timing boundaries: model mapping/repacking, shared-memory setup, host dispatch, input transforms/activation quantization, cache clean/invalidate, DMA in, HVX/HMX compute, DMA out, barriers, state commit/rollback, output readback and sampling. Instrumented device sums must not be substituted for uninstrumented end-to-end latency.
- Promotion criterion: reproduce the current failure without observation-induced suppression, or demonstrate a reproducible host/scheduling cost that a fixed plan removes. Then paired same-artifact A/B/B/A benchmarks and frozen multi-prompt correctness. A passing synthetic harness alone does not qualify full-model/MTP correctness.
- Keep out of initial scope: a full ZML/compiler backend fork, new quant arithmetic, general model loader, tokenizer/server rewrite, or a Dart DSP kernel port. Expand only when the small harness provides evidence.
