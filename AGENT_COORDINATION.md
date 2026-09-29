# Multi-agent rules for the Hexagon NPU campaigns (read before touching anything)

Three agents work in parallel on ONE machine and ONE NPU: `prefill2`, `decode2`, `mamba` (Nemotron). The coordinator (main session) merges.

## Isolation
- Each agent works ONLY in its own git worktree of the nested repo `llama.cpp-hexagon`, branched from `npu-htp-decode` (the merged, installed line, commit ee4381b):
  `git -C llama.cpp-hexagon worktree add ../llama.cpp-hexagon-<name> -b <name> npu-htp-decode`  (name = prefill2 | decode2 | mamba).
  Use your own build dirs (e.g. `build-<name>`) and package dirs (`npu_campaign/pkg-<name>-*`); look at `npu_campaign/build-mtp.cmd`, `mtp-pkg.ps1`,
  `build-prefill-htp.cmd`, `prefill-cand.ps1`, `build-prefill-host.cmd` for working recipes (DSP-only skel build, host build, signing, packaging).
- NEVER modify `llama.cpp-hexagon/pkg-wos` (the installed, verified build) or the `npu-htp-decode` branch. The coordinator installs and merges. Never `git checkout`/`stash`/`reset`
  shared files; never `git add -A`; commit only your own files by explicit path, in the outer repo (docs/scripts) and in your worktree branch (code). End commit messages with a line
  `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Keep a per-agent append-only log: `npu_campaign/<NAME>_LOG.md` (timestamped: hypothesis, change, result, decision).
- The machine has ~500 GB free but three agents build a lot: delete big traces (`*.logits.bin`, `prefill-prof3-*.log.err`) when done; use `cmake --build --parallel 6`.

## The single NPU
- Only ONE inference/benchmark process on the NPU at a time. Every run script already takes `npu_campaign/npu-lock.ps1` (dot-source it and use Enter-NpuLock/Exit-NpuLock in try/finally for any
  other way of launching llama-*). Keep each hold to one benchmark (minutes, not hours). Expect to queue behind the others; plan CPU-side work (reading, coding, building) while waiting.
- Timing drifts 5-10% between sessions: report ONLY interleaved A/B (`ab-prefill.ps1`, `ab-env.ps1`, `mtp-abba.ps1`) with several rounds; never compare numbers from different times.

## Correctness methodology (non-negotiable; this is how the project stays honest)
1. Bit-exact gates for pure-performance changes: prefill perplexity `npu_campaign/run-prefill-ppl.ps1` must print exactly 12.1307 (Qwen3.6, 4x512) and 13.3331 with `-Chunks 16`; run it on the FIRST build of
   every change (a fusion that moves work earlier can silently overwrite an aliased input: PPL 82 vs 12.13 cost two hours once). Greedy-completion equality across prompt sizes around ubatch boundaries:
   `run-fusion-edge-check.ps1` / `run-lib-compare.ps1`. Decode/MTP: full-vocabulary logit rows bit-identical via `compare_logits_traces.py --require-identical` (`mtp-gate.ps1`).
2. New ops/kernels get a per-op oracle: run the op on CPU and on HTP0 on the same frozen inputs (`test-backend-ops -b HTP0 -o <OP>`, or a frozen-fixture script in the style of
   `npu_campaign/check_frozen_*.py`) and require NMSE <= 1e-7 (plus a negative control that must fail, e.g. wrong weight row) before any end-to-end claim.
3. When something diverges: find the FIRST divergent tensor/row (tensor tracing `LLAMA_TENSOR_TRACE_FILE`/`_FILTER`, `compare_tensor_traces.py`), do not guess. Known hazards: (a) allocator aliasing of a fused
   op's output with an input, (b) DMA sources should be 128-byte aligned, (c) the DSP dirty-range cache tracker (`htp-tensor.c`), (d) pre-existing intermittent PLAIN-path run-to-run instability
   (same build differs from itself at the prompt row) - always run a same-build repeat before blaming your change.
4. Intentionally approximate/"spoofed" experiments (fake ops, skipped work, reduced precision to find a bound) are ENCOURAGED for learning where time goes, but must be OFF by default, clearly named,
   logged as such, and never installed or counted as a result.
5. Judge intentionally approximate changes against the CPU reference (CPU PPL on the same text = 12.2015; NPU variants spread ~0.07), not against the NPU baseline.

## Reference material
`OVERNIGHT_SUMMARY_2026-09-29.md` (start here), `PREFILL_CAMPAIGN.md`, `npu_campaign/MTP_SPEEDUP_LOG.md`, `MODEL_QUANT_HARNESS_REVIEW.md`, `UPSTREAM_PR_PLAN.md`, `npu_campaign/FINDINGS.md` and `STATUS.md`
(decode/correctness methodology history; long - grep), `NPU_MOE_20TPS.md`, `../github/specula/docs/` (HMX/QAIRT probing), `../github/trident` (Zig engine). Profiling: `GGML_HEXAGON_PROFILE=1` (per-op) / `=3` (per-stage trace, needs ub512),
`npu_campaign/prefill_prof_sum.py`, `prefill_stage_sum.py`.

## Merging
Finish by merging `npu-htp-decode` INTO your branch (resolve conflicts, you know your code), re-run your gates plus the prefill PPL gate, and report the resulting commit. Do not move `npu-htp-decode`.
Final report (<800 words): measured results (interleaved), correctness status, what remains, exact commits/files.
