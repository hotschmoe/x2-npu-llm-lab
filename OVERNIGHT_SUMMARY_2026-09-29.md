# Overnight summary, 2026-09-28 evening to 2026-09-29 (prefill campaign + MTP + housekeeping)

Read this first. Details live in the linked docs. All numbers: Qwen3.6-35B-A3B Q4_0 on HTP0 (X2 Elite), llama.cpp-hexagon fork,
interleaved A/B, perplexity-gated. Commit trail: outer repo `git log` (prefill/doc commits), `llama.cpp-hexagon` branch `npu-htp-decode`.

## Results

| | before | after |
| --- | ---: | ---: |
| Prefill pp1024 (ub1024) | 454 t/s | ~758 (+67%); ~772 with `GGML_HEXAGON_OPPOLL=1` |
| Prefill pp512 (ub512) | 381 | ~665 |
| Prefill pp16 / pp64 / pp128 (ub512) | 71 / 158 / 244 | 142 / 322 / 420 |
| Prefill, Gemma 4 26B-A4B QAT pp1024 | 720 (stock serial MoE path) | 874 (+21.5%) |
| Decode, plain | ~28 tok/s | unchanged |
| Decode, MTP (draft-max 1) | 24.2 tok/s (14% SLOWER than plain) | +9% to +13% over plain (30.5 / 29.8 / 30.3 vs 27.9 / 27.3 / 26.7) |
| Perplexity (16 x 512 tokens) | 13.3331 | 13.3331 (bit-identical), 1.52 -> 0.82 s per pass |

## What changed (all in `llama.cpp-hexagon`, see PREFILL_CAMPAIGN.md for the numbered experiments)

Prefill: cross-expert software-pipelined HMX MoE kernel (default ON, `GGML_HEXAGON_MMID_PIPE`), 2 MB DDR scratchpad (removes a per-op
`memalign`), exact-range activation gather, row-chunked hot experts, column-chunked weights (extends the pipeline to Gemma-size tiles),
prefill conv-window+SiLU fusion, prefill expert-combine fusion (with an allocator-aliasing fix), DSP-side flatten of tiny-row
elementwise ops. Decode/MTP (by the decode agent, MTP_SPEEDUP_LOG.md): multi-token conv window, MoE tail fusions for 2-4 tokens, small-M
MUL_MAT+ADD, fused catch-up, draft vocabulary cap, diagnostics.

## Installed state (llama.cpp-hexagon/pkg-wos) - FINAL

Installed = the merged build (prefill line + MTP work), `llama.cpp-hexagon` branch `npu-htp-decode` at the merge commit (ee4381b), via
`npu_campaign/install-merged.ps1`: skel 37AD9661, catalog 080F8EA3, host ggml-hexagon.dll E90183F8, llama.dll 4BEA60C0.
Verified on the default installed path: PPL 12.1307 (bit-identical to the original), pp512 673 / pp1024 762 t/s, plain decode 27.2-27.4 tok/s,
MTP (draft-max 1) 29.0 vs plain 26.9 tok/s (+8.0%, 106/149 accepted); merge gates (MTP target logits 128/128 bit-identical on 3 prompts, fusion edge check
6/6) passed. Same-build plain repeat runs differ from themselves at the prompt row for BOTH the old and the new build, i.e. the plain-path instability is
pre-existing and not introduced by the merge. Caveat: tools the decode agent did not rebuild (batched-bench, mtmd-cli, diffusion-cli) are older than the new DLLs.
Rollbacks (every replaced file backed up): `backup-merged-install-prechange` (state just before the merge: prefill line at 7773910 without MTP work),
`backup-prefill-fusions-prechange` (host C9E4D8AD, skel 640AA38D, cat BB78A1E3), `backup-prefill-pipe-prechange` (original D2D98EDE/5CB7F7EF).

## How to use

- Prefill: `-b 2048 -ub 1024`. Measured in llama-server on a 1379-token prompt: 3.17 s (435 t/s, old `-b 512 -ub 128`) -> 1.78 s (775 t/s), decode about -2%
  (26.1 vs 26.5 tok/s); short prompts unchanged. `run-qwen-interactive-server.ps1` now defaults to these values (`-B`/`-Ub` parameters override).
  Optional: `GGML_HEXAGON_OPPOLL=1` (+1.9%, one busy host core), `llama-bench --prio 2` (+0.9%), `GGML_HEXAGON_ROUTER_HVX=0` (router on HMX, +9.7%,
  PPL +0.26%; CPU reference PPL is 12.2015 vs HVX-router 12.1307 vs HMX-router 12.1627, i.e. inside the NPU-vs-CPU spread; upstream's own default).
- MTP: see `npu_campaign/run-mtp-speed.ps1` (`--spec-type draft-mtp --spec-draft-n-max 1`, `LLAMA_MTP_FUSED_CATCHUP=1`, `LLAMA_MTP_DRAFT_VOCAB=65536`,
  expanded MTP GGUF). Draft-max 2 is slower. Known open issue: intermittent garbage output in non-recommended configurations (multi-slot replay, p-min,
  draft-max 2) and the pre-existing intermittent plain-path numerical instability (hypothesis written down in MTP_SPEEDUP_LOG.md, untested).
- Multi-agent NPU discipline: every run script takes `npu_campaign/npu-lock.ps1`.

## Other deliverables

- `UPSTREAM_PR_PLAN.md`: llama.cpp contribution rules (the human must write all PR text; agents may not; AI disclosure and `Assisted-by:` trailers) and a PR plan for
  the Hexagon backend (7 PRs, evidence commands, hygiene checklist). Nothing was pushed or opened.
- `MODEL_QUANT_HARNESS_REVIEW.md`: Qwen3.6 vs Gemma 4 vs Nemotron 3 Nano vs others, quant formats, harness recommendation (stay in llama.cpp, Zig host over the
  existing skel is the natural first step; ZML backend is a compiler project). Measured follow-ups appended (Gemma 4 pp1024 714 -> 874 after the column-chunk fix).
- Disk cleanup (your approval): C: free space 266 -> ~500 GB (Temp scratch, logit dumps, build outputs, caches, old models, WSL distros).
  WSL SSH key backed up to `~\.ssh\wsl-ubuntu-backup`. Kept: Gemma MoE, WoW install, wotlk_highdef, Drive cache, specula. No Nemotron model was on disk.
  Remaining reclaimable: `npu_campaign/mtp-speed/REF-base-mtp1-*.logits.bin` (~760 MB reference traces from the decode agent).

## Negative / neutral results (do not retry without a new idea)

Weight DMA over per-thread queues; fused dispatch per item (alone); l2fetch on the gather; Q4_0 vlut16 dequant and native half multiply (no gain even with HVX 86% busy);
router exact-refine (my exact-dot router did not reproduce the HVX router's PPL: unexplained); ub > 1024 (flat); bigger LUT/instruction-count tweaks in general.

## Lessons worth keeping

1. Always PPL-gate the first build of any fusion. A fusion that moves work earlier in the batch can overwrite an input the allocator aliased with its output (cost ~2 h).
2. HTP DMA sources should stay 128-byte aligned in these kernels (the decode agent hit this independently).
3. Per-op profile + stage trace first: the 2 MB scratchpad fix (+13%) was found by reading the INIT trace, not by guessing.
4. The bit-identical-PPL gate is right for pure performance changes but stricter than the true quality target; judge intentionally approximate changes against the CPU (12.2015).
5. Session-to-session timing drift on this machine is 5-10%; only interleaved multi-round A/B is trustworthy.

## Merge status

Done. `mtp-speedup` (decode agent) and `npu-htp-decode` (prefill line) were merged cleanly at ee4381b (the only content the decode branch lacked was the
column-chunk change in matmul-ops.c, which it did not touch). The decode agent's worktree `llama.cpp-hexagon-mtp` and branch `mtp-speedup` still exist; they can be
deleted once nobody needs them.
