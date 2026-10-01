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

## UPDATE 2026-09-29 evening: second merge installed (prefill2 + decode2 + mamba)

Installed = branch `merged2` of `llama.cpp-hexagon` (worktree `llama.cpp-hexagon-merged2`, tip ce75353 = npu-htp-decode ee4381b + decode2 84c8650 + prefill2 2bd9d77 + mamba 9c5e286; the only conflicts were ADD+NW in binary-ops.c / ggml-hexagon.cpp: mamba's n_src generalization x prefill2's Gemma expert-scale fold, resolved by keeping both, scale fold restricted to n_src == 8). Package `npu_campaign/pkg-merged2-m1` via `install-merged2.ps1` (host ggml-hexagon.dll 09641A0A, skel D9E7CFC0, catalog 063F65B1, llama.dll 70AC0BBE). Rollback: `npu_campaign/backup-merged2-install-prechange` (the state described above).

Root cause of the plain-path instability (decode2): `hex_l2flush` stepped 128 B per `dccleaninva`, but the v81 core cleans a 64 B granule, so every second half of a flushed range stayed stale. Fixed (64 B steps); plain runs are now bit-reproducible and plain == MTP target logits bit for bit.

Interleaved A/B, merged vs previously installed (3 rounds each):
| | installed | merged |
| --- | ---: | ---: |
| Qwen pp1024 ub1024, default (HVX router) | 762 | 764 |
| Qwen pp1024, `GGML_HEXAGON_ROUTER_HVX=0` (HMX router + exact top-16 refine, PPL 12.1611) | - | 843 (+10.6%) |
| Gemma 4 pp1024 | 874 | 984 (+12.6%) |
| Plain decode (run 1 / run 2) | 27.46 / 26.51 | 26.73 / 26.18 (within noise; ~1.5% plausible cost of the 64 B flush) |
| MTP decode (draft-max 1) | 29.82 | 29.32 (-1.7%), acceptance 106/149 unchanged |
| Nemotron-3-Nano Q4_0 (mamba, own session) | CPU fallback | pp1024 ~954, tg 31.7 |
| Nemotron-3.5-Lightning Q4_0 (mamba) | - | pp1024 970, tg 27.5 |

Gates on the merged build: Qwen PPL 12.1307 (4x512) and 13.3331 (16 chunks) exact; fusion on/off edge check 6/6; MTP/plain full-vocab logits bit-identical to decode2's fix build (128 rows, plain-plain, mtp-mtp, plain-mtp); Nano PPL 16.1617 vs 16.1310 (KLD vs CPU 0.00527 vs 0.00516, inside the +-0.00024 error: noise floor). Installed default path re-checked: PPL 12.1307.

Defaults: router stays on HVX (bit-exact); set `GGML_HEXAGON_ROUTER_HVX=0` for the HMX router (+10-11% prefill on Qwen, PPL +0.25%). New: SSM_SCAN on HTP0 (Nemotron), Q5_1 shared-expert repack to Q8_0, cache-flush fix. Target Nemotron is 3.5 Lightning (see NEMOTRON_NPU.md). Not done: decode2's `qq_pending.sh` (Q4_0 MTP head, garbage-output hunt), NPU MTP speed on Lightning, Nemotron ub sweep. Reference `.kld` for Nano regenerated at npu_campaign/mamba-corr/cpu_nemo.kld (267 MB, deletable).

## UPDATE 2026-09-30 evening: third install (upstream sync + ops650 decode work + mtp3), supersedes the "second merge" numbers above

Installed = `llama.cpp-hexagon` branch `npu-htp-decode` = `final` (tip = merge of `mtp3` into `rollup` = `upstream-sync` (origin/master 19e28a2 of 2026-09-29, 78 upstream commits) + `ops650`). Package `npu_campaign/pkg-final2-f1` via `install-final2.ps1` (host ggml-hexagon.dll 70BB6933, skel 9F88BC20, catalog 8E7F9624, llama.dll 9624DDE5). Rollback: `npu_campaign/backup-final2-install-prechange` (the second-merge state: merged2 ce75353).

Interleaved A/B vs the previous install (2 rounds; this machine on AC):
| | previous | now |
| --- | ---: | ---: |
| Plain decode (Hello, 256 tok) | 26.94 tok/s | **36.02 (+33.7%)** |
| Qwen pp1024 ub1024 | 751 | 778 (+3.5%) |
| MTP decode, Hello (acceptance 106/149) | 28.92 | 33.13 (but plain is now faster: use plain for chat) |
mtp3 matrix (separate interleaved blocks, 256 tokens, greedy; code 2 rounds, others 1 round, indicative): code plain 34.4 / MTP d1 37.4 / +p-min 0.5 37.3 / gated d2 37.9; JSON tool call plain 34.2 / MTP 34.7 / +p-min 0.5 35.4; copy-heavy plain 34.0 / MTP 34.2 / +p-min 0.5 35.8 / gated d2 37.2. Recommendation: plain for chat prose; MTP (`--spec-type draft-mtp --spec-draft-n-max 1`, `LLAMA_MTP_FUSED_CATCHUP=1`, `LLAMA_MTP_DRAFT_VOCAB=65536`, `--spec-draft-p-min 0.5`) for code / structured / copy-heavy output. Cost model: verify step M=2 is 43.7 ms DSP vs plain 26.7 ms; extending the M=1-only SHEXP+EP / RMS_NORM+SCALE fusions to M=2..3 is the next MTP lever (~2 ms).

Where the decode speed came from (ops650): the DSP dirty-range tracker cost 2.8-3.2 us on every op (O(64x64) merge twice per op; ~4 ms/token): now incremental; vgather conv-window (WIN 48.8 -> 10.5 us); four matchers disabled since 09-26 re-enabled (exact after the flush fix); SILU+MUL and conv SILU-fold fusions; graph 1384 -> 994 ops. FastRPC dispatch itself is NOT a bottleneck (0.38 us/op fork/join).

Root causes found this round: (1) hex_l2flush 64-byte step (decode2): plain-path instability AND the earlier garbage output in draft-max 2 / p-min / multi-slot replay configs (mtp3: 12/12 clean after, garbage before); (2) HVX/HMX int4 x u8 is no-go (int4u8 agent: HMX does int4 x u8 exactly but accumulator is only readable as u16, per-block Q4_0 scales must be applied on HVX, gain <= ~3% at ub1024).

Gates on the installed build: Qwen PPL 12.1307 / 13.3331 exact; fusion on/off edge check identical (6/40/130/260/520 words; 900 skipped); MTP/plain full-vocab logits bit-identical vs the previous install (128 rows, draft rows too).

Warnings / corrections:
- **Do NOT use `GGML_HEXAGON_OPPOLL=1`** (earlier listed as +1.9% optional): it hung twice in ~14 spin runs at 100% CPU (the dspqueue_read loop in flush_pending retries forever with no timeout); a hung run holds the NPU lock. The earlier "+9% / +13% MTP" and "spin" numbers above are superseded.
- **Gemma 4 produces garbage on the NPU** (every output token soup, even at temp 0; coherent on CPU). The Gemma speed numbers in this document and PREFILL_CAMPAIGN.md (874 -> 984 etc.) measure a model that is not working: ignore them. "Greedy identical" checks between NPU builds cannot catch this. Unresolved, out of scope (decision: skip Gemma).
- Model choice: Qwen3.6-35B-A3B is the focus (MODEL_CHOICE_AGENTIC.md). PI-agent tool-call suite (npu_campaign/pitest, run: `python npu_campaign\pitest
un_suite.py --model qwen --reps 3`): Qwen 33/33, Nemotron Lightning 32/33.
- Not done: Q4_0 MTP head (+1.0% PPL, do not use), temperature-0.6 acceptance, M=2 fusion extension, ops650 host-knob runs (OPBATCH 1280; queue scripts kept), prefill A/B of the tracker fix.
- Old worktrees (prefill2, decode2, mamba, mtp, ops650, mtp3, int4u8, upstream, rollup, merged2) can be removed; all content is in `npu-htp-decode`.

## UPDATE 2026-10-01 morning: fourth install, `qwen4` s14 (plain decode ~40 tok/s) - CURRENT STATE

Installed = `llama.cpp-hexagon` branch `npu-htp-decode` = `qwen4` tip a21dc7f (on top of the third install: upstream sync + ops650 + mtp3). Package `npu_campaign/pkg-qwen4-s14` via `install-s14.ps1`
(skel 653253A7, catalog 666FBFB6, host ggml-hexagon.dll E646F99F, llama.dll D2BB8AB1). Rollback: `npu_campaign/backup-s14-install-prechange` (= previous install, also available whole as `npu_campaign/pkg-final2-f1`).
Verified on the installed default path: Qwen PPL 12.1307 (4x512).

What qwen4 found/changed (all default ON in s14, each with an env opt-out `=0`): GGML_HEXAGON_MV_XPIPE (flat-unit cross-expert DMA pipeline for decode/verify expert matmuls), GGML_HEXAGON_Q4C
(Q4_0/IQ4_NL tiles padded to 640 B in DDR so each tile row is ONE contiguous DMA descriptor, original dot kernel kept; contiguous Q8_0 kernels, one-descriptor DMA for Q4_1/Q6_K; costs ~11% more Q4_0 memory),
GGML_HEXAGON_GDN_SNAPDMA (async DMA for verify recurrent-state snapshots), GGML_HEXAGON_GDN_STATE_MAXT (state gather folded into the GDN op, T<=4). Root cause of the old Q4-vs-Q8 bandwidth gap
(70-98 vs 125-141 GB/s): the 2D weight DMA descriptor (64 rows x 576 B) costs ~12 ns per row; contiguous descriptors reach 133-136 GB/s. Off by default: GDN_RB (-3.6% on one op).
Spin-poll: the infinite dspqueue_read loop is now a bounded spin (400 ms) + blocking-read fallback + 120 s watchdog; BUILT BUT NOT STRESS-TESTED: `GGML_HEXAGON_OPPOLL=1` is still NOT recommended.

Interleaved ABBA x3, AC, 256 tokens, vs the previous install (tok/s means): plain Hello 35.26 -> 40.23 (+14.1%), plain code 35.02 -> 40.16 (+14.7%); MTP d1 Hello 33.08 -> 35.22, code 37.47 -> 40.15;
MTP d1 + p-min 0.5 Hello 33.28 -> 35.56, code 37.86 -> 39.80. => on this build plain decode matches or beats MTP everywhere: USE PLAIN (serve\start-qwen.cmd -Mode chat). MTP verify (M=2) batch 43.3 -> 41.35 ms
(experts at M=2 touch up to 2x the expert bytes, the structural limit of MoE speculation).
Gates on the s14 package (qwen4): PPL 12.1307 / 13.3331 exact; plain and MTP target logits bit-identical vs pkg-final2-f1 on Hello and code (128/128 rows), draft rows 72/72 and 66/66; edge check identical (6..520 words).
Prefill on s14 was NOT A/B-measured (my A/B was interrupted by a laptop sleep and then by unplugging); the 2048-context pp/tg numbers are in the section below when measured on AC.

Also delivered overnight: `serve\` (start-qwen.cmd -Mode chat|code, decisions_shim.py = /v1/decisions-style typed-question endpoint, smoke.ps1; see serve\README.md, SERVE_LOG.md; NPU lock is held while the server runs),
`npu_campaign\pitest` (PI-agent tool-call suite: Qwen 33/33, Nemotron Lightning 32/33), `MODEL_CHOICE_AGENTIC.md` (Qwen chosen), `DEEPGEMM_ASCEND_LEARNINGS.md`, `NEMOTRON_NPU.md`
(Nemotron 3 Nano / 3.5 Lightning run fully on the NPU, pp1024 ~954/970, tg ~32/27.5), negative results: int4xu8 (INT4U8_LOG.md), Q4_0 tile layout rework (LAYOUT_LOG.md: decode dot -26%/-44% per tile but prefill +7% dequant => -3.5% pp, not worth it;
code stays on branch `layout`), MTP at chat prose (below plain).

Operational lessons (this night): (1) ONE model process at a time on this 48 GB machine: two overlapping runs (qwen4, 23:17) pinned ~36 GB of NPU memory, the machine became unusable and the killed processes took ~15 min to exit.
`npu_campaign/ram-guard.py` (kills a second llama-* process by exact PID) was written for this; npu-lock.ps1 has a small race (New-Item), lockrun.py is the atomic lock; a rewrite of npu-lock.ps1 was tried and reverted (flaky).
(2) Never kill processes by command-line pattern from inside the same shell (the pattern matches your own command). (3) The laptop sleeps when unattended: timers and benchmarks stall (03:51-09:19 lost); keep it awake or on AC for long runs.
(4) Gemma 4 produces garbage on the NPU (unresolved; ignore every Gemma number in these docs).
Worktrees were removed (all branches kept in git: prefill2, decode2, mamba, mtp-speedup, ops650, mtp3, int4u8, upstream-sync, rollup, merged2, final, layout, qwen4, npu-htp-decode).
Not done: temp-0.6 MTP acceptance, 30-run spin-poll stress, M=2..3 extension of the SHEXP+EP / RMS_NORM+SCALE fusions (judged ~0.3-0.4 ms each), prefill A/B of s14, `npu-lock.ps1` atomicity.
