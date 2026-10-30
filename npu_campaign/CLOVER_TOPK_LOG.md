# CLOVER_TOPK_LOG (agent `topk`, branch `clover-topk`, worktree `llama.cpp-hexagon-clover-topk`)

Append-only. Goal: Clover router/coreset top-k on HTP with the SAME ids in the SAME order as the CPU custom op
(`build_clover_top_k`: value desc, ties -> lower expert id first), no CUSTOM CPU islands.

## 2026-10-30 - design
- Facts: CPU `ggml_argsort` = std::sort (unstable); CPU `ggml_top_k` = partial_sort + deliberate swap of the first two
  ("order is not important"); HTP bitonic (`argsort-ops.c`) is not stable either. So no existing op can be used as is.
- Order matters downstream: `build_moe_ffn` gathers weights with `selected_experts` order and sums experts in that order
  (fp add order), so ids AND order must match.
- Design (a): new API `ggml_top_k_stable(ctx,a,k)` = `GGML_OP_TOP_K` with op_params[0]=1: descending, ties -> lower index first.
  The comparator (value desc, index asc) is a strict total order, so ANY correct sort returns the identical result as the old
  custom op -> graph change is semantics preserving on CPU (CPU impl: partial_sort with that comparator, no first-two swap).
  HTP: bitonic CAS on composite key (value, then lower index greater) via `key_gt()` in `argsort-ops.c`; only the single-chunk
  paths (ne00 <= 1024, pow2 register sort and padded VTCM sort) implement it, `htp_kparams_solve_sort` rejects stable top-k otherwise
  (-> CPU fallback, still correct). Other backends (CUDA/Metal/Vulkan...) ignore op_params and are NOT audited (not built here).
- Env switches: `CLOVER_TOPK_CUSTOM=1` (graph: old CUSTOM op, A/B + fallback); `GGML_HEXAGON_TOPK_SPOIL=1` (test-only negative control:
  HTP runs the unstable kernel; the stable test cases must FAIL).
- Tests: `test_clover_top_k` block in tests/test-backend-ops.cpp (delimited `BEGIN/END topk agent`), checks CPU and HTP outputs
  against an explicit host std::stable_sort reference; real-logit fixtures via `CLOVER_TOPK_FIXTURE` (npu_campaign/clover_topk_fixture.py).
- Capture of real logits: env `COMMON_DEBUG_DUMP_DIR` + `COMMON_DEBUG_DUMP_RE` in common/debug.cpp (used with llama-eval-callback, CPU).

## 2026-10-30 00:00-00:40 - build + oracles (pkg-clover-topk-a, skel F1C24E5069BC9462, branch commit e511c98)
- Build: `build-clover-topk.cmd` (--parallel 4), `pkg-clover-topk.ps1 -Name a`. First DSP build failed (missed ARGSORT call sites), fixed.
- CPU (host) oracle: `test-backend-ops test -b CPU -o TOP_K -p stable=1` 76/76 vs the explicit std::stable_sort reference.
- HTP0 oracle `topk-tbo.ps1 -Label tbo-topk-1` (`test -b HTP0 -o TOP_K -p stable=1`): **76/76 passed**, no "not supported":
  patterns random, quantized ties {0..3}, dup values, coreset-style -1e30 masks (2..40 active, fewer than k), all-zero, +-0, +-inf,
  denormals, widths 32/64/128/512/1024 + padded 100/240/496, 4D batch; rows 1/16/256; k = 8 and 256. Each case checks CPU AND HTP
  against the host reference (exact ids and order).
- Negative control `GGML_HEXAGON_TOPK_SPOIL=1` (unstable kernel, same op accepted): **FAILS** on the tie patterns
  (e.g. quantized ties 256x256 k=8 ERR=256 rows, masked 36 rows, all-zero 256) -> the oracle bites. Log tbo-topk-spoil.log.
- CPU PPL with the new graph (`topk-ppl.ps1 -Mode cpu`, 4x512): **15.4413** (unchanged vs 15.4413).
- KLD vs CPU (`clover-kld.ps1 -Label topk-a`): mean KLD 0.047923, same top 89.216%, RMS dp 4.647%, PPL 15.0832 (baseline HTP
  0.0479 / 89.2% / 15.0832): not worse (as expected: pure scheduling change).
- Placement audit (GGML_SCHED_DEBUG=2, `topk-sched`): CUSTOM 0. Splits: draft 1162 -> 922, causal verify 762 -> 682,
  decode 582 -> 502. Remaining CPU ops: CPY (446-566) + 2 GET_ROWS (bf16 agent). Note: the -v sched-debug run aborted at the very end
  with `dspqueue_read failed 0x72` after generating text (verbose graph dumps); the non-debug 256-token A/B is the stability check.
- Shared-infra note: `Enter-NpuLock` in npu-lock.ps1 threw `Cannot convert System.Object[] to System.Int32` under lock
  contention (owner.txt read as array -> `[int]$ownerPid`); my scripts retry (logs clover-logs/topk-lockerr.txt).

## 2026-10-30 01:00-01:35 - model crash with k-wide TOP_K output; fix
- pkg-clover-topk-a (TOP_K with k=8 contiguous output, 32 B ids rows) passed every oracle and KLD but CRASHED in generation runs
  (`ggml-hex: dspqueue_read failed: 0x72`, both the sched-debug run and A/B round 1 B). Stock graph never feeds 32 B-stride ids to
  HTP get_rows/mul_mat_id (it uses argsort + view, row stride 1 KB). Hypothesis: a downstream HTP op assumes wide/aligned ids rows.
  Not root-caused further (no time); fix = the old layout: sort the full 256 row (`ggml_top_k_stable(logits, ne0)`) and view the first k.
  pkg-clover-topk-c (commit on clover-topk after e511c98) completes a 32-token run, coherent text. Oracle covers k=256 so no new kernel risk.
  Follow-up: find which consumer needs the wider stride (mul_mat_id / get_rows / add_id) - maybe a kparams check should reject it.
- Lock incident: stopping my chain mid-run left an orphan model process (pkg-clover-base) for ~1 min that overlapped another agent's run;
  killed it. Round-1 A/B numbers in topk-ab1 (00:53-01:12) are discarded (overlap window); `topk-ab2` is the valid A/B.
- Raw perf (HTP0, pkg-a, `perf -b HTP0 -o TOP_K -p pat=0`, measured at lock-contention time, indicative only): TOP_K stable k=8 rows 1/16/256 =
  11.2 / 35.6 / 275 us; k=256 = 11.2 / 20.1 / 274 us; ARGSORT 256 desc rows 1/16/256 = 7.5 / 12.5 / 155 us (separate run). Single-row cost is
  dominated by op dispatch. Interleaved re-measure queued (perf-both-c).

## 2026-10-30 02:00-02:50 - pkg-clover-topk-c gates (graph: stable top-k of whole row + view k; commit 78453af)
- KLD (`clover-kld.ps1 -Label topk-c`): mean KLD 0.047923 +-0.0023, same top 89.216%, RMS dp 4.647%, PPL 15.0832 - identical to base HTP.
- Audit (`topk-c-sched`): CUSTOM 0; remaining CPU ops CPY 446-566 + GET_ROWS 2. Splits: draft 1162->922, causal verify 762->682, decode 582->502
  (and the -v sched-debug run now completes without the DSP error).
- Interleaved A/B, FIFO-valid pairs only (clover-run.ps1 -Accept '' 256 tokens, f16 KV; summaries topk-ab2/topk-ab3.summary.txt):
  | pair | base (pkg-clover-base) | topk-c | delta |
  | ab2 r1 | 4.894 | 5.163 | +5.5% |
  | ab3 r1 | 4.912 | 5.159 | +5.0% |
  | ab3 r2 | 4.899 | 5.143 | +5.0% |
  (ab2 r2 A = 4.566 was disturbed - HOST_CPU 1.53 busy cores - and the chain was restarted; ab1 discarded.) HOST_CPU busy cores: base 1.07-1.16,
  topk-c 0.98-0.99 (the remaining busy core is the CPY islands / scheduler, not the sort).
- Interpretation: removing 40(+80) CPU islands per graph is worth ~5% on the self-draft loop (~+0.25 tok/s); the rest of the 900+ splits (CPY) dominate.

## 2026-10-30 03:00-03:50 - real-logit fixtures, CPU PPL, perf
- CPU PPL with the pkg-clover-topk-c graph (`topk-ppl.ps1 -Mode cpu`): **15.4413 +/- 1.52252**, exactly the CPU reference -> the graph change is
  semantics preserving on CPU.
- Real router logits captured from a CPU run (llama-eval-callback + COMMON_DEBUG_DUMP_*, ~1000 token prompt, all 40 layers; clover-logs/topk-capture/*.f32),
  fixtures built by clover_topk_fixture.py: fixture-raw (1248 real `ffn_moe_logits` rows), fixture-masked (1248 rows after the emulated coreset mask,
  16-token blocks, avg 221/256 entries exactly -1e30), fixture-coresetnorm (78 per-block normalized mass vectors, mostly exact 0.0).
- HTP0 oracle (pkg-clover-topk-e, `topk-tbo.ps1`, `CLOVER_TOPK_FIXTURE=...`): **140/140 passed** (76 synthetic + 64 real-fixture cases; rows 1/16/256,
  several row offsets, k=8 and k=256) - exact ids and order vs the host std::stable_sort reference, CPU also 140/140.
- Negative control on the fixtures (`GGML_HEXAGON_TOPK_SPOIL=1`): **43/64 pass, 21 FAIL** - all failures are k=256 (full order of masked/zero-tie rows),
  k=8 of real rows passes either way (no ties inside the real top-8). So: the router top-8 almost never meets ties on real logits, but the coreset's
  256-wide order does (zeros in the normalized mass), and there the unstable kernel gives a different order.
- Interleaved raw perf (one process, `perf -b HTP0 -o TOP_K|ARGSORT -p ne=[256,`, us/run; includes ~7-10 us op dispatch for 1 row):
  | rows | ARGSORT (unstable, old kernel) | TOP_K k=256 stable | TOP_K k=8 stable | TOP_K k=256 unstable | TOP_K k=8 unstable |
  | 1 | 6.7 | 11.1 | 10.9 | 13.4 | 13.6 |
  | 16 | 12.5 | 20.0 | 34.9 | 26.7 | 45.6 |
  | 256 | 155 | 274 | 275 | 386 | 386 |
  The composite-key compare costs nothing visible (stable is even faster than the unstable TOP_K path - unexplained, the TOP_K op path is ~1.7-2.5x slower
  than the ARGSORT op path for the same network; candidate follow-up: route through ARGSORT op with a stable flag, or look at the TOP_K dst DMA). k=8 with
  16 rows (32 B output rows) is slower than k=256 (35 vs 20 us): small unaligned DMA writebacks. Graph uses k=256 + view, so it takes the fast case.
  Per layer cost: router 1 row 11 us (decode), 16 rows 20 us (draft/verify), coreset sort 11 us -> 40 layers ~ 0.45 ms decode, ~2.4 ms draft graph (3 sorts/layer).

## 2026-10-30 04:00 - spoof bound (OFF by default) and wrap-up
- SPOOF `CLOVER_CORESET_SPOOF=1` (coreset skipped, results change): 4.537 t/s vs 5.034 (same pkg, interleaved, 1 pair; n_accept 185 vs 189). Slower, not faster:
  without the coreset mask every token of the block routes to its own top-8, so many more distinct experts are read per draft pass. The coreset is a speed feature;
  its own op chain cost can not be bounded this way (needs per-op profile: GGML_HEXAGON_PROFILE=1, not done - lock queue).
- Not done: router/coreset kernel fusion. Estimated ceiling from the measured op costs: router sort 11 us x 40 layers ~ 0.45 ms/token of ~97 ms; coreset chain ~16
  small HTP ops/layer (draft only). A fused kernel would save at most a few ms on ~200+ ms draft passes; the 900+ splits from CPY (bf16 agent) are the lever.
- Final packages: pkg-clover-topk-c (model gates), pkg-clover-topk-e (same dll/skel + fixed test-backend-ops). a/b/d deleted.
- Branch `clover-topk`: e511c98 (stable op), 78453af (full-row sort + view, spoof), bd98bbe (test fixture fix). Not pushed; `clover` untouched.
