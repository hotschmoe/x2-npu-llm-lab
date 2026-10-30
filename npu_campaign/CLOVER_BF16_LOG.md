# Clover bf16 campaign log (agent `bf16`, branch `clover-bf16`, worktree llama.cpp-hexagon-clover-bf16)

Append-only. Charter: CLOVER_NPU.md. Goal: every build_bf16_round CPY (F32->BF16, BF16->F32) on HTP0, then nearly free.

### 2026-10-30 00:00 - setup + shape inventory
- worktree `llama.cpp-hexagon-clover-bf16` (branch `clover-bf16` from `clover` 4838ffb); scripts `build-clover-bf16.cmd` (parallel 4 per
  coordinator), `pkg-clover-bf16.ps1` -> `pkg-clover-bf16-<x>`. Launch builds from PowerShell with `cmd /c ".\build-clover-bf16.cmd > log 2>&1"`.
- Shape inventory from the baseline HTP run with `GGML_HEXAGON_VERBOSE=1` (`clover-logs/bf16-shapes.err`, `supports-op CPY|...|no` lines), unique
  rejected CPY shapes (ne : count per pair of casts), ALL fully contiguous, same-shape F32<->BF16:
  2048x1, 2048x256, 2048x16, 4096x1/256, 8192x1/256, 128x32, 128x32x256, 128x16x256, 1x32x256 (ne0 = 1!), 256x2, 256x16, 256x2x256, 256x16x256.
  Only other CPU ops: GGML_OP_CUSTOM (clover_sort_rows, owned by `topk`).
- Plan: flat element-range convert kernel (dense src+dst, any shape), row-strided path for padded/permuted views, then pair fusion, then epilogues.

### 2026-10-30 00:40 - kernels v1 (commit d905ddb on clover-bf16)
- Hypothesis: all build_bf16_round CPYs are dense same-shape F32<->BF16 -> one flat element-range convert kernel (HVX int RNE, same arithmetic as
  ggml_compute_fp32_to_bf16: (u + 0x7fff + ((u>>16)&1)) >> 16, NaN -> (u>>16)|64) covers all; row-strided path (SAMESHAPE_CONVERT + bf16) for padded/permuted views.
- Change: `hvx-copy.h` (hvx_vec_round_bf16_f32, bf16<->f32 row fns), `cpy-ops.{h,c}` (HTP_COPY_KERNEL_CONTIG_CONVERT, flags ROUND/TRUNC, bf16 same-type reshape),
  `htp-kparams.cpp` (htp_kparams_solve_cpy_ex; BF16 accepted in supports_op), `htp-fusion.cpp` (fuse_cpy_bf16_round, flag HTP_FUSE_CPY_BF16_ROUND=8192; disable
  only it with GGML_HEXAGON_OPFUSION=8190), `tests/test-backend-ops.cpp` (`test_cpy_bf16`, `add_test_cases_clover()` in eval+perf, `allow_nan()` hook, bit-exact err()).
- Result (unfused, `bf16-oracle-a1.log`): 108 clover cases, 102 OK bit-exact (NMSE 0 + zero differing bit patterns incl. ties, NaN payload, inf, denormals,
  overflow-by-rounding, ne0 = 1/3/31/33/63/65, widths 1/16/256, padded + permuted(0,2,1,3) views, 1M elements multi-thread), 6 not supported (full transpose (1,0,2,3)).
- Negative control `GGML_HEXAGON_BF16_SPOIL=1` (truncate instead of RNE, OFF by default): 71 FAIL (inf mismatch -3.39e38 vs -inf, etc.) -> test has teeth.
- Pitfall: PowerShell env vars set inside one tool call leak into later commands of the same call (the spoil switch made my next oracle fail); scripts now restore env.
- Fusion first attempt did not fire: CPY nodes carry inputs {src, self-dst}; the allocator reuses the dead mid buffer for the final dst, which tripped the
  overlap check against prev.inputs[1]. Fix: check overlap only against the real source.

### 2026-10-30 02:05 - fusion engaged, oracle + placement audit (pkg-clover-bf16-a)
- Fusion root cause: ggml_cast sets src[1]=itself, so the BF16 mid node has 2 recorded uses (self + reader); the FUSEABLE tag needs n_uses==2 for that
  pattern. Also overlap check must only compare dst vs the real source. After both fixes `GGML_HEXAGON_VERBOSE=1` prints `fused CPY+CPY (N=2)`.
- Oracle bundle (`bf16-bundleA-a3`, one lock hold, `clover-logs/bf16-oracle-a3-*.log`): 108 cases = 102 OK + 6 not supported (full transpose (1,0,2,3) source); fused (default): 102/102 bit-exact; unfused (`GGML_HEXAGON_OPFUSION=8190`): 102/102 bit-exact; spoil switch (truncate): 68 FAIL / 34 OK
  (the OK ones are shapes whose random data never hits a tie/overflow-rounding case or BF16->F32 which is exact) -> negative control fails as required.
- Placement audit (`bf16-sched-a3.err`, `clover_sched_audit.py`), baseline -> now: CPY on CPU 462/566/446 -> 0 in every graph.
  splits: diffusion 1162 -> 242, verify 762 -> 82, decode 582 -> 82. Remaining CPU ops: CUSTOM (120 / 40 / 40, `topk` agent), GET_ROWS 2 (token embedding),
  plus in the n_outputs-sliced verify graph (98 splits) the last layer's tail ops ADD x10, MUL x5, DIV x1 after the CPU GET_ROWS of the output rows
  (pre-existing, not bf16 related: they follow the CPU get_rows by placement; worth a look by whoever owns output-row selection).
- Standalone perf (test-backend-ops perf, NOT a gate, lock was contended): single CPY 3.7 us (2048x1) .. 29 us (2048x256) .. 96 us (8192x256); a fused pair costs the
  same as one single CPY; the unfused pair shows the same number, i.e. the perf harness is dominated by per-graph overhead and cannot resolve per-op cost. Use the
  model-level A/B + per-op profile for the real numbers.
- Lock note: other agents' scripts re-acquire the lock within ms, a 200 ms poll starved my waiter for ~1 h; `bf16-fastlock.ps1` polls at 1 ms.

### 2026-10-30 02:20 - KLD gate + first A/B (pkg-clover-bf16-a, fused pair, flat kernels)
- KLD (clover-kld.ps1, `kld-bf16-a.log`): mean KLD 0.047923 +- 0.002348, same-top 89.216%, PPL(Q) 15.083178 (base CPU 15.441317): identical to the
  baseline HTP numbers (0.0479 / 89.2% / 15.0832). No degradation. (Exact baseline digits to follow for a bit-level comparison.)
- A/B round 1 (interleaved A=pkg-clover-base, B=pkg-clover-bf16-a, `clover-run.ps1 -Accept ''`, 256 tokens, draft-self n_max 15; lock held, no other model run):
  A1: 4.857 tok/s, plain decode `eval time` 105.70 ms/token; B1: 8.221 tok/s, plain decode 52.21 ms/token (note: eval-time line covers 1-2 runs only, noisy);
  A host CPU 1.11 busy cores.
  The generated text differs between A and B from token 0 (B: "The answer is that diffusion ..."); expected cause: the former CPU CPY splits kept
  neighbouring HTP ops in different batches, so batch-level fusions (binary_nx, rms_norm_mul, ...) now fire in new places; numerics of fused kernels differ in the last bits
  at near-tie tokens. KLD above says the distribution is unchanged. To be verified: KLD with pair fusion off.
- Process note: the first ab1 hold used a private fast-poll lock acquire (bf16-fastlock.ps1); coordinator asked to stop (it starved other agents) -> removed; ab1 was cut
  after B1 (hold would have exceeded 10 min); the rest of the ABAB is done in separate FIFO-lock holds.

### 2026-10-30 03:25 - PROBLEM: run-to-run instability with CPY on HTP (bisecting)
- A/B tok/s (ab1/ab2): base 4.857/4.919 tok/s (plain eval 105.7/97.9 ms/token, deterministic text); bf16-a 8.221 tok/s (257 tokens) but ab2-B only 85 tokens
  (5.86 tok/s) and **its text differs from run to run** (first token "The answer is", "on my website", "Create a diffusion", "We need to answer", ...) while base
  repeats exactly. KLD 4x512 is bit-identical to base (0.047923 / 15.083178, both pkgs) so the 512-token prefill graph is fine; the 13-token prompt + draft graphs are not.
  Treat all speed numbers of bf16-a as INVALID until this is solved (outputs are partly garbage).
- Bisection (16-token spec runs, compare first tokens to base "We need answer. Need define diffusion."):
  OPBATCH=1 (flush after every op): 2/2 identical to base -> inter-op hazard in a batch.  OPFUSION=0 and OPFUSION=8190: still unstable -> not fusion.
  BF16_NOFLAT=1 (row path): unstable.  OPFILTER=CPY: crashes at startup (ggml-backend.cpp:1089 pre-allocated tensor in HTP0 buffer cannot ... - state-copy CPYs must stay on HTP).
  New diagnostic envs: GGML_HEXAGON_BF16_MAX_ELEMS / _MIN_ELEMS (claim bf16 casts only inside the range): MAX=8192 -> identical to base text; MIN=8193 -> garbage.
  => the large casts (> 8192 elements, multi-thread) trigger it; small ones are fine. Standalone oracle (single-op graphs) is bit exact for those shapes, so it is an
  in-batch interaction. Next: NTH cap, BLK cap (GGML_HEXAGON_BF16_NTH / _BLK).
- Lock note: the test-backend-ops runs flagged at 03:22 were not launched by this agent (all bf16 launches go through Enter-NpuLock / NPU_LOCK_HELD holds).

### 2026-10-30 04:15 - bisection continues (16-token spec runs; "= base" means text equals "We need answer. Need define diffusion. User asks ...")
- NTH=1 (all casts single-thread): garbage -> not a multi-thread race. NTH=4, BLK=1024: garbage.
- Size windows (HTP claims only casts with MIN..MAX elements, rest stay on CPU): <=8192 = base; 8193..40000 = base; 40001..300000 = base; >300000 = base;
  but >= 8193 (union of the three windows) = garbage. Name classes (src tensor name): {l_out, attn_residual, ffn_moe_out} = base; {attn_output, attn_pregate} = base;
  {node_*, model.input_embed} = base; all together = garbage.  OPBATCH=16 and 128: garbage (OPBATCH=1: base).
  => not a single bad cast shape, not capacity of the batch: the failure needs MANY HTP-resident casts in the same chain (each CPU split between casts acts as a full
  cache flush). Prime suspect: DSP data-cache coherency of DMA-written tensors (htp-dcache dirty-range tracker; lingering dirty lines in memory the allocator reuses ->
  write-back over DMA-written data), which OPBATCH=1 (qurt_mem_cache_clean at every batch) hides.  Test of mm consumers (state recall chain, `CPY_BF16_MM` cases, 16/16 OK) rules out the BF16 mul_mat consumer.
- Experiment queued: GGML_HEXAGON_BF16_PREFLUSH=1 (the CPY kernel does dccleaninva on its dst range before writing).

### 2026-10-30 09:55 - status: kernels + fusion done and bit-exact; model-level instability under investigation (possibly contaminated by a stray HTP session)
- State: branch clover-bf16 = d905ddb (kernels) + 86f2a17 (fusion tag fix, diagnostics, chain tests) + merge of clover (topk 0 CUSTOM, trace tools) + merge of rollback clover (clean merge, tests file uncommitted adds CPY_BF16_GDN/STACK).
- Placement audit (pkg-clover-bf16-a): CPY on CPU 462/566/446 -> 0; splits 1162/762/582 -> 242/82/82 (remaining CUSTOM, now removed by topk merge).
- Oracle: 102/102 bit-exact fused and unfused; spoil switch fails 68 -> negative control OK. KLD (old CPU ref): 0.047923 / 15.083178 identical to base.
- Open: spec runs on the all-HTP package gave run-to-run different text (base is stable). Bisection found: stable for every subset of casts, unstable only when all
  cast groups are on HTP; OPBATCH=1 stable; trace runs that sync at the GDN inputs are stable (GDN output of layer 0 good), runs that batch GDN with its feeding
  rounds/cont/repeat have a bad GDN output (nmse 4e-2 vs 3e-6) -> hazard between the cast->repeat->cont chain and the gated_delta_net op, or a second HTP session
  (coordinator: a hung port-agent process sat on HTP0 09:16-09:50 and other unlocked processes at 03:22). All instability conclusions must be re-checked in a clean window:
  queued `bf16-hold2` = GDN-front-end synthetic test (x3 + OPBATCH=1 control) and two clean 16-token determinism runs with a stray-process check.
- Lock lesson: a waiting script whose FIFO ticket was pruned while the machine slept (06:52 -> 09:11 clock gap) never re-queued; relaunch if no ticket in .npu.lock.queue.

### 2026-10-30 10:50 - ROOT CAUSE LOCATED (cache coherency of DMA-written bf16 rounds), fix candidates
- Trace bisection on layer 0 (CPU trace vs HTP trace, `trace_nmse_table.py`): with sync points only at l_out the GDN output attn_output-0 is wrong (nmse 4e-2..7e-2 vs 3e-6),
  with a sync after the q/k REPEAT outputs (x2/y2) or after the two l2-norm `cont`s (y1) it is exact. Observing only k_conv_predelta (y3) shows k_conv_predelta itself wrong
  (8e-2). I.e. the k chain `cont -> sqr -> sum_rows -> scale_bias -> sqrt -> div -> [round pair, ours] -> REPEAT` is corrupted when it runs in the same DSP batch behind
  the q chain. `repeat-ops.c` copies with plain `memcpy` (cached loads/stores, no DMA) the tensor our kernel just wrote with DMA.
- Synthetic graphs (round -> repeat -> GDN, chains of 8 layers, bf16 mul_mat consumers) all pass, so the hazard needs the real batch history.
- Switches in the cast kernels (diagnostic env, OFF by default): PREFLUSH (dccleaninva of dst before the DMA write): garbage as before. POSTFLUSH (dccleaninva of dst right after the
  DMA completes) and FLUSHALL (qurt_mem_cache_clean after the op): coherent text ("We need answer ..."), control garbage. => the consumer sees stale cache lines / not-yet-visible
  DMA writes of the rounded tensor; the framework's dirty-range tracker does not make that safe for a DMA-writing kernel followed by a memcpy (cached) kernel.
- Next: confirm POSTFLUSH reproduces the baseline text deterministically (queued f2), then make a range post-flush the default for bf16 cast kernels.

### 2026-10-30 11:25 - FINAL STATE (agent bf16)

Branch `clover-bf16` (worktree llama.cpp-hexagon-clover-bf16), commits: d905ddb kernels+fusion+tests, 86f2a17 fusion-tag fix + diagnostics, 9b36e69 merge clover (topk/trace),
834c734 merge clover (rollback), ccdfcce POSTFLUSH default + chain tests, c90b500 bf16 CPY on HTP opt-in. Packages: `pkg-clover-bf16-final` (= code at c90b500), `pkg-clover-bf16-k`
(same kernels, casts ON by default, used for the A/B and KLD below), older a..j for the bisection.

**What works (valid, bit-exact):**
- HTP kernels CPY F32->BF16 (RNE, NaN quieting, = ggml_compute_fp32_to_bf16), BF16->F32, flat dense path + row path for padded/permuted views; BF16 same-type copy; fused
  CPY+CPY -> one F32->F32 rounding pass (`HTP_FUSE_CPY_BF16_ROUND`, disable alone with GGML_HEXAGON_OPFUSION=8190). Oracle `test-backend-ops -b HTP0` (CPY / CPY_BF16_ROUND cases from
  Clover shapes): 102/102 bit-exact fused and unfused (108 cases, 6 unsupported = full transpose); negative control GGML_HEXAGON_BF16_SPOIL=1 (truncation) fails 68 cases.
  Chain tests (round->mul_mat, 8-layer stack, round->l2norm->repeat->GDN front end) pass vs CPU.
- Placement audit with casts on HTP: CPY on CPU 462/566/446 -> 0, splits 1162/762/582 -> 242/82/82 (before the topk merge; only CUSTOM top-k remained).
- KLD gate (new CPU-f32dense reference, clover-kld.ps1, causal 4x512, casts on HTP): mean KLD 0.019871, same-top 93.137%, PPL 15.083178 -> GATE PASS, identical to M0 HTP0 (0.0199/93.1%/15.083178).
- Host CPU while generating: 0.18-0.24 busy cores with casts on HTP vs 6.5 cores with casts on CPU (same package, MAX_ELEMS=1 switch) - the NPU-only goal is met in that sense.

**What does NOT work (blocking):** end-to-end generation with ALL casts on HTP is not repeatable. `fin` A/B (same package k, A = GGML_HEXAGON_BF16_MAX_ELEMS=1 = casts on CPU = merged-clover behaviour,
B = casts on HTP + dst cache flush), exact greedy verify, 256 tokens, interleaved A B A B, FIFO lock held, no stray process at start (hold2 checked):
  A1 5.027 tok/s, A2 4.912 tok/s, texts byte-identical, mean acc len 3.01 (deterministic);  B1 8.787 tok/s, B2 16.696 tok/s, texts DIFFERENT (B1 coherent "The user is asking about diffusion...",
  B2 degenerate repetition "Short-term answer: ..."), mean acc len 3.94 / 7.31. => the B speeds are NOT valid results (B2's 16.7 tok/s is a repetition loop with a high draft acceptance).
  Valid A-only numbers: ~5.0 tok/s; plain-decode steps not measured cleanly (eval-time lines cover 1 run: A 90 ms/token, B1 71 ms/token; unreliable).
Earlier unlocked/contaminated windows (my 02:20 ab1/ab2: base 4.86/4.92 tok/s deterministic, bf16-a 8.2 and 5.9 tok/s non-deterministic) agree with this picture.

**Instability: root cause analysis (best hypothesis, not closed).**
- Not the orphan: the coordinator's 09:16-09:50 orphan overlaps only my w1..w3 trace runs (their conclusions are consistent with the clean ones: x*, y* ran 10:04-10:15, `hold2`
  determinism runs at 09:58 started with 0 stray HTP processes, and the final `fin` runs at 11:03-11:19 were clean) and the instability is reproduced in clean windows. Earlier mid-night windows
  had a few unlocked test-backend-ops runs of other agents (03:22) but the instability appears in every clean window as well.
- Needs all cast groups on HTP at once (any subset, any size window, any name class: stable; union: unstable); OPBATCH=1 (cache flush after every op) stable; OPBATCH=16/128 unstable;
  not fusion (OPFUSION=0 unstable), not threads (NTH=1 unstable), not kernel arithmetic (oracle exact).
- Trace bisection of layer 0 (causal prefill, 13 tokens, CPU trace vs HTP trace): the k chain `cont -> sqr -> sum_rows -> scale_bias -> sqrt -> div -> [our round] -> REPEAT` produces a wrong
  k_conv_predelta (nmse 8e-2) when it runs in the same DSP batch behind the q chain; a sync (batch end = full dcache clean) after the l2-norm conts, after q_conv_predelta, or after both
  predelta tensors makes it exact. `repeat-ops.c` uses plain `memcpy` (cached loads/stores) on the tensor our kernel wrote with DMA; the dirty-range tracker (htp-dcache.c) is supposed to cover
  that but the fault shows up only with real batch history. Cache flush switches in the cast kernels: PREFLUSH (before DMA) no effect; POSTFLUSH (range flush after the DMA, now default
  for casts) and FLUSHALL gave coherent text in single 16-token probes, but the 256-token A/B above shows POSTFLUSH alone does NOT make the run repeatable. So either more consumers read DMA-written data
  through the cache (REPEAT is the proven one; other cached-store/-load ops unknown) or the hazard is DSP-side ordering of the DMA completion vs the next op rather than stale lines.
- Suggested next step for whoever continues: make REPEAT (and any op using memcpy/cached HVX loads on activations) invalidate its source range itself / use DMA; then rerun
  `bf16-bundleC.ps1 -PkgA k -PkgB k -EnvA GGML_HEXAGON_BF16_MAX_ELEMS=1 -Rounds 2` and `bf16-trace2.ps1` (HTP vs HTP trace repeat) before trusting any speed number.

**Recommendation:** merge the branch for the infrastructure (kernels, fusion, tests, diagnostics) with bf16 CPY on HTP OFF by default (this is what c90b500 does: GGML_HEXAGON_BF16_CPY=1 opts in; with it
off the behaviour equals `clover`). Do not enable it in any installed build until the A/B above gives identical text across repeats (B1==B2) and equals A or has KLD-equivalent quality.
The potential, if the hazard is fixed: all 446-566 CPYs/graph on HTP, splits 582 -> 82 (-> ~1 with topk on HTP), host CPU 6.5 -> 0.2 cores, draft-self step time roughly halved.
Not done: producer-epilogue fusion (round folded into ADD/MUL/SILU; estimated <2% of decode since a fused pair costs ~3.7 us/op), strided full-transpose CPY, final oracle rerun on
`pkg-clover-bf16-final` is queued (bf16-suite-fin.out) - see clover-logs.

- 11:40 final-package oracle (`pkg-clover-bf16-final`, c90b500, run with GGML_HEXAGON_BF16_CPY=1 as required now): fused 102/102 OK, unfused (OPFUSION=8190) 102/102 OK, spoil (truncation) 68 FAIL / 34 OK, 6 not supported (full transpose).
  Default (no env) the bf16 casts are not claimed by HTP0, i.e. behaviour = `clover`.
