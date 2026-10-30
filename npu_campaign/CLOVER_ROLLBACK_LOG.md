# CLOVER_ROLLBACK_LOG (agent `rollback`)

Append-only. Goal: remove the re-verify forward (and the checkpoint copies) of `llama-speculative-simple --spec-type draft-self`
on Clover-35B-A3B (recurrent hybrid), exactly. Worktree `llama.cpp-hexagon-clover-rollback`, branch `clover-rollback` (from `clover` 4838ffb).
Build `npu_campaign/build-clover-rollback.cmd` (dir `build-clover-rollback`, --parallel 4), package `pkg-clover-rollback.ps1 -Name <x>`
-> `npu_campaign/pkg-clover-rollback-<x>`. Runner: `npu_campaign/clover-rollback-ab.ps1` (interleaved ABAB, one run per NPU-lock hold,
CPU runs also take the lock per coordinator rule; dumps committed token ids via `LLAMA_SPEC_TOKENS_OUT`).
Token compare: `npu_campaign/clover_tokens_cmp.py REF CAND... -n 256`.

### 2026-10-29 23:5x - setup, code reading

- First CPU run of the unmodified build (`pkg-clover-rollback-base`) gave 4.5 tok/s: a `llama-bench` of another agent was running.
  Discarded. Coordinator rule since then: CPU model runs hold the NPU lock too.
- How the current example rolls back (recurrent memory cannot `seq_rm` partially, `n_rs_seq = 0`):
  draft pass = checkpoint save (PARTIAL_ONLY = recurrent part, 30 GDN layers x (2 MiB ssm + 96 KiB conv)) + masked 16-token
  non-causal decode + checkpoint load + `seq_rm`; then checkpoint save, verify [id_last + 15 draft]; on partial acceptance
  checkpoint load + `seq_rm` and a third forward over [id_last, k accepted, correction] which is then fully accepted
  (the only round counted -> `accept = 100%` artifact).
- Upstream already has bounded recurrent rollback: `cparams.n_rs_seq` widens the RS cache to (1 + n_rs_seq) planes,
  the causal GDN graph writes K = n_rs_seq + 1 per-token snapshots (`ggml_gated_delta_net(..., K)`, slot j = state j tokens back;
  CPU kernel is a sequential recurrence, so snapshots do not depend on batch size) plus K conv windows
  (`build_conv_state`), and `llama_memory_recurrent::seq_rm` turns a partial removal of <= n_rs_seq tokens into a pending
  `rs_idx` that the next graph's `s_copy` reads. `LLM_ARCH_CLOVER_MOE` is already in `llm_arch_supports_rs_rollback`;
  only `common_params_speculative::need_n_rs_seq()` leaves out DRAFT_SELF. HTP: GDN K>1 runs on the HVX recurrent kernel
  (HMX chunked only for K == 1), GDN+CPY fusion exists; conv windows are K separate CPYs per layer.
- The Clover non-causal (diffusion) graph `build_clover` reads conv + ssm state through `build_rs` but never writes them.
  The only reason the draft pass needs a checkpoint is metadata: `find_slot` advances the cell position and `s_copy`
  consumes the pending `rs_idx`.

Design:
1. `need_n_rs_seq()` includes DRAFT_SELF -> n_rs_seq = draft n_max (15): verify of 16 tokens keeps all 16 states,
   `common_context_can_seq_rm` reports RS, the example takes its normal (no-checkpoint) path: verify, accept k, `seq_rm(n_past, -1)`
   installs the state after the k-th draft token and trims the attention KV. No re-verify, no verify checkpoint.
2. `llama_context::decode`: for archs whose non-causal graph only reads the recurrent states (`llm_arch_noncausal_keeps_rs`,
   Clover only) save/restore the recurrent *metadata* (cells, rs_idx, head, used, n, rs_z; no tensor data) around the decode.
   The draft pass then leaves the recurrent memory as it was (incl. a pending rollback); `seq_rm(pos0, -1)` only trims attention KV.
3. draft-self probes once (first pass checkpoints; if `seq_pos_max < pos0` afterwards the pass kept the recurrent memory) and
   then stops checkpointing. `LLAMA_DRAFT_SELF_CKPT=1` forces the old per-pass checkpoint (A/B of item 3 inside one binary).
4. Negative control (off by default, spoof): `LLAMA_RS_ROLLBACK_NEGCTL=1` makes `seq_rm` install the state one token too late.

### 2026-10-30 00:10 - legacy time breakdown (CPU, diffusion; window later found shared with an unlocked HTP run -> indicative only)

`pkg-clover-rollback-prof0` = unmodified behavior + timers (commit 6273122). CPU 12t, exact greedy, n_max 15, 256 tokens.
17.09 tok/s (15.04 s): draft 48 x 120.2 ms (decode 111.2, argmax over 15 x 248k logits 4.2, ckpt save 2.4, restore 2.3),
verify 50 x 124.8 ms (16 tokens), re-verify 48 x 55.6 ms (k+2 tokens), verify ckpt save+load 4.6 ms/step, sampling 2 ms/step.
Share of time: draft 38%, verify 41%, re-verify 18%, checkpoints 3%. 0.568 forwards/token, 5.14 tokens per step.
Real acceptance (from the re-verify accounting, `#acc rate/pos` = P(k+1 >= i)): mean k = 3.2; P(k>=1..7) = .92 .74 .55 .41 .25 .16 .04.

Finding: the re-verify is NOT pure overhead. It re-processes [id_last, k accepted, correction c] and samples one more token b,
so a step commits k+2 tokens. Removing it (rollback) leaves k+1 tokens per step for D+V instead of k+2 for D+V+R:
win iff R/(D+V) > 1/(k+1) ~ 0.24; CPU has 55.6/245 = 0.23 -> about break-even on CPU. Better: rollback + a 1-token causal
decode of c ("bonus decode", LLAMA_SPEC_BONUS=1) gives the same k+2 tokens for D+V+C(1), C(1) = plain decode (22 ms CPU) < R.
Context: plain greedy CPU is 45 tok/s; draft-self pays ~10.7 plain-token costs (D+V) per step for k+2 ~ 5 tokens, because a
16-token MoE forward touches far more experts than a 1-token one. The width / verify-length sweep matters as much as rollback.

### 2026-10-30 00:25 - rollback implemented (commits 202e820, 04f8a81 on clover-rollback), first exactness evidence

Package `pkg-clover-rollback-rs2` (llama.dll C75B8A40, llama-common.dll BBDAE9E5). CPU, diffusion, 256 tokens:
- `draft-self: masked-block pass keeps the recurrent state, no checkpoint needed` (probe), RS cache 1005 MiB (16 planes),
  forwards draft 62 + verify 62 + plain 3, re-verify 0 -> 0.494 forwards/token (legacy 0.568); ckpt save/load 0 ms;
  draft pass 107 ms (decode 102.7, argmax 4.2), verify 126 ms (legacy 124.8: the 16 GDN snapshots cost ~1 ms on CPU).
  17.6 tok/s (not interleaved). Real acceptance: mean acc len 4.00, P(k>=i) = .875 .594 .469 .359 .219 .156 .094 .062 ...
- Token stream IDENTICAL to legacy prof0 for all 257 tokens (and the stdout text is byte-identical).
- Negative control `LLAMA_RS_ROLLBACK_NEGCTL=1` (state one token too late, conv + ssm): DIVERGES at token 16 (first partial
  acceptance boundary: ref 264 11991 7059, cand 279 7059 314) and degenerates into " equilibrium" repeated. The gate can fail.

### 2026-10-30 00:33 - HTP0 legacy breakdown (prof0, diffusion, -t 8, exact, f16 KV)

4.88 tok/s (52.65 s, 257 tokens): draft 65 x 341.9 ms (decode 334.3, argmax 3.3, ckpt 2.2 + 2.2), verify 68 x 277.3 ms,
re-verify 71 x 157.1 ms, verify ckpt save+load 4.4 ms/step, sampling 1.3 ms/step. Shares: draft 42%, verify 36%, re-verify 21%.
0.794 forwards/token, 3.78 tokens/step. HTP acceptance is lower than CPU: mean acc len 3.82 (CPU 5.2),
P(k+1>=i) = .97 .81 .52 .33 .13 .045 .015 (numerics gap, cf. KLD 0.048 in CLOVER_NPU.md M0). Host CPU 1.30 busy cores.
Prediction: R/(D+V) = 0.25 < 1/(k+1) = 0.36, so rollback alone loses on HTP; rollback + bonus decode should win (P1 ~ 97 ms < R).
Note: plain HTP decode is ~97 ms/token (10.3 tok/s) - with D+V ~ 620 ms per step self-spec cannot beat plain decode until the
forwards get cheaper (1162 / 762 splits per draft / verify graph today).

Infra: the A/B jobs died with `Cannot convert System.Object[] to Int32` / `owner.txt being used by another process`:
`npu-lock.ps1`'s stale-owner check reads owner.txt while another agent rewrites it. It throws inside Enter-NpuLock before any
run starts, so `clover-rollback-ab.ps1` now retries those two failures (shared lock script left untouched).

### 2026-10-30 00:50 - research knobs (all off by default, exact verification unchanged)

- `LLAMA_SPEC_BONUS=1` (example, 04f8a81): after a partial acceptance the next step is a plain 1-token decode of the correction
  token c (gets the token the legacy re-verify produced, without re-running the k accepted tokens).
- `LLAMA_SPEC_VERIFY_MAX=n` (example): draft/verify at most n tokens, draft-self block width stays n_max + 1.
- `LLAMA_DRAFT_SELF_MARGIN=x` (draft-self, e17e677, pkg rs3): the argmax scan also tracks the runner-up logit; the draft ends at the
  first position with top1 - top2 < x (adaptive verify length).
All of them only change which tokens get verified, never the acceptance rule, so every config must reproduce the legacy greedy stream.
Queued (interleaved, one run per lock hold): HTP A/B htp2 (legacy/rs/rs+bonus x2), CPU A/B ab2 (3 prompts x2), CPU sweep sw2,
HTP sweep hsw1 (verify cap 5/3, width 8), HTP margin trial hmg1, HTP placement audit audit1.

### 2026-10-30 00:55 - HTP placement audit of the rollback graphs (rs2, GGML_SCHED_DEBUG=2, 24 tokens, clover_sched_audit.py)

draft 4251 nodes / 1162 splits (CPU: CPY 462, CUSTOM 120, GET_ROWS 2), causal verify 3125 / 762 (CPY 566, CUSTOM 40, GET_ROWS 2),
decode 2945 / 582 (CPY 446, CUSTOM 40, GET_ROWS 2): the same split counts and CPU op sets as the M0 audit. The K=16 GDN (HVX
recurrent kernel, snapshots fused into the cache write) and the 16 conv-window CPYs per GDN layer (HTP gather-rows copy) all run
on HTP0; the rollback adds no CPU fallback.

### 2026-10-30 01:10 - lock race: runs 00:40-01:09 discarded, queue restarted

At 00:55:43 three of my waiters (one HTP, two CPU, from three A/B scripts queued in parallel) held `.npu.lock` at the same time:
`New-Item -ItemType Directory` succeeds on an existing directory, so several waiters "won" in one tick. Reported to the coordinator,
who switched `npu-lock.ps1` to an atomic `CreateDirectoryW` (01:09:59). All my jobs were killed; every result from that window
(htp2, ab2, sw2, hsw1, hmg1) is discarded. From here on: one A/B script at a time (`rb-queue1.ps1`: hab1 HTP A/B, cab1 CPU A/B
3 prompts, cneg negative control, csw width sweep), and `clover-rollback-ab.ps1` samples the process list every 2 s and tags each
row `clean` or `OVERLAP(pids)` when a non-rollback llama-* / test-backend* process ran during the run.

### 2026-10-30 01:46 - HTP0 interleaved A/B hab1 (diffusion, exact greedy, n_max 15, -t 8, 2 rounds, order reversed in round 2)

| config | tok/s r1, r2 | mean | tokens/step | D ms | V ms | R ms | P ms (1-token) | forwards | host CPU cores |
|---|---|---|---|---|---|---|---|---|---|
| legacy (prof0) | 4.84, 4.99 | 4.91 | 3.78 | 339 | 274 | 158 | - | 65 D + 68 V + 71 R = 0.794 /token | 1.13, 0.99 |
| rollback (rs3) | 4.73*, 4.78 | 4.75 | 3.06 | 331 | 310 | 0 | 104 | 84 D + 84 V + 2 P = 0.661 /token | 1.32, 1.22 |
| rollback + bonus decode (rs3, LLAMA_SPEC_BONUS=1) | 5.35, 5.37 | **5.36 (+9%)** | 4.02 | 334 | 306 | 0 | 105 | 64 D + 64 V + 66 P = 0.755 /token | 0.98, 0.99 |

(* rs r1: another agent's orphaned HTP process was still in teardown during my model load, gone before generation.)
- As predicted, rollback alone loses on HTP (R/(D+V) = 0.26 < 1/(k+1)): it saves R and the checkpoints but commits k+1
  instead of k+2 tokens per step. Rollback + bonus decode keeps k+2 tokens per step and replaces R (158 ms, k+2 tokens)
  by a 1-token decode (105 ms): 745 ms per 4.02 tokens vs 780 ms per 3.78.
- Cost of rollback on HTP: verify +32 ms (306 vs 274: 16 GDN snapshots of 2 MiB per layer = 960 MiB written per verify via the
  fused GDN+CPY, HVX recurrent kernel instead of the HMX chunked one, plus 16 conv-window copies per layer). Draft pass without the
  checkpoint: 334 vs 339 ms. Host CPU drops from ~1.06 to 0.99 busy cores (no 4 checkpoint copies of ~63 MiB per step).
- HTP streams are deterministic run to run (legacy r1 = r2, rs r1 = r2, rsbonus r1 = r2 for 256 tokens), but they differ between
  methods (legacy vs rs at token 4, rs vs rsbonus at token 58): on HTP the logits of a position depend on the batch shape it was
  computed in (16-token verify vs k+2 re-verify vs 1-token decode; chunked vs recurrent GDN kernel). The exactness gate is the CPU one.
- Context: a plain 1-token HTP decode is 105 ms (9.5 tok/s); every self-spec config is still slower than plain decode on HTP,
  because D + V (640 ms) cost ~6 plain decodes for ~3-4 tokens.

### 2026-10-30 02:35 - FIFO lock era; first clean CPU numbers; snapshot cost; verify cap on HTP

- Lock: the bf16 agent's 1 ms fast-poll starved the 200 ms waiters (my queued runs waited 20+ min, 01:49-02:13). The coordinator
  made `npu-lock.ps1` a FIFO ticket lock (7c47cf6). All my waiters were relaunched, so every row below took a ticket.
  The rollback/legacy split is the same in every package; rs4 = rs3 + 05c2715 (only copy the conv windows a rollback can read).
- Clean CPU legacy (cab1 r1 / cg r1, diffusion): 20.24 / 19.45 tok/s. D 111.1 ms (decode 102.8, argmax 3.7, ckpt 2.3 + 2.3),
  V 93.1 ms, R 49.8 ms, verify checkpoints 4.5 ms/step, sampling 1.6 ms/step. Shares: draft 42%, verify 37%, re-verify 19%, ckpt 3.5%.
  (The 00:10 breakdown had V = 125 ms because another process was running.)
- CPU rs4 (rollback, no bonus): 17.97 tok/s. The token stream is IDENTICAL to legacy for all 256 tokens. D 104.8 ms (no checkpoint),
  but V = 123.8 ms, **+31 ms over legacy**: with n_rs_seq = 15 the CPU GDN writes 16 snapshots x 2 MiB x 30 layers = 960 MiB
  into its output, and a CPY then moves them into the cache. That is ~2.9 GB of traffic per verify, at ~100 GB/s.
  HTP fuses GDN+CPY, yet still pays +32 ms (HVX recurrent kernel instead of HMX chunked, plus the 960 MiB of writes).
  The snapshots are the real price of the rollback. The verify-length cap is the lever, since the op only writes min(n_tokens, 16) snapshots.
- HTP rollback + bonus + `LLAMA_SPEC_VERIFY_MAX=5` (rs3, 1 round, clean in the generation window): **6.41 tok/s** (rsbonus n_max 15:
  5.36-5.43, legacy 4.91). V = 223 ms (6 tokens, 6 snapshots) vs 306 ms; D unchanged at 326 ms (the block stays 16 wide);
  4.15 tokens/step. Interleaved A/B with verify caps 5 and 3 against legacy is queued (hab3).

### 2026-10-30 03:35 - HTP A/B hab2, HTP width sweep, CPU code prompt: bonus decode is not bitwise identical

- hab2 (HTP0, interleaved, 2 rounds, clean): legacy 4.91, 5.01; rs4 + bonus 5.41, 5.48 (+10%). Its stream is identical to the rs3 + bonus
  stream (05c2715 does not change the numerics), and the 1-token decode drops from 105.5 to 101.8 ms (1 conv-window CPY per layer instead of 16).
- HTP sweep, 1 round each (rollback + bonus, diffusion), per-step costs:

| config | tok/s | D ms (block) | V ms (rows) | P ms | tokens/step |
|---|---|---|---|---|---|
| n_max 15 | 5.41-5.48 | 331 (16) | 306 (16) | 102 | 4.02 |
| n_max 15, verify <= 5 (rs3) | 6.41 | 326 (16) | 223 (6) | 103 | 4.15 |
| n_max 7 | 6.18 | 296 (8) | 246 (8) | 102 | 4.02 |
| n_max 5 | **6.72** | 276 (6) | 217 (6) | 101 | 3.95 |

  On HTP, acceptance barely depends on the block width (P(k >= 1..4) = .42 .26 .15 .09), so a narrow block and a short verify only
  remove cost. The masked draft forward keeps a large fixed cost (276 ms for 6 tokens = 2.7 plain decodes): its graph has 1162 splits
  vs 582 for a decode (120 CPU CUSTOM ops vs 40), which is the bf16/topk agents' area.
- CPU code prompt (cg r1): legacy 25.89 tok/s, rollback 20.59 (IDENTICAL stream, V 129 vs 92 ms), rollback + bonus 24.56.
  The rollback + bonus run overlapped an unlocked test-backend-ops (bf16, 03:22:33-39), so that timing is discarded.
  Its stream **DIVERGES from legacy at token 246/256** (ref 1954 30, cand 6249 13), which is not a rollback-state error:
  ggml-cpu's repacked Q4_0 matmul (repack.cpp forward_mul_mat_one_chunk) runs gemm on groups of 4 rows and gemv on the remaining rows,
  with different float accumulation. A token's logits therefore depend on the batch shape it is computed in. The bonus token comes from
  a 1-token decode, while in legacy it comes from row k+1 of the (k+2)-row re-verify; a near-tie flipped. Rollback without bonus
  takes every token from a verify row, the same as legacy, and has matched it so far. Evidence that legacy itself depends on batch shape
  is queued (legacy n_max 11 and plain decode on code).
- c4535cf (CPU): GDN with K > 1 now writes the snapshots straight into the cache when its next non-view node is the CPY of the
  snapshot tail (same floats; HTP already had fuse_gdn_cpy). Package rs5; validation queued.
