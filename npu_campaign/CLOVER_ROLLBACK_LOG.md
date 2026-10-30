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
