# MTP speedup campaign log (append-only)

Goal: make MTP speculative decoding a net decode-speed win on Qwen3.6-35B-A3B Q4_0 (HTP0), correctness preserved.
Worktree: `llama.cpp-hexagon-mtp` (branch `mtp-speedup`, from `d180917`). Build: `npu_campaign/build-mtp.cmd` -> `build-mtp`.
Runner: `npu_campaign/run-mtp-speed.ps1` (takes the NPU lock; artifacts in `npu_campaign/mtp-speed/`).
Profile summarizer: `npu_campaign/mtp_prof_cmp.py` (groups OPBATCH records by op count; compares classes across logs).
Model for all MTP/plain comparisons: expanded `Qwen_Qwen3.6-35B-A3B-Q4_0-mtp-f32.gguf` (E8E27D94...), policy = run-mtp-speed defaults
(OPBATCH 3072, replay 1, fusion 1, router HVX, FA_SELECT 1, GDN_SELECT 1, chat `Hello`, seed 1234, temp 0, -c 1024 -b/-ub 128).

## 2026-09-29 00:xx  A. Cost model (installed pkg-wos: host C9E4, DSP 640AA38D)

llama-bench pp1..pp8 (plain model, 5 reps; one output row): pp1 28.32 t/s (35.3 ms), pp2 37.36 (53.5 ms / 2 tok),
pp3 45.79 (65.5 ms), pp4 52.61 (76.0 ms), pp8 79.56 (100.5 ms). So an M=2 forward is ~1.52x M=1 with *no* rollback snapshots.

Untraced llama-cli Hello128: plain 28.11 tok/s (35.58 ms/tok). MTP draft-max 1: 24.23 tok/s, acceptance 52/74 (0.70).
LLAMA_MTP_PROFILE: verify (2-token target decode) ~62 ms wall, catch-up 2.1-2.3 ms, draft 6.0 ms.
Step = 128 tok / 75 steps -> ~70.8 ms per step, 1.70 tokens per step.

Device profile (GGML_HEXAGON_PROFILE=1, 16 tokens, replay disabled by profiling), DSP time per batch class:
| graph | ops | DSP ms |
| --- | ---: | ---: |
| plain decode M=1 | 1384 | 33.65 |
| MTP verify M=2 (K=2 rollback snapshots) | 2024 | 58.7 (median), 61.0 mean |
| MTP draft (1 row) | 40 | 5.31 |
| MTP catch-up (2 rows) | 48 | 1.52 |
Host overhead ~3 ms/step (MTP) vs ~1.9 ms/token (plain). Graph replay (single-slot) never hits in MTP mode: target, draft and
catch-up graphs alternate on the same HTP session, so every step re-enqueues ~2100 op descriptors.

Where M=2 verify (58.7) exceeds M=1 (33.65), per op class (ms per graph):
| item | M=1 | M=2 | note |
| --- | ---: | ---: | --- |
| conv window: SSM_CONV+WIN (M=1) vs CONCAT 3.64 + CPY(60) 2.96 + SSM_CONV 2.20 | 1.41 | 8.80 | decode-only fusion, K=2 snapshot copies |
| MoE sum / shexp gate / router norm: ADD+NW, ADD+SG, MOE+WNORM vs ADD(430) MUL DIV CLAMP SUM_ROWS SIGMOID | ~0.9 | ~4.4 | decode-only fusions |
| MUL_MAT+ADD (ssm_out, attn_output, ssm_alpha) | fused | unfused | M=1 only for HVX |
| experts gate/up NX + down | 7.30 | 13.07 | ~2x weights (two tokens' experts) |
| GDN (hvx-recurrent-scalar), K=2 snapshots | 1.76 | 4.05 | |
| dense matmuls (qkv, head, ssm_out, attn_gate, shexp) | ~12 | ~14.5 | |
| FA, ROPE, RMS_NORM+MUL, SOFT_MAX, SWIGLU | small | +~1.5 | |

Draft graph 5.31 ms: output head (Q6_K, 248320 x 2048, 417 MB) 2.95 ms + TOP_K over 248k logits 1.01 ms + PAD 0.18; the MTP
block itself ~1.1 ms. Catch-up 1.52 ms is the MTP block on 2 rows (no head).

Policy sweep, untraced Hello128 (all slower than plain 28.1):
| policy | tok/s | acceptance |
| --- | ---: | --- |
| draft-max 1 | 24.23 | 52/74 |
| draft-max 2 | 23.76 | 71/112 |
| draft-max 3 | 21.67 | 76/152 |
| dmax 1, p-min 0.5 | 24.51 | 50/63 |
| dmax 1, p-min 0.8 | 23.90 | 43/50 |
| dmax 2, p-min 0.5 | 24.99 | 67/92 |

Break-even: with verify V, draft+catch-up D, host H and mean tokens/step L, MTP wins when (V+D+H)/L < 35.6 ms.
Today (61+6.8+3)/1.70 = 41.7 ms. Targets (estimates): conv-window M=2 fusion -6 ms, MoE-sum/router/shexp M=2 fusions -3 ms,
catch-up merged into draft -2 ms, draft head over a truncated vocabulary -3 ms, multi-slot graph replay -1..2 ms
=> ~55 ms/step => ~31 tok/s (+10%) at 0.70 acceptance. Plan: implement in that order of (gain/effort), host-only items first.

## 2026-09-29 00:50  B1. Multi-token conv window fusion (SSM_CONV+WINK)

Hypothesis: the M=2 verify pays CONCAT + 2 snapshot CPYs + unfused SSM_CONV per GDN layer (8.8 ms/verify) because the
decode conv-window fusion only matches T=1 with a single CPY. Change (worktree `mtp-speedup`): new DSP mode
`fused_window=3` in `htp/ssm-conv.c` (T<=4 tokens, up to 3 snapshot outputs, same qf32 MAC order as the unfused kernel)
and host matcher `try_fuse_ssm_conv_windowk` (fusion bit 1<<13, `GGML_HEXAGON_FUSE_SSM_CONV_WINK`).
Also fixed a latent bug in the existing T=1 matcher: with MTP (n_rs_seq=1) the M=1 graph has a second snapshot CPY that
reads a *view* of the CONCAT window; the old check only compared `in == window`, so it fused and left that CPY reading
the unmaterialized window (slot-1 snapshot garbage on single-token MTP steps). It now also rejects `in->view_src == window`.

Result (profile, Hello16): verify graph 2024 -> 1934 ops, DSP 58.7 -> 52.0 ms (median); SSM_CONV+WINK 2.03 ms/30 ops
vs CONCAT+CPY+SSM_CONV 8.8 ms. Correctness gate `mtp-gate.ps1` (Hello128, traced): candidate MTP vs base MTP all 128
target rows bit-identical + all 74 draft-candidate rows identical; candidate plain vs candidate MTP 128/128 identical;
acceptance unchanged 52/74. Decision: keep.
Build notes: the preset build's catalog signing fails (skels are in the ExternalProject dir) -> build without cert and sign
in `mtp-pkg.ps1`; our OpenCL DLL crashes in the Adreno compiler at init, so packages reuse the installed ggml-opencl.dll;
the v81 ExternalProject does not track DSP sources, so build-mtp.cmd builds it explicitly.

## 2026-09-29 01:10  B2. Per-uid replay cache, fused catch-up, draft vocab cap

- Replay cache (`GGML_HEXAGON_REPLAY_SLOTS`, default 4): target verify, draft and catch-up graphs now all replay
  (log shows `graph replay hit (slot cache)`). Also no longer arms a capture when the op batch is non-empty.
  Gate (Hello128 traced, vs WINK-only build): MTP target 128/128 and draft 74/74 bit-identical.
- Fused catch-up (`LLAMA_MTP_FUSED_CATCHUP=1`, common/speculative.cpp + server seq_rm keep): the first draft row
  (h[pos0-1], id_last)@pos0 is exactly the catch-up row the next verify would compute, so its MTP KV is kept and the
  verify catch-up is deferred; only accepted draft tokens' rows are decoded, prepended to the next draft batch.
  One ctx_dft decode per step instead of two. Gate (Hello128): target 128/128 AND draft candidate rows 74/74
  bit-identical to base MTP, acceptance unchanged 52/74. Draft+catch-up DSP time 5.3+1.55 -> one 2.4 ms graph (with dv).
- Draft vocab cap (`LLAMA_MTP_DRAFT_VOCAB=N`): the MTP draft head scores only token ids [0, N) (view of output.weight;
  BPE ids are ~frequency ordered; Hello128 output has 5/128 tokens >= 65536, all emoji). Draft-only by construction.
  N=65536: head 2.95 -> ~0.8 ms, backend TOP_K over 64k instead of 248k.
  Gates vs base MTP, target logits: Hello128 128/128 identical (acceptance 54/72 vs 52/74); sky prose128 128/128
  identical (acceptance 59/68 both).
- Single untraced runs (Hello128): fc 27.73 tok/s (with MTP profile syncs), fc+dv65536 30.01 tok/s (plain ~28.1).

## 2026-09-29 01:20  B3. MoE tail fusions for T=2..4 token rows

DSP: ADD_NW, ADD_SG, MOE_WNORM loop over T token rows (T=1 path unchanged). Host: new `try_fuse_moe_addnw_t`
(MUL + 7 strided-view ADDs -> ADD+NW), shexp gate and router-norm matchers accept T rows, HVX MUL_MAT+ADD allowed for
2..4 activation rows when src2 has the destination shape (fusion bits 1<<14 MM_ADD_SMALLM, 1<<15 MOE_TAIL_T).
Profile (Hello16, fc+dv65536): verify graph 1934 -> 1414 ops, DSP 52.0 -> 49.0 ms (plain M=1: 33.65 ms);
draft graph (incl. folded catch-up) 2.44 ms. Remaining M=2 excess over M=1: experts +5.8 ms (two tokens' experts,
already deduplicated per expert in the id kernel), dense HVX matmuls +2.6, GDN K=2 +2.3, FA +0.6, small ops ~+1.5.

### 01:45  Tail-fusion exactness bug found and fixed; first ABBA win

First tail build failed the gate: target row 0 (the prompt's final 2-token ubatch; MTP splits the 13-token prompt
11+2 because of n_keep_tail) differed from base. Bisect via `GGML_HEXAGON_TAIL_T_MASK`: ADD+SG-T and MOE+WNORM-T exact,
ADD+NW-T not. Cause: the per-token weight DMA started at `w->data + t*32` (off the 128-byte grid) and did not return
that token's weights. Fix: read all T weight rows from the single 128-byte DMA of `w->data`. After the fix, all fusions
on (incl. small-M MUL_MAT+ADD) + fc + dv65536: Hello128 target logits 128/128 bit-identical to base MTP.
Lesson: HTP DMA sources in these kernels must stay 128-byte aligned.

Note on plain-vs-MTP comparisons: a plain run of the same build differed from base MTP at the prompt row (first bit
row 0), i.e. the known intermittent *plain-path* instability; MTP-vs-MTP target gates remained bit-exact throughout.

Untraced ABBA (256 tokens, Hello, rounds: inst/base/mtp/mtp/base/inst):
| variant | tok/s runs | mean |
| --- | --- | ---: |
| installed pkg-wos plain | 26.89 26.73 | 26.81 |
| base worktree build plain | 26.90 26.88 | 26.89 |
| MTP dmax1, tail3 build, fc + dv65536 | 30.01 30.03 | 30.02 (+11.7%) |
Acceptance 106/149 (0.71). Draft-max 2 / 3 on this build: 27.94 / 21.13 tok/s (Hello128), worse than dmax1.
Wall breakdown per MTP step (Hello128, MTP profile): step 57.2 ms = verify llama_decode 52.2 (DSP ~49) +
draft 3.3 (DSP 2.44) + ~1.7 sampling/server; host+sync ~5.8 ms/step.

### 02:20  Gates on two more prompts; policy sweep on the tail3 build

Full config (tail3 build, fc + dv65536, dmax 1) vs base-build MTP, complete target logit rows (traced 128 tokens):
sky prose 128/128 bit-identical (acceptance 59/68 both), hash-table prose 128/128 bit-identical (54/72 base,
55/71 candidate). With Hello (01:45) that is 3 prompts x 128 target rows exact. Reference traces kept as
`mtp-speed/REF-base-mtp1-{hello,sky,hash}128.logits.bin`.

Single-run policy sweep, Hello 256 tokens, tail3 + fc + dv65536 (plain on this prompt ~26.9):
| dmax | p-min | tok/s | acceptance |
| ---: | ---: | ---: | --- |
| 1 | 0 | ~30.0 (ABBA) | 106/149 |
| 1 | 0.3 | 30.47 | 106/142 |
| 1 | 0.5 | 29.73 | 101/126 |
| 1 | 0.7 | 25.62 | 72/136 (!) |
| 2 | 0.5 | 29.21 | 126/170 |
| 2 | 0.7 | 30.97 | 114/138 |
The dmax1/p0.7 outlier is a dv interaction, not a bug: with p0.7 and no dv, acceptance is 89/104-106 (fc on or off).
Truncating the draft vocabulary removes emoji tokens (ids >= 65536, frequent in this chat reply) from the draft
softmax, which inflates the in-vocab top-1 probability exactly where the draft is wrong, so a p-min gate lets those
bad drafts through. Use p-min only with dv off or a larger dv.
Verify cost now scales ~linearly: M=1 33.65, M=2 49.0, M=3 64.0 ms DSP (+15 ms per extra token: experts ~6.5,
dense HVX matmuls ~2.6, GDN ~2.3, rest small ops).

## 2026-09-29 03:00  C. Host/transport overhead study (affects plain and MTP alike)

New env-gated diagnostics (worktree): `LLAMA_DECODE_TIMING=1` (host phase timing of llama_context::decode),
`GGML_HEXAGON_TIME_BATCH=1` (DSP batch usec from the response vs host-observed submit->response),
`GGML_HEXAGON_TIME_GET=1` (host memcpy of outputs).
- Host work per decode is small: batch prep/graph reuse/set_inputs ~0.1 ms; logits copy (1-2 MB) ~0.03 ms.
  The draft graph is rebuilt on every other call (fused catch-up alternates 1- and 2-row draft batches): ~1.3 ms
  enqueue for those calls.
- The big effect is on the DSP side: plain decode batch (1384 ops) takes 35.4 ms of DSP time when the host thread
  blocks in dspqueue_read, but 33.35 ms when the host spins (`GGML_HEXAGON_OPPOLL=1`); host-observed wake latency is
  only ~1.0 ms (block) vs 0.15 ms (spin). So an idle host CPU slows the NPU itself by ~6% (likely cluster idle
  states / coherent-memory snoop or fabric clocks; the DSP already votes max bus/core corners, DCVS off).
  A predicted-duration hybrid (`OPPOLL=2`: high-res sleep then spin the last 2 ms) recovers the wake latency but not
  the DSP slowdown (27.72 vs spin 29.54 tok/s plain), confirming the NPU needs the host core awake for the whole batch.
- Untraced single runs, Hello256: plain block 27.14 / spin 29.52 (+8.8%); MTP block 30.07 / spin 32.73 (+8.8%).
  The relative MTP gain is unchanged (+10.8% vs +10.9%); spinning costs one busy host core (power not yet measured).

### 03:25  Constant-shape draft batches: no gain; plain-path instability hypothesis (not tested)

`LLAMA_MTP_PAD_ROWS=1` (recompute the previous kept row after a rejection so draft batches always have 2 rows) cut
draft-graph rebuilds from 38/64 to 0/64, but steady-state per-call cost was already only ~0.1 ms for a rebuild
(build 0.07 + enqueue 0.04); outputs+wait unchanged (~3.0 ms). Left off by default. Run-to-run variation of the verify
wait (49.7 vs 51.9 ms in consecutive runs) is ~5%, which is why only ABBA numbers are reported as results.

Hypothesis for the intermittent plain-path numerical failure (from reading htp-tensor.c, not yet tested): the DSP
dirty-range tracker only covers lines *written* through the cache. A line that is merely cached *clean* (e.g. a source
read by HVX) is never invalidated before a later op DMA-writes a new tensor to the same (allocator-reused) address.
If a subsequent op then does a cached store into a *different* tensor sharing that 128-byte line, the store merges into
the stale clean line, and its eventual write-back overwrites the DMA-written bytes. The corruption depends on when the
line is evicted, i.e. it is intermittent, it needs sub-line / small tensors, and it is cured by clean+invalidate of
small tensors before each op but not by clean alone - all consistent with the decode campaign's actsrc-small results.
A targeted test: invalidate (not just clean) the head/tail partial lines of every DMA destination before the op, or
make the allocator 128-byte align every activation tensor/view that is written by DMA.

## D. Policy ABBA (fixed env leak) and a correctness failure with draft-max 2

Harness bug found and fixed: `mtp-abba.ps1` / `mtp-gate.ps1` invoke run-mtp-speed.ps1 in-process, so an ExtraEnv
setting leaked into later runs (abba2h was invalid: dv stayed on). run-mtp-speed now clears every campaign knob first.
The earlier gates are unaffected (their ref arms always ran before cand arms); abba1 is unaffected (plain arms).

Policy ABBA (256 tokens, 2 rounds each, pmin pkg = tail3 code + LLAMA_MTP_PMIN_NEXT):
| prompt | d1 + fc + dv65536 | d1 + fc, full vocab | d2 + fc, p-min 0.7 |
| --- | --- | --- | --- |
| Hello | 30.89 (31.87, 29.90) acc 106/149 | 28.48 acc 107/148 | 25.69 acc 112/146 |
| sky prose | 32.41 (32.42, 32.40) acc 117/138 | 31.37 acc 120/134 | 22.30 (28.19, **16.40**) |
Draft vocab cap is worth +3..8% at equal acceptance. Draft-max 2 with p-min 0.7 is worse.

**Correctness failure:** one of the two d2/p0.7 sky runs produced garbage ("//////..." after "**Explanation:**",
acceptance 58/356). Reproduced 1/3 with the pmin build (fc on, 4 replay slots). Controls on the same prompt/policy:
base build 4/4 clean (acceptance 150/160); pmin build with fc OFF 4/4 clean (150/160); fc ON + new fusions off
3/3 clean (132/160); fc ON + `GGML_HEXAGON_REPLAY_SLOTS=1` 5/5 clean (132/160). So the failure needs fc + the
multi-slot replay cache + draft-max 2 (varying draft/verify graph shapes); dmax1 runs (>20 so far, all gates) never
showed it. Note also that fc changes draft-max-2 drafts (132/160 vs 150/160 accepted), i.e. fc is only draft-exact
for draft-max 1 (checked by the 74/74 draft-row gate); for dmax>1 it needs a separate audit. Suspected mechanism for
the *target* garbage: a corrupt draft token/embedding fed into the verify batch can put NaN into K/V of a later row,
and 0 * NaN in masked attention poisons earlier rows even though the draft is then rejected. Mitigation now: replay
slots limited / fc restricted to draft-max 1 (see next entry).

### Correction + mitigation (replay slots)

Correction to the 02:20 sweep note: the dmax1/p-min 0.7 outlier (72/136 accepted, 25.62 tok/s) was NOT a draft-vocab
calibration effect. A scan of every run output found three garbage transcripts ("/////..."): that run
(t-pol-d1-p0.7-hello256, fc + dv + 4 replay slots) and the two dmax2/p0.7 sky runs. With the same config but
`GGML_HEXAGON_REPLAY_SLOTS=1` it is clean 3/3 with 89/106 accepted (0.84, same as the no-dv run).
Common factor of all failures: fused catch-up + multi-slot replay + varying verify/draft graph shapes (p-min or dmax2).
Root cause not identified (read of the replay code found no stale-uid path; uids are never reused after a rebuild).
ABBA rs4 vs rs1 (dmax1, fc + dv, Hello256): 29.97 vs 29.70 tok/s (<1%), so the default is now 1 slot
(`opt_replay_slots = 1`, env override kept for research), and fused catch-up auto-disables for draft-max > 1
(its drafts differ from the unfused path there: 132/160 vs 150/160). All reported gates/ABBA for the recommended
config (dmax1, no p-min) never showed garbage (>20 runs incl. 3-prompt target gates).

## Merge with main line (npu-htp-decode 7773910) - coordinator request

mtp-speedup merge commit `9831408` (llama.cpp-hexagon worktree branch `mtp-speedup`; npu-htp-decode not moved).
Conflicts in ggml-hexagon.cpp / ssm-conv.c / ssm-conv.h resolved by keeping both sides:
- kernel params: `fused_window` (2 = prefill, 3 = MTP), then main's `silu`, then MTP `n_ns`, `ns_off[3]`.
- conv windows disjoint by T: MTP K-snapshot window T <= 3 (host matcher and DSP check), prefill window T >= 4.
- ADD_NW: MTP T-row path (T <= 4, kernel_params[2] = 0 -> existing worker loops over src->ne[2]) is tried before
  the prefill expert-combine matcher (kernel_params[2] = T > 1 -> prefill worker); binary-ops.c auto-merged.
- Found and fixed in the prefill conv matcher the same latent bug as in the old decode matcher: in MTP mode
  (n_rs_seq > 0) a prompt ubatch has a second snapshot CPY that reads a *view* of the CONCAT window; the matcher only
  checked `in == window`, so it would fuse and leave that CPY reading the unmaterialized window. It now also rejects
  `in->view_src == window` (such windows fall to the MTP fusion only when T <= 3, else stay unfused).
Gates on the merged build are running (`mtp-merge-gates.ps1`: 3-prompt MTP target logits vs installed main line,
plain vs installed, prefill PPL, fusion edge check).

### Merged-build gates (pkg-mtp-merged: host E90183F8, skel EDB46343)

- MTP target logits, merged build (fc + dv65536, dmax1) vs installed main line (pkg-wos host 1B80C6A9, skel 5D253AAF,
  plain MTP): Hello 128/128, sky 128/128, hash 128/128 full-vocabulary rows bit-identical.
- Prefill PPL (run-prefill-ppl.ps1, 4x512, merged bin+lib): 12.1307 +/- 1.11278 (exact).
- Plain Hello128, merged vs installed: first bit difference at the prompt row, transcript split at token 85.
  Plain-only changes in the merge are none by construction (all MTP fusions need T <= 4 rows, the prompt is 13
  tokens, M=1 decode uses the unchanged matchers, replay defaults to the original single slot); this pattern
  (row-0 difference) is the known intermittent plain-path instability, but without repeat pairs it is not proven
  here.
- Fusion edge check (run-fusion-edge-check.ps1, merged bin+lib, OPFUSION=1 vs 8062, prompts of 6/40/130/260/520/900 words, ub 512): all 6 greedy transcripts identical.
