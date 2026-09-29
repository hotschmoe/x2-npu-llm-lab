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
