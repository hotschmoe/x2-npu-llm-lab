# Clover on the Hexagon NPU: first diffusion LM on X2

Started 2026-10-29. Goal: run [Diffuspace/Clover-35B-A3B-Preview](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview)
(block-diffusion MoE, Qwen3.6-35B-A3B derivative) with the whole graph on HTP0, correct first, then fast.
Same method as the Qwen3.6 MoE campaign (AGENT_COORDINATION.md rules apply unchanged: one NPU process at a time
via `npu_campaign/npu-lock.ps1`, interleaved A/B only, per-agent worktree + append-only log, spoofs off by default).

## Inputs

| item | value |
| --- | --- |
| model | official `Clover-35B-A3B-Preview-Q4_0.gguf`, 19.59 GB, `models/gguf/clover-35b-a3b-q4_0/` (SHA256SUMS alongside) |
| code | llama.cpp PR #31306 (`Maphist0:dev-clover`, 4838ffb) on upstream master f4d8af6 (2026-10-28) |
| worktree | `llama.cpp-hexagon-clover`, branch `clover` (nested repo `llama.cpp-hexagon`) |
| build / package | `npu_campaign/build-clover.cmd`, `npu_campaign/pkg-clover.ps1 -Name <x>` -> `npu_campaign/pkg-clover-<x>` |
| driver | `llama-speculative-simple --spec-type draft-self` (masked-block diffusion draft + causal verify) |

Why Q4_0: official source, and every quantized tensor is a Hexagon-native type (Q4_0/Q4_K/IQ4_NL/MXFP4 list).
The Clover-only tensors (`ssm_conv1d.weight_right`, `ssm_bd_*`) are F32 in every published GGUF.

## What the PR adds (no ggml backend changes)

Two graphs per layer stack:
- **causal / verify path** (`causal_attn = true`): the Qwen3.6 graph (GDN via `build_recurrent_attn`, full attention,
  MoE) plus many `build_bf16_round` (F32->BF16->F32 CPY pairs) and a router top-k done by `ggml_custom_4d`
  (`clover_sort_rows`, CPU-only by construction, stable: lower expert id wins ties).
- **diffusion / draft path** (`causal_attn = false`, `build_clover`): per GDN layer, left `ssm_conv` over history +
  right `ssm_conv` over future tokens, L2 norms, BF16 `mul_mat`s for state recall and in-block QK scores,
  `soft_max` over [scores | history bias], BF16 `mul_mat` for local mix, decay; router "coreset" (softmax, top-k,
  get/set_rows, cumsum, step) masks experts to a shared top-p set for the whole block.

Hexagon op inventory (htp-kparams.cpp) covers every op type the graph uses except `GGML_OP_CUSTOM`. Whether each
*instance* is accepted (types, shapes, strides) is decided by the per-op checks and must be measured.

## Phases

1. **Baseline (CPU)**: CPU-only run of the draft-self example -> golden token stream, acceptance stats, tok/s.
   CPU perplexity on `npu_campaign/ppl_text.txt` (causal path) as the quality reference.
2. **Placement audit**: `GGML_SCHED_DEBUG=2` on HTP0 -> list of nodes that fall back to CPU, splits per graph,
   for both the causal (verify) graph and the diffusion (draft) graph.
3. **Kill the fallbacks**, biggest first. Expected: router top-k custom op (40+ per graph), anything the BF16 round
   trip or BF16 mul_mat shapes trip over. Each new/changed op: per-op oracle CPU vs HTP0 on frozen inputs,
   NMSE <= 1e-7, plus a negative control that must fail.
4. **End-to-end correctness**: greedy token stream on HTP0 equal to CPU for N prompts (or, where the NPU is
   intentionally approximate, judged by KLD/PPL against CPU, never against an NPU baseline);
   first-divergent-tensor tracing when it differs.
5. **Latency**: per-op profile (`GGML_HEXAGON_PROFILE=1`), decode tok/s, prefill tok/s, draft acceptance,
   interleaved A/B on every change. Fork perf work (npu-htp-decode / qwen4: tile layout, expert pipeline,
   GDN snapshot DMA) is a later port onto this branch.

## Log

Append below: timestamp, hypothesis, change, result, decision.

### 2026-10-29 23:30-00:05 - M0: Clover runs end to end on HTP0 (stock upstream Hexagon + PR #31306)

Package `pkg-clover-base` (skel 9B394BC35EBC3C5A). Prompt `clover-prompts/diffusion.txt`, greedy, 256 tokens,
draft-self n_max 15, `-b 256 -ub 256`, AC power. KV cache **f16** (bf16 KV aborts: HTP SET_ROWS has no bf16 dst).

| run | decode tok/s | notes |
| --- | ---: | --- |
| CPU 12t, plain greedy (llama-completion) | **45.1** | 22.2 ms/token |
| CPU 12t, draft-self, entropy accept 0.9/0.6, bf16 KV | 25.7 | README settings |
| CPU 12t, draft-self, exact greedy verify, bf16 KV | 23.7 | 41 draft calls, mean acc len 6.10 |
| CPU 12t, draft-self, exact, f16 KV | 19.5 | |
| **HTP0, draft-self, exact, f16 KV** | **5.1** | first diffusion LM on the X2 NPU; coherent output |
| HTP0 plain causal decode step (from same run) | ~10.3 | 96.7 ms/token vs Qwen3.6 ~24-28 tok/s on stock upstream |

Draft-self per-position acceptance (CPU exact): 0.98 0.95 0.69 0.57 0.50 0.38 0.29 0.21 0.14 0.12 0.10 0.07 0.05 0.05.
CPU draft forward (16 tokens, non-causal) ~105 ms.

**The example's `accept = 100%` is an accounting artifact**: with a recurrent (checkpointed) context a partial
acceptance restores the checkpoint and re-submits accepted+correction as a new draft, which is then fully accepted
and is the only round counted. Real numbers come from the `draft-self: #calls ... #mean acc len` statistics line
(`-v`). Cost per step = draft forward + verify forward + **re-verify forward**. The re-verify is pure overhead of
the rollback mechanism -> per-position recurrent-state snapshots (the fork's GDN snapshot idea) remove it.

**Correctness (causal path)**: `llama-perplexity` 4x512 on `ppl_text.txt`, KLD vs CPU logits:
CPU PPL 15.4413; HTP0 PPL 15.0832, **mean KLD 0.0479, same top token 89.2%, RMS dp 4.65%**. That is about the size
of a whole 4-bit quantization step (na0x2c6 Q4_K_M vs BF16 KLD 0.058). Not acceptable as "the same model":
root cause to be found by first-divergent-tensor tracing. Greedy text differs from token 0 (the first thinking
token is a near-tie style choice: CPU bf16-KV, CPU f16-KV and HTP each pick a different opening).

**Placement audit** (`GGML_SCHED_DEBUG=2 -v`, `npu_campaign/clover_sched_audit.py`):

| graph | nodes | splits (CPU/HTP) | ops on CPU |
| --- | ---: | ---: | --- |
| diffusion draft (16 tok, non-causal) | 4251 | 1162 (581/581) | CPY 462, CUSTOM 120, GET_ROWS 2 |
| causal verify/prefill | 3155 | 762 (381/381) | CPY 566, CUSTOM 40, GET_ROWS 2 |
| causal decode | 2975 | 582 (291/291) | CPY 446, CUSTOM 40, GET_ROWS 2 |

1. CPY on CPU = `build_bf16_round` (F32->BF16->F32 pairs) and copies of their consumers: HTP CPY has no BF16.
2. CUSTOM = `build_clover_top_k` (stable host sort) - 40 router top-k per graph, +80 in the coreset (draft only).
3. Everything else (ssm_conv, soft_max, cumsum, concat, pad, fill, set_rows, BF16 mul_mat, GDN, FA, MoE) is on HTP.

Decision: kernel work = (a) HTP BF16 CPY + fused round, (b) HTP-resident stable top-k; in parallel, (c) trace the
KLD gap, (d) remove the re-verify forward.
