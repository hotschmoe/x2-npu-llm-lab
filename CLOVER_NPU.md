# Clover on the Hexagon NPU: first diffusion LM on X2

Started 2026-10-29. Goal: run [Diffuspace/Clover-35B-A3B-Preview](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview)
(block-diffusion MoE, Qwen3.6-35B-A3B derivative) with the whole graph on HTP0, correct first, then fast.
Same method as the Qwen3.6 MoE campaign (AGENT_COORDINATION.md rules apply unchanged: one NPU process at a time
via `npu_campaign/npu-lock.ps1`, interleaved A/B only, per-agent worktree + append-only log, spoofs off by default).
**Addition for this campaign: CPU-only model runs and `test-backend-ops perf` also hold the lock** (unified LPDDR:
a CPU model run steals the NPU's DRAM bandwidth; Clover HTP runs still execute fallback ops on the CPU). Builds use
`--parallel 4`. Timings from runs that overlapped another model process are invalid.

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

### 2026-10-30 00:10 - calibration bench (CONTAMINATED: agent builds + a concurrent CPU-only Clover run)

`llama-bench -ngl 99 -dev HTP0 -fa 1 -ctk f16 -ctv f16 -r 2`, pkg-clover-base:

| model | pp16 | pp256 | tg32 |
| --- | ---: | ---: | ---: |
| Qwen3.6-35B-A3B Q4_0 (same upstream build) | 76.9 +- 8.4 | 360.0 +- 8.7 | 27.05 +- 0.34 |
| Clover-35B-A3B Q4_0 | **2.30** +- 0.01 | 50.4 +- 8.4 | 6.54 +- 2.00 |

Qwen3.6 is the target ceiling for Clover's causal path once the fallbacks are gone (same graph modulo bf16 rounds
and the Clover router). Clover pp16 = 2.3 tok/s is pathological (draft-self's 16-token verify batches ran ~46 tok/s
in the same build), suspect: `n_tokens != n_outputs` "prefill" branch of the graph. Re-measure on a quiet machine.
Rule added: CPU-only model runs take the NPU lock (clover-run.ps1 now always locks).

### 2026-10-30 00:20 - product KPI: host CPU while generating

The user's goal for the deployed product is NPU-only inference: the laptop stays fully usable (no UI lag, CPU and GPU
free). CPU/GPU may be used freely for testing. `clover-run.ps1` now samples the process CPU time (250 ms) and prints
`HOST_CPU busy cores during generation`; every A/B reports it next to tok/s. Host-side costs to drive to ~0:
CPU fallback ops, host spin-waits on HTP completion, full-vocab argmax/entropy over 248k x 16 rows per step,
recurrent-state checkpoint memcpy. `clover-kld.ps1` is the locked KLD gate helper.
Incident: an unlocked HTP0 perplexity run overlapped locked runs 00:09-00:15 (one run measured 0.57 tok/s);
timings in that window are void.

### 2026-10-30 ~04:10 - M1: router/coreset top-k on HTP (topk agent, merged into `clover` d587232)

`ggml_top_k_stable` (GGML_OP_TOP_K with op_params[0]=1: value desc, ties -> lower index; strict total order, so any
correct sort equals the old host std::sort custom op). CPU: partial_sort with that comparator. HTP: composite
(value, index) key in argsort-ops.c, rows <= 1024. Graph sorts the full 256 row + view of k (a k-wide TOP_K output
crashed the DSP in-model; root cause not found - open item). Oracles: 140/140 HTP0 and CPU vs explicit
std::stable_sort (76 synthetic incl. heavy ties/masks/+-inf/denormals, 64 real-logit fixtures); negative control
(GGML_HEXAGON_TOPK_SPOIL=1) fails broadly. Audit: 0 CUSTOM; splits draft 1162->922, verify 762->682, decode 582->502.
KLD vs CPU unchanged (0.047923, same-top 89.216%), CPU PPL 15.4413 exactly. Interleaved clean A/B: +5.0..5.5%
(4.89-4.91 -> 5.14-5.16 tok/s), host CPU 1.07-1.16 -> 0.98-0.99 busy cores. Details: npu_campaign/CLOVER_TOPK_LOG.md.
Pushed: hotschmoe/llama.cpp `clover-hexagon`.
Interim (rollback agent): rollback alone 4.75 vs 4.91 legacy, rollback + bonus decode 5.36 (+9%) on HTP0.

### 2026-10-30 ~05:00 - M2: the KLD gap was the CPU reference, not the NPU (trace agent; tools merged 0478cfe)

- Calibration on the same build: Qwen3.6 HTP0 vs CPU KLD 0.0060 / same-top 96.1% -> the 0.048 gap is Clover-specific.
- First divergent tensor: `linear_attn_out-0` (ssm_out, Q4_0, K=4096). Exact f64 recompute from each device's own inputs:
  HTP dense matmuls 5-7e-8 from exact (fp16 HMX), CPU 2e-6..1.3e-4 = ggml-cpu Q8_0 activation quantization (Clover
  activations have per-row absmax/rms up to 63). Routed experts same picture.
- Near-exact reference (311 dense tensors dequantized to F32, experts unchanged, CPU): PPL 15.125.
  vs it: **HTP0 KLD 0.0199 / same-top 93.1%**, CPU Q4_0 0.0458 / 90.0%. HTP is 2.3x closer. Floor ~0.018
  (one HTP build vs itself with an exact router: 0.0177). bf16 rounds are not the driver (removing all: 0.048 -> 0.043).
- New gate (clover-kld.ps1 default): reference `cpu-f32dense-kld-base.bin`, PASS if KLD <= 0.025 and same-top >= 92%.
- Only HTP-side approximation found: fp16 router on HMX (F32 weights), flips 0.6-1.2% of top-8 sets;
  `GGML_HEXAGON_F32_HMX=0` gives exact routing, same 0.0199 KLD, +6.5% prefill time -> default unchanged.
- Diffusion draft path (465 teacher-forced mask rows): HTP vs near-exact KLD 0.036, argmax agreement 89.2%
  (CPU 84.3%); draft token accuracy CPU/HTP/ref 16.6/15.7/15.9%. Draft path is far more sensitive (block coreset
  differs in 24/40 layers between CPU and HTP).
- Note: a full CPU trace (`-Filter '.'`) perturbs CPU results (breaks RMS_NORM+MUL fusion); named filters are safe.
Details: npu_campaign/CLOVER_TRACE_LOG.md. Tools: LLAMA_TENSOR_TRACE_FILE/_FILTER, COMMON_DEBUG_DUMP_DIR/_RE,
clover-draft-dump (CLOVER_MULTI_STRIDE), LLAMA_CLOVER_NO_BF16_ROUND (spoof), make_f32dense_gguf.py.

### 2026-10-30 ~05:30 - pkg-clover-m2 (merged clover 0478cfe: topk + trace tools) gates

- KLD vs F32-dense reference: 0.019871, same-top 93.137%, PPL 15.0832 -> **GATE PASS** (identical to M0: top-k change is exact).
- llama-bench HTP0 (inside the lock, no other llama process; agent builds possibly running): pp16 27.3 +- 5.2,
  pp256 169.1 +- 15.7, tg32 5.87 +- 0.32. The M0 "pp16 = 2.3" was contamination + host top-k, not a graph pathology.
  tg32 ~170 ms/token over ~500 CPU<->HTP splits (~0.34 ms per split round trip) -> the bf16 CPY islands are the lever.
- rollback agent interim (HTP, pre-merge): legacy loop 4.91, rollback+bonus 5.44, rollback+bonus n_max 5 **6.66**;
  plain causal decode 10.1 tok/s still beats speculation while every forward pays the CPU islands.
