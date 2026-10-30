# CLOVER_TRACE_LOG (trace agent, correctness lead) - append-only

Worktree `llama.cpp-hexagon-clover-trace`, branch `clover-trace` (from `clover` 4838ffb), build dir `build-clover-trace`
(`npu_campaign/build-clover-trace.cmd`), packages `npu_campaign/pkg-clover-trace-<x>` (`pkg-clover-trace.ps1`).
KLD gate: `npu_campaign/trace-kld.ps1` (llama-perplexity 4x512 `ppl_text.txt`, -b/-ub 512, f16 KV, 12 threads, takes the NPU lock).

### 2026-10-30 00:00 - calibration: Qwen3.6-35B-A3B Q4_0 on the same package (pkg-clover-base)

Hypothesis: the Clover KLD gap (0.0479) is generic upstream-HTP numerics. Test: same KLD procedure on Qwen3.6 Q4_0.
New CPU base `clover-logs/qwen36-cpu-kld-base.bin` (CPU PPL 12.1955, matches the known 12.20).

| model | CPU PPL | HTP0 PPL | mean KLD | same top | RMS dp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 Q4_0 | 12.1955 | 12.1438 | 0.00602 | 96.08% | 1.75% |
| Clover Q4_0 (M0) | 15.4413 | 15.0832 | 0.04792 | 89.22% | 4.65% |

Decision: the gap is ~8x the generic HTP noise -> Clover-specific. Clover per-chunk KLD 0.030/0.021/0.037/0.105 (chunk 4
alone ~0.1): a position/content-dependent blow-up, not uniform noise. Note: the Qwen run co-ran with a bf16-shapes run
(both acquired the lock at 23:58:12 - `npu-lock.ps1` New-Item race; numerics unaffected, timing irrelevant here).

### 2026-10-30 00:15 - lock incident + OPFILTER bisection setup

Coordinator flagged an "unlocked" HTP0 perplexity at 00:09:47: that was `trace-clover-f-mmid`, launched through
`trace-kld.ps1` which does Enter-NpuLock (owner.txt held my tag at 00:09:47; another agent's clover-run.ps1 tag
replaced it at 00:09:49 while my run was live). `npu-lock.ps1` acquisition is not atomic against a concurrent waiter
(PS 5.1 New-Item -ItemType Directory = exists-check + CreateDirectory). My scripts now go through `trace-lock.ps1`
(re-read owner.txt 0.7 s after acquiring, back off and retry if it is not ours; release only if still ours). The f-mmid
numbers are numerically valid (timing is not used).

Bisection tool: `GGML_HEXAGON_OPFILTER=<regex>` (ggml_op_desc of nodes the HTP must not claim -> CPU). rep1 (same build,
no env) is bit-identical to M0: PPL 15.083178, KLD 0.047923 -> deterministic, any change is real.

### 2026-10-30 00:20-00:35 - first divergent tensor + frozen-input oracle: the CPU reference is the noisy side

OPFILTER bisection (KLD vs cpu-kld-base): MUL_MAT_ID->CPU 0.0424 (PPL 15.217), dense MUL_MAT->CPU 0.0420 (PPL 15.231).
Neither removes the gap; both move PPL toward the CPU's 15.44.

Tensor trace (new `common_debug_tensor_trace` port + `llama-clover-draft-dump` tool, chunk 4 = 512 tokens, causal,
`trace_nmse_table.py`): layer 0 `attn_norm` NMSE 1e-14, wqkv/z/alpha/beta 2-3e-6, ssm_conv 3e-7, GDN 9e-7,
final_output 1.7e-6, then the FIRST jump: `linear_attn_out-0` (ssm_out, Q4_0, K=4096) 1.4e-4 (80x its input), then
`ffn_moe_out-0` 2.9e-3. Nothing later jumps out of line (attention/FA/GDN/ssm_conv stay at input-level error).

Frozen-input oracle `check_frozen_clover_mm.py` (GGUF weight dequantized, f64 exact on each device's own input):

| layer-0 weight | type | CPU vs exact | CPU vs Q8_0-act emulation | HTP vs exact | HTP vs fp16-HMX emulation | act absmax/rms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| attn_qkv | Q4_0 | 3.0e-6 | 3.6e-10 | 5.5e-8 | 1.2e-11 | 15 / 35 |
| attn_gate | Q4_0 | 3.3e-6 | 3.9e-10 | 6.2e-8 | 6.4e-11 | |
| ssm_alpha | Q4_0 | 3.1e-6 | 3.7e-10 | 5.9e-8 | 1.5e-19 | |
| ssm_beta | Q4_0 | 2.2e-6 | 1.7e-10 | 5.3e-8 | 0 | |
| ssm_out | Q4_0 | **1.3e-4** | 7.9e-9 | 7.4e-8 | 2.3e-11 | **63** |
| ffn_gate_inp (router) | F32 | 7e-15 | - | 5.1e-8 | 2.4e-11 | |

(layer 1: CPU 2-7e-5 vs exact, HTP 6e-8.) So for every Q4_0 dense matmul the HTP (HMX, fp16 activations) is within
~6e-8 of exact and the CPU is 40-2000x worse: ggml-cpu multiplies Q4_0 by activations quantized to Q8_0 (vec_dot_type),
and Clover's activations have massive per-row outliers (absmax/rms up to 63 at the ssm_out input) that Q8_0 blocks
handle badly. The only HTP-side approximation found: F32 router weights are run as fp16 on HMX -> HTP top-8 set differs
from exact for 0.6-1.2% of tokens (CPU is exact there). CPU-vs-HTP top-8 sets differ for 7-10% of tokens, driven by
the input differences.

Second finding: Clover's graph is not a function of the tokens alone. `prefill = n_tokens != n_outputs` switches extra
bf16 rounds (q/k l2-norm, beta) on. llama-perplexity outputs only rows >= n_ctx/2, so it runs the "prefill" graph;
a dump of the same chunk with all rows output runs the other graph. **CPU vs CPU** (same tokens, same rows 256-510):
KLD 0.102, same-top 80% on chunk 4 -> the model amplifies rounding-level perturbations enormously (bf16 round trips
after nearly every op: a difference d << ulp becomes 0-or-1-ulp flips, variance ~d*ulp instead of d^2; plus router
top-8 flips). The CPU-vs-HTP 0.048 therefore cannot be read as HTP error.

Next: near-exact reference = same model with the dense weights dequantized to F32 (`make_f32dense_gguf.py`,
`Clover-35B-A3B-Preview-Q4_0-f32dense.gguf`, 311 tensors, identical values, CPU then uses F32 activations; experts
still Q4_0 x Q8_0). Compare CPU-Q4_0 and HTP against it.

### 2026-10-30 00:30 - near-exact reference: F32-dense CPU PPL 15.1254 (HTP 15.0832, CPU-Q4_0 15.4413)

`cpu-f32dense-kld-base.bin` (CPU, `Clover-35B-A3B-Preview-Q4_0-f32dense.gguf`, dense matmuls in F32, experts still
Q4_0 x Q8_0): **PPL 15.1254**. Same weights as the Q4_0 file; removing the CPU's Q8_0 activation quantization on the
dense matmuls alone moves CPU PPL 15.44 -> 15.13, i.e. most of the way to the HTP's 15.08. The HTP was never worse
than the CPU; the CPU reference is the less exact of the two.

CPU-vs-CPU sensitivity (rounding-only graph change, same tokens/rows, llama-perplexity "prefill" graph vs all-rows
graph): chunk 1 KLD 0.019 (same-top 94%), chunk 4 KLD 0.102 (80%). HTP-vs-CPU per chunk was 0.030/0.021/0.037/0.105:
the same chunk is the sensitive one, at the same magnitude. Tracer check: an untraced HTP dump (ctx 512) is
bit-identical to the traced one (ctx 4096), so the tensor trace does not perturb HTP numerics.
CPU with all bf16 rounds removed (spoof): PPL 15.3957.

### 2026-10-30 00:35 - the CPU reference vs the near-exact reference (offline, kld_rows.py on saved logit files)

Same estimator as llama-perplexity (rows 256-510 of each 512 chunk, 1020 rows):

| base | candidate | mean KLD | same-top | per-chunk KLD |
| --- | --- | ---: | ---: | --- |
| cpu-f32dense (near-exact) | **cpu-kld-base (the gate's reference, CPU Q4_0)** | **0.0458** | **90.0%** | 0.020 / 0.023 / 0.036 / 0.104 |
| cpu-kld-base | cpu no-bf16-round spoof | 0.0399 | 89.8% | 0.019 / 0.017 / 0.020 / 0.103 |
| cpu-kld-base | HTP0 pkg-clover-base (M0, for comparison) | 0.0479 | 89.2% | 0.030 / 0.021 / 0.037 / 0.105 |

The CPU Q4_0 reference is itself as far from the near-exact reference (0.046) as the HTP is from the CPU (0.048), and
a rounding-only change on the CPU alone costs 0.040. KLD ~0.04-0.05 vs `cpu-kld-base.bin` is the noise floor of
Clover on this text, not an HTP error budget. Chunk 4 is the chaotic chunk for every perturbation (~0.10).

### 2026-10-30 00:45 - layer-by-layer vs the near-exact reference (F32-dense CPU trace, chunk 4, all rows)

`f32d-c4.trace` (CPU, F32-dense GGUF; experts still Q4_0 x Q8_0) as reference, `trace_nmse_table.py` + per-token
breakdown (share of the squared error carried by the worst 1% = 5 tokens):

| tensor | CPU-Q4_0 NMSE | HTP NMSE | HTP: share of sq-err in worst 1% tokens |
| --- | ---: | ---: | ---: |
| linear_attn_out-0 (ssm_out) | 1.4e-4 | 1.5e-6 | 0.08 |
| attn_post_norm-0 | 2.0e-4 | 2.0e-6 | 0.04 |
| ffn_moe_out-0 | 2.6e-3 | 1.0e-3 | **0.87** (median token 8.5e-5) |
| l_out-0 | 2.0e-4 | 4.5e-5 | 0.80 |
| l_out-2 / -5 | 3.0e-4 / 1.2e-3 | 1.1e-4 / 6.7e-4 | |
| l_out-10 / -20 / -39 | 1.2e-2 / 8.1e-2 / 4.5e-2 | 1.2e-2 / 8.1e-2 / 3.9e-2 | |

Up to layer 5 the HTP is 2-100x closer to the near-exact reference than the CPU is; from layer ~10 on both saturate
at the same distance (the residual stream diverges chaotically from any perturbation). The HTP's one notable
layer-0 error is the routed MoE output, and 87% of it sits in 5 of 512 tokens = top-8 routing flips, matching the
frozen-oracle finding that the fp16 (HMX-tiled F32) router flips the exact top-8 set for 0.6-1.2% of tokens. The
only F32 weights the HMX tiler touches in Clover are the 40 `ffn_gate_inp` routers [2048 x 256] (ssm_conv1d has
ne0 < 32), so `GGML_HEXAGON_F32_HMX=0` (clover-trace 79fa561) changes exactly the router precision.
Tracer check (CPU): untraced CPU ctx-512 dump bit-identical to the traced ctx-4096 one.

Lock incident 2: at 00:35 every queued run of mine died inside npu-lock.ps1's wait loop (owner.txt briefly
multi-line -> `[int]` cast of an array; and Get-Content IOException while the owner was writing it), escalated by my
scripts' ErrorActionPreference=Stop. `trace-lock.ps1` now catches errors while waiting and retries, and writes
owner.txt itself if its own write did not land. No run was executing at the time (all were waiting); requeued.

### 2026-10-30 00:46 - HTP vs the near-exact reference: KLD 0.0199 (CPU Q4_0 reference: 0.0458)

HTP runs now save their own logits (`trace-kld.ps1 -MakeBase` on HTP0) and are compared offline with `kld_rows.py`
against every reference. Validation: pkg-clover-trace-t1 default on HTP0 = PPL 15.0832 (M0 reproduced bit-for-bit
by the trace package), offline KLD vs cpu-kld-base 0.047913 / same-top 89.22% vs llama-perplexity's 0.047923 /
89.216% (estimator matches to 1e-5).

| candidate (PPL) | ref cpu-kld-base (CPU Q4_0, the gate) | ref cpu-f32dense (near-exact) |
| --- | --- | --- |
| HTP0 default, M0 (15.0832) | 0.0479 / 89.2% / chunks .030 .021 .036 .104 | **0.0199 / 93.1% / chunks .022 .019 .021 .018** |
| CPU Q4_0 = the gate's reference (15.4413) | - | 0.0458 / 90.0% / chunks .020 .023 .036 .104 |

Against the near-exact reference the HTP is 2.3x closer than the CPU reference is, and chunk 4 (0.104 vs the CPU)
is 0.018 for the HTP: the "chaotic" chunk is the CPU's Q8_0-activation error on Clover's outlier activations, not an
HTP fault. Root cause of the gate's 0.048: the reference, not the NPU.

Lock incident 3 (00:43-00:54): two other agents' runs both started at 00:43:10 (`rb-ab2-legacy-diffusion-d15-r1`
and `rb-audit1-rs-diffusion-d15-r1`, i.e. npu-lock.ps1's non-atomic acquisition let both in). ab2 finished at 00:43:38
and its Exit-NpuLock deleted the lock dir unconditionally while audit1 was still running; my queued (redundant)
`t1-htp-default` perplexity then acquired a free lock at 00:43:42 and ran concurrently with audit1 (both ~20 GB RSS,
both crawling). At 00:54 I found it, rewrote owner.txt to audit1's shell (pid 8804) so the lock stays held until audit1
ends, and killed my run (its numbers are not used; M0 reproduction comes from htp-t1-default-mk). **rb-audit1 timings
from 00:43:42-00:54 are contaminated.** Protocol fix to suggest: an OS-atomic lock (exclusive CreateNew file handle
or a named mutex held by the owner process for the whole run; released by the OS on death; release only by the owner).

Lock incident 4 (00:55:43): three other runs started in the same second (`rb-hsw1-rsbonusv5` on HTP0, `rb-ab2-rs` and
`rb-sw2-rs` on CPU), all via clover-run.ps1 -> plain Enter-NpuLock; all three crawl (RSS thrash). With 4+ waiters
polling New-Item every 200 ms, the exists-check/CreateDirectory window makes collisions at every release likely.
My owner.txt re-check only protects against waiters that also re-check, so `trace-lock.ps1` now additionally waits,
while holding the lock, until no other llama-*/test-backend-ops process is alive (and re-takes the dir if a colliding
owner's unconditional Exit-NpuLock deletes it). My queue was idle-waiting at the time; restarted as queue3.

01:17 - coordinator fixed npu-lock.ps1 (ac5e2c8: atomic CreateDirectoryW acquire, owner-checked release).
`trace-lock.ps1` simplified on top of it: catch-and-retry while waiting, owner.txt re-check, then wait (holding the
lock) until no straggler llama-*/test-backend-ops process from the pre-fix window is alive; no more own dir
re-creation. Queue3 restarted (it was waiting, not holding). None of my KLD/trace numbers came from a run that
crashed; overlaps only affect timings, and I use no timings from that window.

### 2026-10-30 01:25 - F32 router on HTP (GGML_HEXAGON_F32_HMX=0, pkg-clover-trace-t2): no accuracy gain; noise floor ~0.02

| candidate | PPL | s/pass | vs cpu-kld-base | vs cpu-f32dense (near-exact) |
| --- | ---: | ---: | --- | --- |
| HTP default (fp16 router on HMX) | 15.0832 | 1.84 | 0.0479 / 89.2% | 0.0199 / 93.1% |
| HTP F32 router (HVX F32 matmul) | 15.0580 | 1.96 | 0.0435 / 89.9% | 0.0199 / 92.3% |

HTP default vs HTP F32-router (one HTP build, only the router precision differs, ~1% of tokens re-routed in layer 0):
KLD 0.0177, same-top 93.9%. So a rounding-level change of a single op costs ~0.018 on this model/text: that is the
model's noise floor, and the HTP's 0.0199 vs the near-exact reference sits on it. The F32 router does not move the
HTP closer to the reference and costs +6.5% prefill time -> leave GGML_HEXAGON_F32_HMX at its default (1); the option
stays in clover-trace 79fa561 as a diagnostic only. The CPU Q4_0 reference (0.046 vs near-exact) is the one clearly
above the floor.

### 2026-10-30 01:50 - task 4, diffusion draft path, prompt A (clover-prompts/diffusion.txt, 13 tokens)

`trace-draft-cmp.ps1 -Tag dA` (pkg-clover-trace-t1): CPU causal prefill -> anchor = CPU argmax 198 -> one non-causal
draft forward (anchor + 15 masks), logits of all 16 rows. HTP e2e = own prefill + CPU anchor; HTP iso = CPU's saved
sequence state loaded (no prefill) + CPU anchor, i.e. only the draft forward differs.

| CPU vs | causal last row | draft row 0 (anchor) | mask rows 1-15: argmax agree | mean KLD | logit rel-RMS |
| --- | --- | --- | ---: | ---: | ---: |
| HTP e2e | KLD 2e-6, rel 1.5e-3 | KLD 0.102 | 14/15 | 0.0356 | 0.105 |
| HTP iso | - | KLD 0.040 | 12/15 | 0.0149 | 0.056 |

HTP e2e vs HTP iso (same device, only the 13-token prefill state differs): mask rows 13/15, KLD 0.040. The
disagreeing rows are near-ties between the repeated draft tokens (7285 / 55411 / 264). The draft forward is far more
sensitive than the causal row (rel-RMS 5-10% vs 0.15%), so it needs its own reference: F32-dense draft runs and full
traces of the isolated draft forward on CPU / HTP / F32-dense are queued (queue4).

02:16 - coordinator: npu-lock.ps1 is now a FIFO ticket lock (7c47cf6). My three waiting children still ran the
pre-FIFO code (direct CreateDirectoryW polling, i.e. outside the ticket queue): stopped while waiting (none held the
lock) and relaunched as one sequential queue (queue5, one run per hold, each < 3 min). `trace-lock.ps1` is now a thin
wrapper: Enter-NpuLock (ticket queue) with retry-on-throw, owner.txt rewrite only if our own write did not land, and
the no-straggler wait before running.

### 2026-10-30 02:30 - task 4: why the draft forward is so sensitive - Clover's block-shared expert coreset

Full traces of the isolated draft forward (CPU state dA-cpu.state loaded, anchor 198 + 15 masks, non-causal):
`dA-cpu-iso-tr` (CPU) vs `dA-htp-iso-tr` (HTP; its logits are bit-identical to the untraced dA-htp-iso run).
Layer 0: attn_norm 2e-14, qkv 2e-6, clover block-attention 5e-6, **linear_attn_out (ssm_out) 2.6e-4** (the same CPU
Q8_0-activation error as in the causal path), attn_post_norm 6.7e-4, router logits 2.6e-5; then routing.
In the draft (non-causal) graph the routed experts are restricted to a *coreset* shared by the whole 16-token block
(`build_clover_coreset`: top-p over the block-summed top-8 expert mass, ~8-19 experts kept). `coreset_cmp.py`:

- coreset differs CPU vs HTP in 24 of 40 layers (by 1-6 experts); top-8 sets differ in 265/640 token-layers (41%)
  (causal path: 7-10% of tokens at layers 0/1); a 1-expert coreset change flips top-8 for most tokens of the block
  (layer 1: 3 experts differ -> 14/16 tokens; layer 23: 6 -> 15/16).
- l_out NMSE grows 2.5e-4 (L0) -> 4e-3 (L15) -> 7e-3 (L24).

The coreset cut is a discontinuous, block-coupled amplifier of rounding-level differences, which is why draft logits
differ by 5-10% rel-RMS while the causal row differs by 0.15%. Even CPU vs CPU (e2e in-memory state vs the same state
saved and reloaded) gives mask-row KLD 0.016 / 12 of 15 argmax: that is the floor this path has. CPU-iso vs HTP-iso:
KLD 0.0158, 13/15 argmax - at the floor.

### 2026-10-30 02:56 - draft-path floor measured on the CPU alone; tracer caveat

- CPU run-to-run: `dA-cpu-rep` is bit-identical to `dA-cpu` (logits and saved state) -> the CPU is deterministic.
- An eval callback that asks for no tensor (`-Filter '^NOMATCH$'`): bit-identical -> the callback itself is harmless.
- Full trace (`-Filter '.'`) on the CPU is NOT bit-faithful: asking for every node splits the graph after `norm-N`, so
  ggml-cpu's RMS_NORM+MUL fusion (ggml_cpu_try_fuse_ops) no longer applies. The traced e2e run's prefill row differs
  from the untraced one by rel-RMS 1.7e-3 and its **draft mask rows by KLD 0.063, argmax 9/15** - a pure ulp-level
  CPU code-path change moves the draft more than CPU -> HTP does (e2e KLD 0.036, 14/15). Named-tensor filters that do
  not ask for the RMS_NORM node (all my causal traces) were verified bit-identical; HTP full traces are bit-identical.
  Consequence: per-op comparisons of `-iso-tr` traces are valid as CPU(unfused)-vs-HTP evidence, but the final
  draft numbers must come from untraced runs.
- Traced e2e vs traced iso (same CPU code path): layer-0 state and every layer-0 tensor of the draft forward identical
  (state save/load is exact); differences start at the layer-1 recurrent state written by the (fused vs unfused)
  prefill.

Near-exact reference for the draft forward (`dA-f32d-iso-tr`, F32-dense CPU, same loaded state): HTP is ~100x closer
in layers 0-2 (l_out-0 NMSE 2.7e-6 vs CPU 2.5e-4; linear_attn_out-0 1.1e-6 vs 2.6e-4), then both saturate (l_out-20
5e-3 both); coreset differs from the reference in 21/40 (CPU) and 24/40 (HTP) layers, top-8 sets in 238/640
token-layers for both. Logits vs reference on 15 mask rows: CPU(unfused) KLD 0.0085 / 14 of 15, HTP 0.0138 / 12 of 15
(15 rows; the CPU also shares the reference's expert path, Q4_0 x Q8_0). 15 rows are too few to rank devices, so I
added a teacher-forced multi-draft mode to the dump tool (clover-trace 87a56f2, CLOVER_MULTI_STRIDE): 31 draft
forwards per process over chunk1+chunk4 (stride 32, anchor = true token), queued for CPU, HTP, F32-dense CPU and HTP
with F32 router.

### 2026-10-30 03:20 - task 4 numbers (untraced runs)

Prompt B (`chunk1-300.tok`, 300 tokens, CPU anchor 220): CPU e2e vs HTP e2e: causal last row KLD 0.029; draft
mask rows KLD 0.028, argmax 13/15. Isolated (both load the same CPU state): mask rows KLD 0.0039, argmax 13/15.
CPU e2e vs CPU state-reloaded (untraced): bit-identical -> state save/load is exact; iso isolates the draft forward.

Teacher-forced multi-draft (`llama-clover-draft-dump` CLOVER_MULTI_STRIDE=32, pkg-clover-trace-t3, chunk1+chunk4 =
1024 tokens, 31 draft forwards, anchor = true token, 465 mask rows; `draft_multi_cmp.py`):

| | mask-row argmax vs truth | anchor-row next-token acc | mean correct mask prefix |
| --- | ---: | ---: | ---: |
| CPU (md-cpu) | 16.6% | 67.7% | 1.10 |
| HTP0 (md-htp) | 15.7% | 67.7% | 1.13 |

CPU vs HTP: mask-row KLD mean 0.053 (median 0.023), argmax agreement 392/465 = 84.3%, anchor row KLD 0.035 /
29 of 31; per-block mask KLD 0.014-0.44 (a few blocks with coreset flips dominate the mean).
(Measured convention: mask row i, at position p+i, predicts the token at p+i; row 0 predicts p+1.)

Multi-draft against the near-exact reference (md-f32d, F32-dense CPU, same teacher-forced blocks) and an HTP floor:

| candidate | vs md-f32d: mask KLD mean / median | argmax agree | accuracy vs truth | correct mask prefix |
| --- | --- | ---: | ---: | ---: |
| md-f32d (reference) | - | - | 15.9% | 1.23 |
| CPU Q4_0 | 0.0388 / 0.0188 | 84.3% | 16.6% | 1.10 |
| HTP0 | **0.0356 / 0.0176** | **89.2%** | 15.7% | 1.13 |
| HTP0, F32 router | 0.0330 / 0.0173 | 88.6% | 16.6% | 1.26 |

HTP vs HTP-with-F32-router (one router precision change): mask KLD 0.0169, 91.2% argmax agreement = the floor of
this path. CPU vs HTP 0.053 / 84.3% is roughly floor(CPU) + floor(HTP); the HTP is at least as close to the reference
as the CPU, and draft accuracy vs the true tokens is the same within noise (465 rows, SE ~1.7 points).
Single prompts, untraced, vs the F32-dense CPU draft with the same anchor (mask rows): prompt A CPU KLD 0.043 / 12 of
15, HTP 0.017 / 13 of 15; prompt B CPU 0.018 / 13 of 15, HTP 0.019 / 15 of 15.

### 2026-10-30 04:00 - diagnostic spoof (not a result): all bf16 rounds removed on both devices

`LLAMA_CLOVER_NO_BF16_ROUND=1` (clover-trace 693f87b, off by default) on HTP0 (pkg-clover-trace-t1): PPL 15.1540
(CPU with the same spoof 15.3957). HTP-noround vs CPU-noround: KLD 0.0434, same-top 89.2%, chunks .024 .018 .039
.092 (with rounds: 0.0479 / 89.2%). Removing Clover's bf16 round trips barely closes the CPU-HTP gap and chunk 4 stays
at ~0.09, so the gap is driven by the CPU's activation quantization (still present), not by bf16 rounding of HTP-vs-CPU
differences. Spoof stays a diagnostic only.

### 2026-10-30 04:30 - frozen-input oracle for the routed experts (layer 0, chunk 4, 512 tokens)

`check_frozen_clover_moe.py` on `cpu-c4-moe0` / `htp-c4-moe0` (named filter, fusion-safe): each device's expert
matmuls recomputed in f64 from its own inputs (attn_post_norm, top-8 ids, swiglu) with the GGUF Q4_0 experts:

| | gate vs exact | up vs exact | down vs exact | matches |
| --- | ---: | ---: | ---: | --- |
| CPU | 1.1e-5 | 7.2e-5 | 5.5e-5 | Q8_0-activation emulation to 1e-14 / 1e-10 |
| HTP0 (HMX MUL_MAT_ID) | 5.0e-8 | 8.1e-8 | 1.1e-7 | fp16-activation level |

swiglu absmax/rms median 9.8, max 22.5. CPU vs HTP top-8 sets equal for 93.0% of tokens. Same picture as the dense
matmuls: the HTP experts are 200-700x closer to exact than the CPU reference's.

Cleanup: all traces and candidate logit files deleted (~22 GB). Kept: `cpu-f32dense-kld-base.bin` (proposed gate
reference), `qwen36-cpu-kld-base.bin` (calibration), small dA/dB logit files, and
`models/gguf/clover-35b-a3b-q4_0/Clover-35B-A3B-Preview-Q4_0-f32dense.gguf` (24.4 GB; regenerate with
`make_f32dense_gguf.py` if deleted).
