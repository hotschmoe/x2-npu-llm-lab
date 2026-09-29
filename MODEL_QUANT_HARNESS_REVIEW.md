# MoE model fit, quant formats and harness review (2026-09-29)

Agent-produced (Opus), model shapes from HF config.json; throughput projections are first-order models (+/-30%),
calibrated on our measured Qwen3.6 profile - use for ranking only. Nothing here was run on the NPU.

## Key finding: expert prefill is now weight-streaming (DMA) bound, not dequant bound

At ub512/1024 nearly every expert is hit every ubatch, so each ubatch streams all routed weights once:
256 x 3 x 2048 x 512 x 40 layers x 0.5625 B = ~18.1 GB. Measured expert time (435 ms per 1024-token pass, pipe9) is
~42-49 GB/s = ~87% of the DMA floor (377 ms at 48 GB/s). Floors per 1024-token ubatch: DMA 377 ms, HVX dequant ~193 ms,
HMX ~116 ms. Consequences:
- per-token expert cost ~ routed bytes / ub: larger ub (or layer-major multi-ubatch) and fewer bytes/weight are the levers;
  faster dequant (pipe8 LUT: no gain) and W4A8 are not, at ub >= 512.
- an fp16 tile-packed copy would be ~3.6x slower (2 B/weight over the same DMA).
- the measured ub2048 regression (578 vs 614 t/s) contradicts the model (expert cost should halve) -> profile ub2048 first.

## Model comparison (derived from configs)

| | Qwen3.6-35B-A3B | Gemma 4 26B-A4B | Nemotron 3 Nano 30B-A3B | Qwen3-30B-A3B | GPT-OSS-20B | ERNIE-4.5-21B-A3B |
| --- | --- | --- | --- | --- | --- | --- |
| layers / hidden | 40 / 2048 | 30 / 2816 | 52 blocks (23 Mamba2, 23 MoE, 6 attn) / 2688 | 48 / 2048 | 24 / 2880 | 28 / 2560 |
| experts / top-k / width | 256 / 8 / 512 | 128 / 8 / 704 | 128 / 6 / 1856 (non-gated ReLU^2) | 128 / 8 / 768 | 32 / 4 / 2880 | 64 / 6 / 1536 |
| mixer | 30 GDN + 10 attn | 25 sliding + 5 global | Mamba2 + attn | full attn | sliding+full, sinks | full attn |
| rows/expert @ ub512/1024/2048 | 16/32/64 | 32/64/128 | 24/48/96 | 32/64/128 | 64/128/256 | 48/96/192 |
| HMX tile use @ ub512 | 50% | 100% | 75% | 100% | 100% | 75% |
| routed weights streamed / ubatch | 32.2 B | 22.8 B | 29.4 B | 29.0 B | 19.1 B | 20.4 B |
| Q4 size | 20.8 GB | 14.4 GB (local QAT Q4_0) | 18.6 GB | 17.6 GB | 12.1 GB MXFP4 | 12.8 GB |
| unfriendly | GDN chain, F32 router, 248k head | d512 FA, softcap, 262k head | no SSM_SCAN op on Hexagon (CPU fallback x23) | 96 KiB/tok KV | attention sinks | none |
| proj. pp @ ub1024 (router on HMX) | ~730 (650 today) | ~890 | ~760 (only with new SSM_SCAN kernel) | ~970 | ~1300 | ~1250 |
| proj. decode tok/s | 28 measured (MTP available) | ~21 | ~26 | ~27 | ~19.5 | ~26 |

Ranking - prefill: GPT-OSS > ERNIE > Gemma 4 > Qwen3-30B > Qwen3.6 > Nemotron. Decode: Qwen3.6 (MTP) ~ Qwen3-30B > ERNIE >
Nemotron > GPT-OSS > Gemma. Quality: not evaluated. Recommendation: keep Qwen3.6 primary; benchmark Gemma 4 QAT next (zero new
ops, already local, Q4_0 is its trained format); use GPT-OSS-20B MXFP4 as format/shape control; skip Nemotron until an
SSM_SCAN kernel exists. Nemotron 3 Nano is real (arch nemotron_h_moe, llama.cpp issue #18064).

## Quant formats (bytes/weight in DSP packed tile)

Q4_0 0.5625 (current); Q4_1/Q4_K 0.625; IQ4_NL 0.5625 (LUT dequant, better quality at equal bytes, already on the repack
list); MXFP4 0.531 (only real bytes lever: -5.6% traffic; native for GPT-OSS); NVFP4 no Hexagon support; W4A8 on integer HMX
needs group-128/per-channel + calibration (block-32 scales cannot fold into the int accumulator), layout undecoded, high risk;
AutoRound int4 exports as GGUF Q4_0 (calibration only, no kernel change). Prefill: keep 4-bit; zero-kernel accuracy upgrades =
IQ4_NL or AutoRound->Q4_0; MXFP4 experts if KLD acceptable. Decode: same resident tiles, one layout; try a Q4_0/IQ4_NL head
instead of Q6_K (~-0.19 GB/token, ~+10% decode). Dual resident layouts not justified.

## Harness

Stay in llama.cpp for model/serving work (host gaps ~0.5 ms of 1.4 s; wins are DSP kernels). Genuinely missing capability:
wavefront overlap between ubatches (experts DMA/HMX-bound with idle HVX while GDN/elementwise are HVX-bound with idle DMA;
upper bound sum -> max ~+40%, speculative and unmeasured). Phases: 0 in-llama.cpp work (profile ub2048/4096, HMX router + exact
HVX top-k recheck, CONCAT+SSM_CONV+SILU fusion, prefill ADD_N, Q4 head); 1 model/format A/Bs with llama-bench + PPL/KLD (Gemma 4
QAT, GPT-OSS MXFP4, Qwen IQ4_NL, Qwen MXFP4_MOE); 2 Zig host replaying one frozen layer to test expert/GDN co-running on split
thread pools (go/no-go); 3 two-ubatch pipeline if phase 2 shows >=25% overlap; 4 optional W4A8 HMX. ZML/vLLM/SGLang backends:
compiler/porting projects with no evidence of payoff.
