# Gemma MoE Hexagon fit review

User-requested agent review, 2026-09-27; no model download, inference, build, or model switch. Main Qwen goal unchanged. Candidate: Gemma4 26B-A4B, not dense Gemma3 or Gemma4 dense variants.

Primary sources:
- https://ai.google.dev/gemma/docs/core/model_card_4
- https://huggingface.co/google/gemma-4-26B-A4B/raw/main/config.json
- https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/tree/main
- https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json

## Shape facts

30 layers, hidden2816,128 experts/top8, routed width704, shared width2112,16 query heads.25 sliding layers: head256,8KVheads,GQA2,window1024.5global layers: head512,2KVheads,GQA8. Vocabulary262144, tied embeddings, GELU, logit softcap30. No GDN/conv recurrent blocks. Global K projection also feeds V, but subsequent norms/RoPE/cache semantics differ; do not assume identical cached K/V. Google official QAT Q4_0 text GGUF about14.4GB plus optional1.19GB vision projector; pin revision and inspect tensor-type histogram before qualification.

GGML matrix shapes [input,output,expert]: routed gate/up[2816,704,128], down[704,2816,128]; shared gate/up[2816,2112], down[2112,2816]; router[2816,128]. Local Q[2816,4096], K/V[2816,2048], O[4096,2816]. Global Q[2816,8192], K[2816,1024], O[8192,2816]. Output head[2816,262144]. Local gemma4.cpp and conversion/gemma.py support exists, but loader support is not fullHTP placement proof.

## Expected fit, not benchmark evidence

- All key matrix dimensions divide32/64.704/2112 do not divide128 elements, but F32 row byte sizes DO divide128 bytes. Audit packed strides and worker slices instead of rejecting by element count alone.
- HMX matmul requires token M>4 with current eligibility; top8 experts does not turn one-token GEMV into HMX GEMM. Decode remains mainly HVX.
- Attention256/512 dimensions meet basic guards, subject to VTCM solver. Local G2 is not covered by current Qwen G8 batch/reuse attention tests.
- No GDN reduces recurrent-state/rollback complexity, but introduces25 local attention caches. Separate F16K/V conventional local-window storage about200MiB; five global layers about20KiB per context position, before allocator padding/sharing. Long-context advantage is not automatic.
- Additional router scales/norms, expert output scales and shared/routed output norms create fusion opportunities and more operations. Must qualify actual backend placement.

Derived active FFN matrix coefficients per token (not measured DRAM traffic):

Gemma:30*3*2816*(8*704+2112)=1,962,639,360.
Qwen:40*3*2048*(8*512+512)=1,132,462,080.

Gemma is1.73x. Ideal Q4_0 coefficient bytes roughly1.104GB versus.637GB, excluding packing/cache/mixed types. Output head738.2M coefficients versus508.6M,1.45x; tied embeddings save residency, not output GEMV work. Thus fewer resident experts/fewer layers does not establish faster single-stream decode.

## Later qualification backlog

Official frozen QAT Q4_0 text-only model; tensor histogram/fullHTP placement first. Focused GEMV widths704/2112/2816, G2/D256 and G8/D512 attention, expert scaling. Repeated bitidentity and CPU-reference selected tokens before paired decode at128/1K/4K contexts with stage profiles. Gemma assistant/MTP is a separate later experiment, not reuse of Qwen's existing draft head. No model download or migration authorized by this review.
