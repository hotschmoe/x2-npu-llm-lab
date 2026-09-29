Set-Location 'C:\Users\hotschmoe\documents\llm_inference\npu_campaign'
$bin = 'llama.cpp-hexagon-prefill2\pkg-prefill2\bin'
$G = 'models\gguf\gemma4-26b-a4b-qat-q4_0\gemma-4-26B_q4_0-it.gguf'
# 1. Qwen main A/B: installed-like base vs comb1 (exact set) vs comb1+router refine vs dmag1 (DMA gather); one lock hold per round (4 runs)
powershell -NoProfile -File ab-multi.ps1 -Specs "base|npu_campaign\prefill2-cand-base|-|PF_ROUTER=1","combE|npu_campaign\prefill2-cand-comb1|$bin|PF_ROUTER=1","combR|npu_campaign\prefill2-cand-comb1|$bin|PF_ROUTER=0","dmag1|npu_campaign\prefill2-cand-dmag1|$bin|PF_ROUTER=1" -Rounds 3 -Reps 3 -PP 1024 -UB 1024 -Tag h1 > ab-h1.out 2>&1
# 2. Gemma: base vs REPEAT fast path (rep) vs combined build with expert-scale fold (gemfuse)
powershell -NoProfile -File ab-multi.ps1 -Specs "gbase|npu_campaign\prefill2-cand-base|-|PF_ROUTER=1","grep|npu_campaign\prefill2-cand-rep|-|PF_ROUTER=1","gfuse|npu_campaign\prefill2-cand-gemfuse|$bin|PF_ROUTER=1" -Rounds 3 -Reps 3 -PP 1024 -UB 1024 -Tag h2 -Model $G > ab-h2.out 2>&1
# 3. small-M thresholds for MUL_MAT_ID (HVX no-dequant path) with the comb1 build
powershell -NoProfile -File ab-multi.ps1 -Specs "t4|npu_campaign\prefill2-cand-comb1|$bin|GGML_HEXAGON_ID_HVX_MAX_T=4","t32|npu_campaign\prefill2-cand-comb1|$bin|GGML_HEXAGON_ID_HVX_MAX_T=32","t128|npu_campaign\prefill2-cand-comb1|$bin|GGML_HEXAGON_ID_HVX_MAX_T=128","t512|npu_campaign\prefill2-cand-comb1|$bin|GGML_HEXAGON_ID_HVX_MAX_T=512" -Rounds 2 -Reps 2 -PP 16,64,256 -UB 512 -Tag h3 > ab-h3.out 2>&1
