Set-Location 'C:\Users\hotschmoe\documents\llm_inference\npu_campaign'
$bin = 'llama.cpp-hexagon-prefill2\pkg-prefill2\bin'
$G = 'models\gguf\gemma4-26b-a4b-qat-q4_0\gemma-4-26B_q4_0-it.gguf'
powershell -NoProfile -File ab-multi.ps1 -Specs "gbase|npu_campaign\prefill2-cand-base|-|PF_ROUTER=1","grep|npu_campaign\prefill2-cand-rep|-|PF_ROUTER=1","gfin|npu_campaign\prefill2-cand-dmag2|$bin|PF_ROUTER=1" -Rounds 2 -Reps 3 -PP 1024 -UB 1024 -Tag h2 -Model $G > ab-h2.out 2>&1
powershell -NoProfile -File ab-multi.ps1 -Specs "t4|npu_campaign\prefill2-cand-dmag2|$bin|GGML_HEXAGON_ID_HVX_MAX_T=4","t64|npu_campaign\prefill2-cand-dmag2|$bin|GGML_HEXAGON_ID_HVX_MAX_T=64","t512|npu_campaign\prefill2-cand-dmag2|$bin|GGML_HEXAGON_ID_HVX_MAX_T=512" -Rounds 1 -Reps 2 -PP 16,64,256 -UB 512 -Tag h3 > ab-h3.out 2>&1
