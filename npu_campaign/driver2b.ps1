Set-Location 'C:\Users\hotschmoe\documents\llm_inference\npu_campaign'
$bin = 'llama.cpp-hexagon-prefill2\pkg-prefill2\bin'
$G = 'models\gguf\gemma4-26b-a4b-qat-q4_0\gemma-4-26B_q4_0-it.gguf'
powershell -NoProfile -File run-fusion-edge-check2.ps1 -LibDir npu_campaign\prefill2-cand-dmag2 -BinDir $bin -OffMask 63486 -WordsCsv 40,260,900 -Tag p2edge > edge-dmag2.out 2>&1
powershell -NoProfile -File run-lib-compare3.ps1 -Model $G -LibA npu_campaign/prefill2-cand-base -MaskA 63230 -LibB npu_campaign/prefill2-cand-dmag2 -MaskB 1 -BinDir $bin -WordsCsv 33,1025 -Tag p2gem-final > gem-final-cmp.out 2>&1
$env:PF_ROUTER = '0'
powershell -NoProfile -File run-prefill2-ppl.ps1 -Label p2-dmag2-c16-hmx -LibDir npu_campaign/prefill2-cand-dmag2 -BinDir $bin -Chunks 16 > ppl-dmag2-c16-hmx.out 2>&1
