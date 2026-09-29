# Gates for the merged mtp-speedup build (see MTP_SPEEDUP_LOG.md): MTP target logits vs installed main line,
# plain vs installed, prefill PPL, fusion edge check. Each run takes the NPU lock (via the called scripts).
param([string]$Pkg = 'npu_campaign\pkg-mtp-merged', [string]$Tag = 'mg')
$ErrorActionPreference = 'Continue'
$c = $PSScriptRoot
$prompts = @{ hello = 'Hello'; sky = 'Explain why the sky appears blue in three clear sentences.'; hash = 'Explain how a hash table handles collisions, with a concrete example.' }
foreach ($k in 'hello', 'sky', 'hash') {
    powershell -NoProfile -File "$c\mtp-gate.ps1" -Prefix "$Tag-$k" -Prompt $prompts[$k] -RefPkg 'llama.cpp-hexagon\pkg-wos' -CandPkg $Pkg `
        -CandExtraEnv 'LLAMA_MTP_FUSED_CATCHUP=1;LLAMA_MTP_DRAFT_VOCAB=65536' -Arms 'ref-mtp,cand-mtp' 2>&1 | Select-String 'GATE ref-mtp vs cand-mtp :|accept'
}
powershell -NoProfile -File "$c\mtp-gate.ps1" -Prefix "$Tag-plain-hello" -RefPkg 'llama.cpp-hexagon\pkg-wos' -CandPkg $Pkg -Arms 'ref-plain,cand-plain' 2>&1 | Select-String 'GATE|eval time'
powershell -NoProfile -File "$c\run-prefill-ppl.ps1" -Label "$Tag-merged" -LibDir "$Pkg\lib" -BinDir "$Pkg\bin" 2>&1
powershell -NoProfile -File "$c\run-fusion-edge-check.ps1" -LibDir "$Pkg\lib" -BinDir "$Pkg\bin" -Tag "$Tag-edge" 2>&1 | Select-Object -Last 12
