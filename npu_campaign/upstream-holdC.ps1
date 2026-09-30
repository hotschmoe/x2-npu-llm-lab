# upstream sync A/B hold (run via lockrun3.py): prefill pp1024 Qwen + Gemma, decode plain + MTP, u1 vs merged2-m1, 3 rounds interleaved
$ErrorActionPreference = 'Continue'
$R = 'ref|npu_campaign/pkg-merged2-m1/lib|npu_campaign/pkg-merged2-m1/bin|'
$U = 'up|npu_campaign/pkg-upstream-u1/lib|npu_campaign/pkg-upstream-u1/bin|'
& "$PSScriptRoot\ab-multi.ps1" -Specs $R,$U -Rounds 3 -PP 1024 -UB 1024 -Reps 3 -Tag abup-qwen
$gem = 'models\gguf\gemma4-26b-a4b-qat-q4_0\gemma-4-26B_q4_0-it.gguf'
& "$PSScriptRoot\ab-multi.ps1" -Specs $R,$U -Rounds 3 -PP 1024 -UB 1024 -Reps 3 -Tag abup-gemma -Model $gem
$env2 = 'LLAMA_MTP_FUSED_CATCHUP=1,LLAMA_MTP_DRAFT_VOCAB=65536'
& "$PSScriptRoot\mtp-abba.ps1" -Prefix abup-dec -Variants "refplain=npu_campaign/pkg-merged2-m1|none|1|;upplain=npu_campaign/pkg-upstream-u1|none|1|;refmtp=npu_campaign/pkg-merged2-m1|mtp|1|$env2;upmtp=npu_campaign/pkg-upstream-u1|mtp|1|$env2" -Rounds 3 -Tokens 256
