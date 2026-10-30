# A/B 1: baseline topk-c vs port packages (dr, pad, all) and OPPOLL variants of 'all'; 2 interleaved rounds (order reversed in round 2).
$p = 'npu_campaign\pkg-clover-port-'
$arms = @(
    @{ n = 'topkc'; pkg = 'npu_campaign\pkg-clover-topk-c'; env = @{} },
    @{ n = 'dr';    pkg = "${p}dr";  env = @{} },
    @{ n = 'pad';   pkg = "${p}pad"; env = @{} },
    @{ n = 'all';   pkg = "${p}all"; env = @{} },
    @{ n = 'allpoll'; pkg = "${p}all"; env = @{ GGML_HEXAGON_OPPOLL = '1' } },
    @{ n = 'allpoll05'; pkg = "${p}all"; env = @{ GGML_HEXAGON_OPPOLL = '1'; GGML_HEXAGON_OPPOLL_SPIN_MS = '0.5' } }
)
& "$PSScriptRoot\abab.ps1" -Tag ab1 -Arms $arms -Rounds 2
