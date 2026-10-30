# Blocks of 4 runs under one lock hold each. A = topk-c baseline, B = port 'all'; then attribution (dr/pad) and OPPOLL blocks.
$p = 'npu_campaign\pkg-clover-port-'
$A = @{ n = 'topkc'; pkg = 'npu_campaign\pkg-clover-topk-c'; env = @{} }
$B = @{ n = 'all'; pkg = "${p}all"; env = @{} }
$D = @{ n = 'dr'; pkg = "${p}dr"; env = @{} }
$P = @{ n = 'pad'; pkg = "${p}pad"; env = @{} }
$O1 = @{ n = 'allpoll'; pkg = "${p}all"; env = @{ GGML_HEXAGON_OPPOLL = '1' } }
$O2 = @{ n = 'allpoll05'; pkg = "${p}all"; env = @{ GGML_HEXAGON_OPPOLL = '1'; GGML_HEXAGON_OPPOLL_SPIN_MS = '0.5' } }
& "$PSScriptRoot\blk.ps1" -Tag ab2a -Seq @($A, $B, $B, $A)
& "$PSScriptRoot\blk.ps1" -Tag ab2b -Seq @($B, $A, $A, $B)
& "$PSScriptRoot\blk.ps1" -Tag ab2c -Seq @($D, $P, $P, $D)
& "$PSScriptRoot\blk.ps1" -Tag ab2d -Seq @($B, $O1, $O2, $B)
& "$PSScriptRoot\blk.ps1" -Tag ab2e -Seq @($O2, $B, $O1, $O2)
