# OPPOLL variants of the 'all' package (host cores + speed). Blocks of 4 under one lock each.
$pre = 'npu_campaign\pkg-clover-port-'
$B  = @{ n = 'all'; pkg = "${pre}all"; env = @{} }
$O1 = @{ n = 'allpoll'; pkg = "${pre}all"; env = @{ GGML_HEXAGON_OPPOLL = '1' } }
$O2 = @{ n = 'allpoll05'; pkg = "${pre}all"; env = @{ GGML_HEXAGON_OPPOLL = '1'; GGML_HEXAGON_OPPOLL_SPIN_MS = '0.5' } }
& "$PSScriptRoot\blk.ps1" -Tag ab3a -Seq @($O1, $B, $O2, $B)
& "$PSScriptRoot\blk.ps1" -Tag ab3b -Seq @($B, $O2, $O1, $B)
