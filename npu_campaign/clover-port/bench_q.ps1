$A = @{ n = 'topkc'; pkg = 'npu_campaign\pkg-clover-topk-c'; env = @{} }
$B = @{ n = 'all'; pkg = 'npu_campaign\pkg-clover-port-all'; env = @{} }
& "$PSScriptRoot\bench.ps1" -Tag b1 -Seq @($A, $B, $B, $A)
