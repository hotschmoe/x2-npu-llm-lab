$pre = 'npu_campaign\pkg-clover-port-'
$G1 = @{ n = 'gc1'; pkg = "${pre}final"; env = @{ LLAMA_SPEC_BONUS = '1'; GGML_HEXAGON_HOSTSTAT = '1' }; extra = '--spec-draft-n-max 5 -v' }
$G2 = @{ n = 'gc4096'; pkg = "${pre}final"; env = @{ LLAMA_SPEC_BONUS = '1'; GGML_HEXAGON_HOSTSTAT = '1'; GGML_HEXAGON_GRAPH_CACHE = '4096' }; extra = '--spec-draft-n-max 5 -v' }
& "$PSScriptRoot\blk2.ps1" -Tag fin-d -Seq @($G1, $G2) -NPredict 128
