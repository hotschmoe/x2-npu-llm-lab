# Final gates + A/B on pkg-clover-port-final (clover merged with rollback + all port features). Every step takes the NPU lock in its own process.
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $repo
$pre = 'npu_campaign\pkg-clover-port-'
'== KLD (new default reference)'
& "$repo\npu_campaign\clover-kld.ps1" -Label port-final -Pkg "${pre}final"
'== GDN cache fusion (K>1, exercises SNAP kernel)'
& "$PSScriptRoot\tbo.ps1" -Label final-gdncf -Pkg "${pre}final" -Mode test -Op GATED_DELTA_NET_CACHE_FUSION
& "$PSScriptRoot\tbo.ps1" -Label final-gdn -Pkg "${pre}final" -Mode test -Op GATED_DELTA_NET
# A: legacy-compatible default (n_max 15, env defaults) vs m2
$M2 = @{ n = 'm2'; pkg = 'npu_campaign\pkg-clover-m2'; env = @{} }
$F  = @{ n = 'final'; pkg = "${pre}final"; env = @{} }
& "$PSScriptRoot\blk2.ps1" -Tag fin-a -Seq @($M2, $F, $F, $M2)
# B: product config of the rollback agent (bonus, n_max 5): rs6 vs final
$RS = @{ n = 'rs6'; pkg = 'npu_campaign\pkg-clover-rollback-rs6'; env = @{ LLAMA_SPEC_BONUS = '1' }; extra = '--spec-draft-n-max 5' }
$FB = @{ n = 'final'; pkg = "${pre}final"; env = @{ LLAMA_SPEC_BONUS = '1' }; extra = '--spec-draft-n-max 5' }
& "$PSScriptRoot\blk2.ps1" -Tag fin-b -Seq @($RS, $FB, $FB, $RS)
# C: attribution inside final (rollback product config): snap off/on, OPPOLL, graph cache + hoststat
$S1 = @{ n = 'snap1'; pkg = "${pre}final"; env = @{ LLAMA_SPEC_BONUS = '1'; GGML_HEXAGON_HOSTSTAT = '1' }; extra = '--spec-draft-n-max 5' }
$S0 = @{ n = 'snap0'; pkg = "${pre}final"; env = @{ LLAMA_SPEC_BONUS = '1'; GGML_HEXAGON_HOSTSTAT = '1'; GGML_HEXAGON_GDN_SNAPDMA = '0' }; extra = '--spec-draft-n-max 5' }
$PO = @{ n = 'poll'; pkg = "${pre}final"; env = @{ LLAMA_SPEC_BONUS = '1'; GGML_HEXAGON_HOSTSTAT = '1'; GGML_HEXAGON_OPPOLL = '1' }; extra = '--spec-draft-n-max 5' }
$GC = @{ n = 'gc4096'; pkg = "${pre}final"; env = @{ LLAMA_SPEC_BONUS = '1'; GGML_HEXAGON_HOSTSTAT = '1'; GGML_HEXAGON_GRAPH_CACHE = '4096' }; extra = '--spec-draft-n-max 5' }
& "$PSScriptRoot\blk2.ps1" -Tag fin-c -Seq @($S1, $S0, $PO, $GC)
'== DONE'
