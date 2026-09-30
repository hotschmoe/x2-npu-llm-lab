# upstream sync gates, one NPU hold (run via lockrun3.py): edge check, GET_ROWS/ARGSORT/TOP_K oracles, Nemotron Nano gates
$ErrorActionPreference = 'Continue'
$repo = Split-Path $PSScriptRoot -Parent
foreach ($op in 'GET_ROWS','ARGSORT','TOP_K','CONCAT','SET_ROWS','UNARY','GLU') {
  & "$PSScriptRoot\mamba-run.ps1" -Pkg pkg-upstream-u1 -Exe test-backend-ops -Log "up-tbo-$op" -ArgLine "test -b HTP0 -o $op" | Out-Null
  "TBO $op : " + ((Get-Content "$PSScriptRoot\up-tbo-$op.log" | ForEach-Object { $_ -replace '\x1b\[[0-9;]*m','' } | Select-String 'tests passed|Backend HTP0.*(OK|FAIL)' | ForEach-Object { $_.Line.Trim() }) -join ' ')
}
& "$PSScriptRoot\mamba-gates.ps1" -Pkg pkg-upstream-u1 -Tag u1
