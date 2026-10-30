# trace agent: diffusion (non-causal draft) path CPU vs HTP0 for one draft forward (anchor + 15 masks).
#   1) CPU: causal prefill of the prompt, save seq state, anchor = CPU argmax, draft forward  -> <Tag>-cpu
#   2) HTP0 end-to-end: own prefill, anchor forced to the CPU anchor, draft forward           -> <Tag>-htp-e2e
#   3) HTP0 isolated: load the CPU state (no prefill), CPU anchor, draft forward               -> <Tag>-htp-iso
# Each run goes through trace-dump.ps1 (verified NPU lock). Compare with draft_cmp.py.
param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [string]$PromptFile = '',
    [string]$Tokens = '',
    [string]$Pkg = 'npu_campaign\pkg-clover-trace-t1',
    [hashtable]$HtpEnv = @{}
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
$src = if ($PromptFile) { "-PromptFile $PromptFile" } else { "-Tokens $Tokens" }
$state = "npu_campaign\clover-logs\trace\$Tag-cpu.state"
$envLit = '@{' + (($HtpEnv.GetEnumerator() | ForEach-Object { "$($_.Key)='$($_.Value)'" }) -join ';') + '}'
powershell -NoProfile -Command "& .\npu_campaign\trace-dump.ps1 -Label $Tag-cpu -Device none -Pkg $Pkg $src -DraftN 16 -StateSave $state"
$hdr = [IO.File]::ReadAllBytes("$PSScriptRoot\clover-logs\trace\$Tag-cpu.logits")
$anchor = [BitConverter]::ToInt32($hdr, 16)
"CPU anchor $anchor"
powershell -NoProfile -Command "& .\npu_campaign\trace-dump.ps1 -Label $Tag-htp-e2e -Device HTP0 -Pkg $Pkg $src -DraftN 16 -Anchor $anchor -Env $envLit"
powershell -NoProfile -Command "& .\npu_campaign\trace-dump.ps1 -Label $Tag-htp-iso -Device HTP0 -Pkg $Pkg $src -DraftN 16 -Anchor $anchor -StateLoad $state -Env $envLit"
python "$PSScriptRoot\draft_cmp.py" "$PSScriptRoot\clover-logs\trace\$Tag-cpu.logits" "$PSScriptRoot\clover-logs\trace\$Tag-htp-e2e.logits" --rows
python "$PSScriptRoot\draft_cmp.py" "$PSScriptRoot\clover-logs\trace\$Tag-cpu.logits" "$PSScriptRoot\clover-logs\trace\$Tag-htp-iso.logits" --rows
