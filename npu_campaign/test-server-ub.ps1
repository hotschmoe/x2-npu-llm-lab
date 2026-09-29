# Start llama-server with a given -b/-ub, send one long-prompt completion, print the server's own timings, stop it. Takes the NPU lock.
param([int]$B = 2048, [int]$Ub = 1024, [int]$Words = 1200, [int]$NPredict = 16, [int]$Port = 8090, [string]$Pkg = 'llama.cpp-hexagon\pkg-wos')
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'npu-lock.ps1')
$root = Split-Path $PSScriptRoot -Parent
Enter-NpuLock "server-ub$Ub"
$proc = $null
try {
    $env:ADSP_LIBRARY_PATH = "$root\$Pkg\lib;"
    $env:GGML_HEXAGON_OPBATCH = '3072'
    foreach ($n in 'GGML_HEXAGON_ROUTER_HVX','GGML_HEXAGON_MM_SELECT','GGML_HEXAGON_FA_SELECT','GGML_HEXAGON_GDN_SELECT','GGML_HEXAGON_OPFUSION','GGML_HEXAGON_REPLAY') { Remove-Item "Env:$n" -ErrorAction SilentlyContinue }
    $args = "-m `"$root\models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0.gguf`" --host 127.0.0.1 --port $Port --device HTP0 -ngl 99 -fit off --load-mode none -c 8192 -b $B -ub $Ub -t 4 -np 1 --cache-prompt --metrics --jinja --reasoning auto"
    $proc = Start-Process -FilePath "$root\$Pkg\bin\llama-server.exe" -ArgumentList $args -WindowStyle Hidden -PassThru -RedirectStandardOutput "$PSScriptRoot\server-ub$Ub.log" -RedirectStandardError "$PSScriptRoot\server-ub$Ub.log.err"
    $ready = $false
    for ($i = 0; $i -lt 180; $i++) {
        Start-Sleep -Seconds 2
        try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3; if ($h.status -eq 'ok') { $ready = $true; break } } catch {}
        if ($proc.HasExited) { throw "server exited early: $($proc.ExitCode)" }
    }
    if (-not $ready) { throw 'server not ready' }
    $src = Get-Content "$root\PREFILL_CAMPAIGN.md" -Raw
    $corpus = ($src -split '\s+' | Where-Object { $_ -match '^[A-Za-z]{3,}$' })
    $prompt = ($corpus | Select-Object -First $Words) -join ' '
    foreach ($round in 1, 2) {
        $body = @{ prompt = "$prompt (round $round)"; n_predict = $NPredict; temperature = 0; cache_prompt = $false } | ConvertTo-Json
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 600
        "b=$B ub=$Ub round=$round prompt_n=$($r.timings.prompt_n) prompt_ms=$([math]::Round($r.timings.prompt_ms)) prompt_tps=$([math]::Round($r.timings.prompt_per_second,1)) gen_tps=$([math]::Round($r.timings.predicted_per_second,1))"
    }
} finally {
    if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    Get-Process llama-server -ErrorAction SilentlyContinue | Where-Object { $_.Id -eq $proc.Id } | Stop-Process -Force -ErrorAction SilentlyContinue
    Exit-NpuLock
}
