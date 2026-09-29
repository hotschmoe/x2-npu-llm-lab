# Greedy-completion equality check across prompt lengths that hit partial-block/tail paths of the prefill kernels.
# Compares OPFUSION=1 (all fusions) against OPFUSION=<mask without the given bit> for each prompt size.
param(
    [string]$LibDir = 'npu_campaign\prefill-cand-convfuse',
    [string]$BinDir = 'llama.cpp-hexagon\pkg-prefill\bin',
    [int]$OffMask = 8062,            # 8190 - 128 (SSM_CONV_WIN off)
    [string]$WordsCsv = '6,40,130,260,520,900',
    [int]$Ub = 512,
    [string]$Tag = 'edge'
)
$ErrorActionPreference = 'Stop'
$Words = @($WordsCsv -split ',' | ForEach-Object { [int]$_ })
$repoRoot = Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot 'npu-lock-fast.ps1')
$src = Get-Content "$repoRoot\PREFILL_CAMPAIGN.md" -Raw
$corpus = ($src -split '\s+' | Where-Object { $_ -match '^[A-Za-z]{3,}$' })
$model = "$repoRoot\models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0.gguf"
$exe = (Resolve-Path (Join-Path $repoRoot $BinDir)).Path + '\llama-completion.exe'
$lib = (Resolve-Path (Join-Path $repoRoot $LibDir)).Path
$results = @()
foreach ($w in $Words) {
    $prompt = ($corpus | Select-Object -First $w) -join ' '
    $pf = "$PSScriptRoot\$Tag-prompt-$w.txt"
    Set-Content -LiteralPath $pf -Value $prompt -NoNewline -Encoding ascii
    $texts = @{}
    foreach ($mask in 1, $OffMask) {
        $env:GGML_HEXAGON_OPFUSION = "$mask"; $env:GGML_HEXAGON_OPBATCH = '3072'; $env:GGML_HEXAGON_REPLAY = '1'
        $env:GGML_HEXAGON_ROUTER_HVX = '1'; $env:GGML_HEXAGON_MM_SELECT = '2'; $env:GGML_HEXAGON_FA_SELECT = '2'; $env:GGML_HEXAGON_GDN_SELECT = '2'
        $env:ADSP_LIBRARY_PATH = "$lib;"
        $out = "$PSScriptRoot\$Tag-$w-m$mask"
        $a = "-m `"$model`" -f `"$pf`" -n 24 -c 2048 -b $Ub -ub $Ub --seed 1 --temp 0 --device HTP0 -ngl 99 -fit off -no-cnv --ignore-eos -lv 3"
        Enter-NpuLock "edge-check-$w-$mask"
        try { $run = Start-Process -FilePath $exe -ArgumentList $a -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.log.err" } finally { Exit-NpuLock }
        $t = Get-Content "$out.log" -Raw
        $texts[$mask] = $t
        "words=$w mask=$mask exit=$($run.ExitCode) chars=$($t.Length)"
    }
    $same = $texts[1] -eq $texts[$OffMask]
    $results += [pscustomobject]@{ Words = $w; Identical = $same }
}
$results | Format-Table -AutoSize
