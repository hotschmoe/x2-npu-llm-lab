# Greedy-completion equality between two DSP skel candidates (each with its own OPFUSION mask), for any model.
# Example (Gemma): run-lib-compare.ps1 -Model 'models\gguf\...gguf' -LibA npu_campaign\prefill-cand-base -MaskA 7934 -LibB npu_campaign\prefill-cand-nchunk1 -MaskB 1
param(
    [string]$Model = 'models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0.gguf',
    [Parameter(Mandatory = $true)][string]$LibA,
    [int]$MaskA = 7934,
    [Parameter(Mandatory = $true)][string]$LibB,
    [int]$MaskB = 1,
    [string]$BinDir = '',
    [string]$WordsCsv = '5,33,129,257,511,1025',
    [int]$Ub = 512,
    [int]$NGen = 24,
    [string]$Tag = 'libcmp',
    [int]$RouterB = 1,
    [int]$RouterA = 1
)
$ErrorActionPreference = 'Stop'
$Words = @($WordsCsv -split ',' | ForEach-Object { [int]$_ })
$repoRoot = Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot 'npu-lock-fast.ps1')
$src = Get-Content "$repoRoot\PREFILL_CAMPAIGN.md" -Raw
$corpus = ($src -split '\s+' | Where-Object { $_ -match '^[A-Za-z]{3,}$' })
$bin = if ($BinDir) { (Resolve-Path (Join-Path $repoRoot $BinDir)).Path } else { "$repoRoot\llama.cpp-hexagon\pkg-wos\bin" }
$exe = "$bin\llama-completion.exe"
$results = @()
foreach ($w in $Words) {
    $prompt = ($corpus | Select-Object -First $w) -join ' '
    $pf = "$PSScriptRoot\$Tag-prompt-$w.txt"
    Set-Content -LiteralPath $pf -Value $prompt -NoNewline -Encoding ascii
    $texts = @{}
    foreach ($v in @(@{ n = 'A'; lib = $LibA; mask = $MaskA }, @{ n = 'B'; lib = $LibB; mask = $MaskB })) {
        $env:GGML_HEXAGON_OPFUSION = "$($v.mask)"; $env:GGML_HEXAGON_OPBATCH = '3072'; $env:GGML_HEXAGON_REPLAY = '1'
        $env:GGML_HEXAGON_ROUTER_HVX = "$(if ($v.n -eq 'A') { $RouterA } else { $RouterB })"; $env:GGML_HEXAGON_MM_SELECT = '2'; $env:GGML_HEXAGON_FA_SELECT = '2'; $env:GGML_HEXAGON_GDN_SELECT = '2'
        $env:ADSP_LIBRARY_PATH = "$((Resolve-Path (Join-Path $repoRoot $v.lib)).Path);"
        $out = "$PSScriptRoot\$Tag-$w-$($v.n)"
        $a = "-m `"$repoRoot\$Model`" -f `"$pf`" -n $NGen -c 2048 -b $Ub -ub $Ub --seed 1 --temp 0 --device HTP0 -ngl 99 -fit off -no-cnv --ignore-eos -lv 3"
        Enter-NpuLock "libcmp-$w-$($v.n)"
        try { $run = Start-Process -FilePath $exe -ArgumentList $a -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.log.err" } finally { Exit-NpuLock }
        $texts[$v.n] = Get-Content "$out.log" -Raw
        "words=$w $($v.n) exit=$($run.ExitCode) chars=$($texts[$v.n].Length)"
    }
    $results += [pscustomobject]@{ Words = $w; Identical = ($texts['A'] -eq $texts['B']) }
}
$results | Format-Table -AutoSize
