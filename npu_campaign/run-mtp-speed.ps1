# MTP speed/correctness runner (decode/MTP speedup campaign, see MTP_SPEEDUP_LOG.md).
# Takes the cross-agent NPU lock. Outputs go to npu_campaign\mtp-speed\<Label>.*
# -Pkg: package dir (relative to repo root) whose bin\ holds llama-cli; default installed pkg-wos.
# -LibDir: DSP lib dir for ADSP_LIBRARY_PATH; default <Pkg>\lib.
param(
    [Parameter(Mandatory = $true)][string]$Label,
    [string]$Prompt = 'Hello',
    [ValidateSet('none', 'mtp', 'ngram')][string]$Spec = 'none',
    [ValidateRange(0, 8)][int]$DraftMax = 1,
    [double]$PMin = -1,
    [int]$Tokens = 128,
    [string]$Pkg = 'llama.cpp-hexagon\pkg-wos',
    [string]$LibDir = '',
    [string]$ModelPath = 'models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0-mtp-f32.gguf',
    [switch]$LogitsTrace,
    [switch]$MtpProfile,
    [int]$DeviceProfile = 0,
    [ValidateSet(1, 2)][int]$GdnSelect = 1,
    [ValidateSet(1, 2)][int]$FaSelect = 1,
    [string]$ExtraEnv = '',   # "NAME=VAL;NAME2=VAL2"
    [string]$ExtraArgs = ''
)
$ErrorActionPreference = 'Stop'
if ($Label -notmatch '^[a-zA-Z0-9_.-]+$') { throw "Invalid label [$Label]" }
if ($Prompt -match '["\r\n]') { throw 'Prompt must be a single line without double quotes' }
$repoRoot = Split-Path $PSScriptRoot -Parent
$outDir = Join-Path $PSScriptRoot 'mtp-speed'
New-Item -ItemType Directory -Force $outDir | Out-Null
$base = Join-Path $outDir $Label
foreach ($suffix in @('.log', '.log.err', '.detail.log', '.logits.bin')) {
    if (Test-Path -LiteralPath "$base$suffix") { throw "Existing artifact $base$suffix; use a new label" }
}
$pkgDir = (Resolve-Path (Join-Path $repoRoot $Pkg)).Path
$lib = if ($LibDir) { (Resolve-Path (Join-Path $repoRoot $LibDir)).Path } else { Join-Path $pkgDir 'lib' }
$model = (Resolve-Path (Join-Path $repoRoot $ModelPath)).Path

# frozen production-like decode policy (same as the MTP revisit gates)
$env:GGML_HEXAGON_OPBATCH = '3072'
$env:GGML_HEXAGON_REPLAY = '1'
$env:GGML_HEXAGON_OPFUSION = '1'
$env:GGML_HEXAGON_VERBOSE = '0'
$env:GGML_HEXAGON_PROFILE = "$DeviceProfile"
$env:GGML_HEXAGON_MOE_TOPK_ALIAS = '0'
$env:GGML_HEXAGON_MM_SELECT = '2'
$env:GGML_HEXAGON_HEAD_PREFETCH = '0'
$env:GGML_HEXAGON_FA_SELECT = "$FaSelect"
$env:GGML_HEXAGON_GDN_SELECT = "$GdnSelect"
$env:GGML_HEXAGON_ROUTER_HVX = '1'
$env:LLAMA_MTP_PROFILE = if ($MtpProfile) { '1' } else { '0' }
$env:LLAMA_MTP_KV_ONLY = '0'
# also clear every campaign knob so a caller that dot-sources/&-invokes this script repeatedly in one process
# does not leak ExtraEnv settings into the next run
foreach ($v in @('GGML_HEXAGON_ALLOC_FILL', 'GGML_HEXAGON_NHVX', 'LLAMA_FUSE_MTP_VERIFY', 'LLAMA_TENSOR_TRACE_FILE', 'LLAMA_TENSOR_TRACE_FILTER',
                 'LLAMA_RETAIN_TRACE_FILE', 'LLAMA_RETAIN_TRACE_FILTER', 'LLAMA_RECURRENT_SNAPSHOT_LAYER',
                 'LLAMA_MTP_FUSED_CATCHUP', 'LLAMA_MTP_DRAFT_VOCAB', 'LLAMA_MTP_PAD_ROWS', 'LLAMA_MTP_PMIN_NEXT', 'LLAMA_DECODE_TIMING',
                 'GGML_HEXAGON_OPPOLL', 'GGML_HEXAGON_TAIL_T_MASK', 'GGML_HEXAGON_REPLAY_SLOTS',
                 'GGML_HEXAGON_TIME_BATCH', 'GGML_HEXAGON_TIME_GET')) {
    Remove-Item "Env:$v" -ErrorAction SilentlyContinue
}
$env:LLAMA_LOGITS_TRACE = if ($LogitsTrace) { '1' } else { '0' }
if ($LogitsTrace) { $env:LLAMA_LOGITS_TRACE_FILE = "$base.logits.bin" } else { Remove-Item Env:LLAMA_LOGITS_TRACE_FILE -ErrorAction SilentlyContinue }
if ($ExtraEnv) {
    foreach ($kv in $ExtraEnv.Split(';')) {
        if (-not $kv) { continue }
        $k, $v = $kv.Split('=', 2)
        Set-Item -Path "Env:$k" -Value $v
    }
}
$env:ADSP_LIBRARY_PATH = "$lib;"
$specArgs = switch ($Spec) {
    'none'  { '--spec-type none' }
    'mtp'   { "--spec-type draft-mtp --spec-draft-n-max $DraftMax" }
    'ngram' { "--spec-type ngram-simple --spec-draft-n-max $DraftMax" }
}
if ($PMin -ge 0) { $specArgs += " --spec-draft-p-min $PMin" }
$verbosity = if ($DeviceProfile -gt 0) { 5 } else { 4 }
$arguments = "-m `"$model`" -n $Tokens -c 1024 -b 128 -ub 128 -p `"$Prompt`" --seed 1234 --temp 0 --device HTP0 -ngl 99 -fit off --load-mode none $specArgs --single-turn --reasoning off --ignore-eos -lv $verbosity --log-file `"$base.detail.log`" $ExtraArgs"
"LABEL=$Label PKG=$pkgDir LIB=$lib SPEC=$Spec DRAFTMAX=$DraftMax EXTRAENV=$ExtraEnv ARGS=$arguments" | Out-File "$base.cmd.txt" -Encoding utf8
. (Join-Path $PSScriptRoot 'npu-lock.ps1'); Enter-NpuLock "run-mtp-speed.ps1 $Label"
try {
    $run = Start-Process -FilePath (Join-Path $pkgDir 'bin\llama-cli.exe') -ArgumentList $arguments -PassThru -Wait -WindowStyle Hidden -RedirectStandardOutput "$base.log" -RedirectStandardError "$base.log.err"
} finally { Exit-NpuLock }
"RUN_EXIT=$($run.ExitCode)"
if ($run.ExitCode -ne 0) { throw "run failed: $($run.ExitCode)" }
Select-String -Path "$base.detail.log" -Pattern 'eval time|draft acceptance|accepted|n_drafted|mtp-profile (draft|process).*total' -ErrorAction SilentlyContinue | Select-Object -Last 8 | ForEach-Object { $_.Line }
