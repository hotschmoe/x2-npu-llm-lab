$ErrorActionPreference = 'Stop'
$root = 'C:\Users\hotschmoe\Documents\llm_inference\models\gguf'
$log = Join-Path $root 'download.log'
New-Item -ItemType Directory -Force -Path $root | Out-Null
function Say($msg) {
    $line = '{0} {1}' -f (Get-Date -Format 'HH:mm:ss'), $msg
    Add-Content -Path $log -Value $line
    Write-Host $line
}
$jobs = @(
    @{ Repo = 'bartowski/Qwen_Qwen3.6-35B-A3B-GGUF'; File = 'Qwen_Qwen3.6-35B-A3B-Q4_0.gguf'; Dir = 'qwen36-35b-a3b-q4_0' },
    @{ Repo = 'FreedomAISVR/Qwen3.6-35B-A3B-MXFP4-MOE-GGUF'; File = 'qwen3.6-35b-a3b-mxfp4_moe.gguf'; Dir = 'qwen36-35b-a3b-mxfp4-moe' },
    @{ Repo = 'google/gemma-4-26B-A4B-it-qat-q4_0-gguf'; File = 'gemma-4-26B_q4_0-it.gguf'; Dir = 'gemma4-26b-a4b-qat-q4_0' }
)
foreach ($job in $jobs) {
    $dest = Join-Path $root $job.Dir
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Say ("START {0} -> {1}" -f $job.File, $dest)
    & hf download $job.Repo $job.File --local-dir $dest
    if ($LASTEXITCODE) { Say ("FAILED {0} exit {1}" -f $job.File, $LASTEXITCODE); exit $LASTEXITCODE }
    $path = Join-Path $dest $job.File
    $len = (Get-Item $path).Length
    Say ("DONE {0} bytes={1}" -f $job.File, $len)
}
Say 'ALL_DONE'
exit 0
