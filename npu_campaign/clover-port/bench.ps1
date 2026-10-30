# llama-bench (Clover, HTP0): pp16, pp256, tg32 for a sequence of packages under ONE lock hold (ABBA).
param([Parameter(Mandatory = $true)][string]$Tag, [Parameter(Mandatory = $true)][object[]]$Seq, [int]$Reps = 2)
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
. "$repo\npu_campaign\npu-lock.ps1"
Enter-NpuLock "bench.ps1 $Tag"
$i = 0
try {
    foreach ($a in $Seq) {
        $i++
        $pkgDir = (Resolve-Path (Join-Path $repo $a.pkg)).Path
        $env:ADSP_LIBRARY_PATH = "$pkgDir\lib;"
        foreach ($k in $a.env.Keys) { Set-Item "env:$k" $a.env[$k] }
        $out = "$repo\npu_campaign\clover-logs\bench-$Tag-$i-$($a.n)"
        $args_ = "-m `"$repo\models\gguf\clover-35b-a3b-q4_0\Clover-35B-A3B-Preview-Q4_0.gguf`" -p 16,256 -n 32 -ngl 99 --device HTP0 -fa 1 -ctk f16 -ctv f16 -t 8 -r $Reps -o md"
        $run = Start-Process -FilePath "$pkgDir\bin\llama-bench.exe" -ArgumentList $args_ -WindowStyle Hidden -PassThru -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err"
        if (-not $run.WaitForExit(540000)) { Stop-Process -Id $run.Id -Force; "TIMEOUT" }
        foreach ($k in $a.env.Keys) { Remove-Item "env:$k" -ErrorAction SilentlyContinue }
        "--- $Tag #$i $($a.n) exit $($run.ExitCode)"
        Get-Content "$out.log" | Where-Object { $_ -match 'pp16|pp256|tg32' }
    }
} finally { if ($run -and -not $run.HasExited) { Stop-Process -Id $run.Id -Force -ErrorAction SilentlyContinue }; Exit-NpuLock }
