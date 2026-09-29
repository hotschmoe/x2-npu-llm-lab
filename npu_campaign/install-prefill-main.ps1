# Install the signed prefill-cand-main skel+catalog into llama.cpp-hexagon\pkg-wos\lib, backing up what is there first.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'npu-lock.ps1'); Enter-NpuLock 'install-prefill-main'
try {
    $repo = Split-Path $PSScriptRoot -Parent
    $lib = "$repo\llama.cpp-hexagon\pkg-wos\lib"; $cand = "$PSScriptRoot\prefill-cand-main"; $bk = "$PSScriptRoot\backup-prefill-pipe-prechange"
    if (-not (Test-Path $bk)) {
        New-Item -ItemType Directory $bk | Out-Null
        foreach ($f in 'libggml-htp-v81.so','libggml-htp.cat') { Copy-Item "$lib\$f" "$bk\$f" }
        "BACKUP: " + ((Get-ChildItem $bk | ForEach-Object { "$($_.Name)=$((Get-FileHash $_.FullName).Hash.Substring(0,8))" }) -join ' ')
    }
    foreach ($f in 'libggml-htp-v81.so','libggml-htp.cat') { Copy-Item "$cand\$f" "$lib\$f" -Force }
    "INSTALLED: " + ((Get-ChildItem $lib -Filter 'libggml-htp*' | ForEach-Object { "$($_.Name)=$((Get-FileHash $_.FullName).Hash.Substring(0,8))" }) -join ' ')
} finally { Exit-NpuLock }
