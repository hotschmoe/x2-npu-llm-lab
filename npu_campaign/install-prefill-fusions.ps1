# Install the signed candidate skel+catalog AND the host ggml-hexagon.dll into llama.cpp-hexagon\pkg-wos, backing up what is there first.
param([string]$Cand = 'main2', [string]$HostDll = 'llama.cpp-hexagon\build-prefill-host\bin\ggml-hexagon.dll')
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'npu-lock.ps1'); Enter-NpuLock 'install-prefill-fusions'
try {
    $repo = Split-Path $PSScriptRoot -Parent
    $lib = "$repo\llama.cpp-hexagon\pkg-wos\lib"; $bin = "$repo\llama.cpp-hexagon\pkg-wos\bin"
    $cand = "$PSScriptRoot\prefill-cand-$Cand"; $bk = "$PSScriptRoot\backup-prefill-fusions-prechange"
    if (-not (Test-Path $bk)) {
        New-Item -ItemType Directory $bk | Out-Null
        foreach ($f in 'libggml-htp-v81.so','libggml-htp.cat') { Copy-Item "$lib\$f" "$bk\$f" }
        Copy-Item "$bin\ggml-hexagon.dll" "$bk\ggml-hexagon.dll"
        "BACKUP: " + ((Get-ChildItem $bk | ForEach-Object { "$($_.Name)=$((Get-FileHash $_.FullName).Hash.Substring(0,8))" }) -join ' ')
    }
    foreach ($f in 'libggml-htp-v81.so','libggml-htp.cat') { Copy-Item "$cand\$f" "$lib\$f" -Force }
    Copy-Item "$repo\$HostDll" "$bin\ggml-hexagon.dll" -Force
    "INSTALLED: " + (@("$lib\libggml-htp-v81.so","$lib\libggml-htp.cat","$bin\ggml-hexagon.dll") | ForEach-Object { "$(Split-Path $_ -Leaf)=$((Get-FileHash $_).Hash.Substring(0,8))" }) -join ' '
} finally { Exit-NpuLock }
