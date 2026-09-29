# Install npu_campaign\pkg-final (merged prefill + MTP build) into llama.cpp-hexagon\pkg-wos, backing up every file it replaces.
param([string]$Src = 'pkg-final')
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'npu-lock.ps1'); Enter-NpuLock 'install-merged'
try {
    $repo = Split-Path $PSScriptRoot -Parent
    $dst = "$repo\llama.cpp-hexagon\pkg-wos"; $src = "$PSScriptRoot\$Src"; $bk = "$PSScriptRoot\backup-merged-install-prechange"
    if (Test-Path $bk) { throw "backup dir exists: $bk (refusing to overwrite an earlier backup)" }
    New-Item -ItemType Directory "$bk\bin", "$bk\lib" | Out-Null
    foreach ($sub in 'bin', 'lib') {
        foreach ($f in Get-ChildItem "$src\$sub" -File) {
            $t = "$dst\$sub\$($f.Name)"
            if (Test-Path $t) { Copy-Item $t "$bk\$sub\$($f.Name)" }
            Copy-Item $f.FullName $t -Force
        }
    }
    "BACKED_UP=" + (Get-ChildItem "$bk\bin", "$bk\lib" -File | Measure-Object).Count + " files, " + [math]::Round((Get-ChildItem $bk -Recurse -File | Measure-Object Length -Sum).Sum / 1MB) + " MB"
    "INSTALLED: " + (@("$dst\lib\libggml-htp-v81.so", "$dst\lib\libggml-htp.cat", "$dst\bin\ggml-hexagon.dll", "$dst\bin\llama.dll") | ForEach-Object { "$(Split-Path $_ -Leaf)=$((Get-FileHash $_).Hash.Substring(0,8))" }) -join ' '
} finally { Exit-NpuLock }
