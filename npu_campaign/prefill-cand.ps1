# Package + sign a prefill candidate DSP skel into npu_campaign\prefill-cand-<Name>\ (installed pkg-wos untouched).
# Bench with:  run-prefill-bench.ps1 -Label x -LibDir npu_campaign\prefill-cand-<Name>
param([Parameter(Mandatory = $true)][string]$Name, [string]$Build = 'build-prefill-htp')
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$sdkBin = 'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x86'
$cert = 'C:\Users\hotschmoe\Certs\ggml-htp-v1.pfx'
$so = Get-ChildItem "$repo\llama.cpp-hexagon\$Build" -Recurse -Filter libggml-htp-v81.so | Select-Object -First 1
if (-not $so) { throw 'No built libggml-htp-v81.so' }
$dst = "$PSScriptRoot\prefill-cand-$Name"
if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
New-Item -ItemType Directory $dst | Out-Null
foreach ($f in 'libggml-htp.inf','libggml-htp-v73.so','libggml-htp-v75.so','libggml-htp-v79.so') { Copy-Item "$repo\llama.cpp-hexagon\pkg-wos\lib\$f" $dst }
Copy-Item $so.FullName "$dst\libggml-htp-v81.so"
& "$sdkBin\inf2cat.exe" "/driver:$dst" '/os:10_25H2_ARM64' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'inf2cat failed' }
& "$sdkBin\signtool.exe" sign /fd sha256 /f $cert "$dst\libggml-htp.cat" | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'sign failed' }
& "$sdkBin\signtool.exe" verify /pa /c "$dst\libggml-htp.cat" "$dst\libggml-htp-v81.so" | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'verify failed' }
"CAND=$dst SO=$((Get-FileHash "$dst\libggml-htp-v81.so").Hash)"
