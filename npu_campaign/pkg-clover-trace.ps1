# Package the Clover worktree build (llama.cpp-hexagon-clover-trace\build-clover-trace) into npu_campaign\pkg-clover-trace-<Name>\{bin,lib}.
# Host binaries come from build-clover\bin; the v81 DSP skel from the build's htp-v81 external project; catalog is re-signed.
# Run with:  llama-speculative-simple from npu_campaign\pkg-clover-trace-<Name> ...
param([Parameter(Mandatory = $true)][string]$Name)
$ErrorActionPreference = 'Stop'
if ($Name -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Invalid name' }
$repo = Split-Path $PSScriptRoot -Parent
$bld = "$repo\llama.cpp-hexagon-clover-trace\build-clover-trace"
$dst = "$PSScriptRoot\pkg-clover-trace-$Name"
if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
New-Item -ItemType Directory "$dst\bin", "$dst\lib" | Out-Null
Copy-Item "$bld\bin\*.dll" "$dst\bin\"
# our OpenCL build crashes in the Adreno compiler (CB.dll) at backend init; the installed one is unused on HTP0 runs
Copy-Item "$repo\llama.cpp-hexagon\pkg-wos\bin\ggml-opencl.dll" "$dst\bin\" -Force
foreach ($e in 'llama-cli.exe', 'llama-bench.exe', 'llama-completion.exe', 'llama-perplexity.exe', 'test-backend-ops.exe', 'llama-server.exe', 'llama-speculative-simple.exe', 'llama-eval-callback.exe', 'llama-clover-draft-dump.exe') {
    if (Test-Path "$bld\bin\$e") { Copy-Item "$bld\bin\$e" "$dst\bin\" }
}
$so = "$bld\ggml\src\ggml-hexagon\htp-v81-prefix\src\htp-v81-build\libggml-htp-v81.so"
if (-not (Test-Path $so)) { throw 'no v81 skel' }
foreach ($f in 'libggml-htp.inf', 'libggml-htp-v73.so', 'libggml-htp-v75.so', 'libggml-htp-v79.so') { Copy-Item "$repo\llama.cpp-hexagon\pkg-wos\lib\$f" "$dst\lib\" }
Copy-Item $so "$dst\lib\libggml-htp-v81.so"
$sdkBin = 'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x86'
& "$sdkBin\inf2cat.exe" "/driver:$dst\lib" '/os:10_25H2_ARM64' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'inf2cat failed' }
& "$sdkBin\signtool.exe" sign /fd sha256 /f 'C:\Users\hotschmoe\Certs\ggml-htp-v1.pfx' "$dst\lib\libggml-htp.cat" | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'sign failed' }
& "$sdkBin\signtool.exe" verify /pa /c "$dst\lib\libggml-htp.cat" "$dst\lib\libggml-htp-v81.so" | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'verify failed' }
$h = @{}
foreach ($f in "bin\ggml-hexagon.dll", "bin\llama.dll", "bin\llama-common.dll", "lib\libggml-htp-v81.so") { $h[$f] = (Get-FileHash "$dst\$f").Hash.Substring(0, 16) }
"PKG=$dst " + (($h.GetEnumerator() | Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join ' ')
