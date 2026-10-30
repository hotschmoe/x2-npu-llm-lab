@echo off
rem Host+DSP build of the upstream worktree (Clover PR 31306 on upstream master) into build-clover-topk.
rem Usage: build-clover-topk.cmd [extra cmake -D flags]   (configure only happens once; pass flags on first run or delete cache)
rem Then package with pkg-clover.ps1 -Name <x>.
setlocal EnableExtensions
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=arm64 -host_arch=arm64 >nul
if errorlevel 1 ( echo BUILD_EXIT=1 & exit /b 1 )
set "PATH=C:\Program Files\LLVM\bin;%PATH%"
set "OPENCL_SDK_ROOT=C:\Qualcomm\OpenCL_SDK\2.3.2"
set "HEXAGON_SDK_ROOT=C:\Qualcomm\Hexagon_SDK\6.6.0.0"
set "HEXAGON_TOOLS_ROOT=C:\Qualcomm\Hexagon_SDK\6.6.0.0\tools\HEXAGON_Tools\19.0.07"
set "HEXAGON_HTP_CERT="
set "WINDOWS_SDK_BIN=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0"
cd /d "C:\Users\hotschmoe\Documents\llm_inference\llama.cpp-hexagon-clover-topk"
if not exist CMakeUserPresets.json copy /Y "docs\backend\snapdragon\CMakeUserPresets.json" "CMakeUserPresets.json" >nul
if not exist build-clover-topk\build.ninja (
  cmake --preset arm64-windows-snapdragon-release -B build-clover-topk -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=ON -DGGML_HEXAGON_HTP_CERT= %*
  if errorlevel 1 ( echo BUILD_EXIT=2 & exit /b 2 )
)
cmake --build build-clover-topk --parallel 4 --target llama-cli llama-bench llama-completion llama-perplexity llama-server llama-speculative-simple llama-eval-callback test-backend-ops
if %errorlevel% neq 0 ( echo BUILD_EXIT=3 & exit /b 3 )
rem build the v81 skel ExternalProject target explicitly (upstream master layout)
cmake --build build-clover-topk --parallel 4 --target htp-v81
if %errorlevel% neq 0 ( echo BUILD_EXIT=4 & exit /b 4 )
echo BUILD_EXIT=0
exit /b 0
