@echo off
rem Host+DSP build of the MTP-speedup worktree (..\llama.cpp-hexagon-decode2, branch mtp-speedup) into build-decode2.
rem Usage: build-decode2.cmd [extra cmake -D flags]   (configure only happens once; pass flags on first run or delete cache)
rem Then package with mtp-pkg.ps1 -Name <x>.
setlocal EnableExtensions
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=arm64 -host_arch=arm64 >nul
if errorlevel 1 ( echo BUILD_EXIT=1 & exit /b 1 )
set "PATH=C:\Program Files\LLVM\bin;%PATH%"
set "OPENCL_SDK_ROOT=C:\Qualcomm\OpenCL_SDK\2.3.2"
set "HEXAGON_SDK_ROOT=C:\Qualcomm\Hexagon_SDK\6.6.0.0"
set "HEXAGON_TOOLS_ROOT=C:\Qualcomm\Hexagon_SDK\6.6.0.0\tools\HEXAGON_Tools\19.0.07"
set "HEXAGON_HTP_CERT="
set "WINDOWS_SDK_BIN=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0"
cd /d "C:\Users\hotschmoe\Documents\llm_inference\llama.cpp-hexagon-decode2"
if not exist CMakeUserPresets.json copy /Y "docs\backend\snapdragon\CMakeUserPresets.json" "CMakeUserPresets.json" >nul
if not exist build-decode2\build.ninja (
  cmake --preset arm64-windows-snapdragon-release -B build-decode2 -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DGGML_HEXAGON_HTP_CERT= %*
  if errorlevel 1 ( echo BUILD_EXIT=2 & exit /b 2 )
)
cmake --build build-decode2 --parallel 6 --target llama-server
if %errorlevel% neq 0 ( echo BUILD_EXIT=3 & exit /b 3 )
rem the ExternalProject does not track DSP source changes: build the v81 skel explicitly
if %errorlevel% neq 0 ( echo BUILD_EXIT=4 & exit /b 4 )
echo BUILD_EXIT=0
exit /b 0
