@echo off
rem Host-side (ggml-hexagon.dll) build for prefill experiments into llama.cpp-hexagon\build-prefill-host; produces pkg-prefill\bin.
setlocal EnableExtensions
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=arm64 -host_arch=arm64 >nul
set "PATH=C:\Program Files\LLVM\bin;%PATH%"
set "OPENCL_SDK_ROOT=C:\Qualcomm\OpenCL_SDK\2.3.2"
set "HEXAGON_SDK_ROOT=C:\Qualcomm\Hexagon_SDK\6.6.0.0"
set "HEXAGON_TOOLS_ROOT=C:\Qualcomm\Hexagon_SDK\6.6.0.0\tools\HEXAGON_Tools\19.0.07"
set "HEXAGON_HTP_CERT=C:\Users\hotschmoe\Certs\ggml-htp-v1.pfx"
set "WINDOWS_SDK_BIN=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0"
cd /d "%~dp0..\llama.cpp-hexagon"
if not exist build-prefill-host\build.ninja (
  cmake --preset arm64-windows-snapdragon-release -B build-prefill-host
  if errorlevel 1 ( echo BUILD_EXIT=2 & exit /b 2 )
)
cmake --build build-prefill-host --target htp-v73 htp-v75 htp-v79 htp-v81 --parallel 8
if errorlevel 1 ( echo BUILD_EXIT=4 & exit /b 4 )
cmake --build build-prefill-host --target ggml-hexagon --parallel 8
if errorlevel 1 ( echo BUILD_EXIT=3 & exit /b 3 )
echo BUILD_EXIT=0
