@echo off
setlocal EnableExtensions
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=arm64 -host_arch=arm64
if errorlevel 1 (
  echo BUILD_EXIT=1
  exit /b 1
)
set "PATH=C:\Program Files\LLVM\bin;%PATH%"
set "OPENCL_SDK_ROOT=C:\Qualcomm\OpenCL_SDK\2.3.2"
set "HEXAGON_SDK_ROOT=C:\Qualcomm\Hexagon_SDK\6.6.0.0"
set "HEXAGON_TOOLS_ROOT=C:\Qualcomm\Hexagon_SDK\6.6.0.0\tools\HEXAGON_Tools\19.0.07"
set "HEXAGON_HTP_CERT=C:\Users\hotschmoe\Certs\ggml-htp-v1.pfx"
set "WINDOWS_SDK_BIN=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0"
cd /d "C:\Users\hotschmoe\Documents\llm_inference\llama.cpp-hexagon"
copy /Y "docs\backend\snapdragon\CMakeUserPresets.json" "CMakeUserPresets.json"
if errorlevel 1 (
  echo BUILD_EXIT=11
  exit /b 11
)
echo CONFIGURE_START
cmake --preset arm64-windows-snapdragon-release -B build-wos
if errorlevel 1 (
  echo BUILD_EXIT=2
  exit /b 2
)
echo BUILD_START
cmake --build build-wos --parallel 12
if errorlevel 1 (
  echo BUILD_EXIT=3
  exit /b 3
)
echo INSTALL_START
cmake --install build-wos --prefix pkg-wos
if errorlevel 1 (
  echo BUILD_EXIT=4
  exit /b 4
)
echo BUILD_EXIT=0
exit /b 0
