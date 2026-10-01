@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=arm64 -host_arch=arm64 >nul
set "PATH=C:\Program Files\LLVM\bin;%PATH%"
set "OPENCL_SDK_ROOT=C:\Qualcomm\OpenCL_SDK\2.3.2"
set "HEXAGON_SDK_ROOT=C:\Qualcomm\Hexagon_SDK\6.6.0.0"
set "HEXAGON_TOOLS_ROOT=C:\Qualcomm\Hexagon_SDK\6.6.0.0\tools\HEXAGON_Tools\19.0.07"
cd /d C:\Users\hotschmoe\Documents\llm_inference\llama.cpp-hexagon-layout
cmake --build build-layout --target htp-v81 --parallel 4
echo EXIT=%errorlevel%
