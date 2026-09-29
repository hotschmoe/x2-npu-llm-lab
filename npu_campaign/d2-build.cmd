@echo off
rem usage: d2-build.cmd "DEF1=1;DEF2=1"   (empty string = no diag defines). Reconfigures the v81 skel build dir, then builds host+skel (build-decode2.cmd).
setlocal EnableExtensions
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=arm64 -host_arch=arm64 >nul
set "PATH=C:\Program Files\LLVM\bin;%PATH%"
set "HEXAGON_SDK_ROOT=C:\Qualcomm\Hexagon_SDK\6.6.0.0"
set "HEXAGON_TOOLS_ROOT=C:\Qualcomm\Hexagon_SDK\6.6.0.0\tools\HEXAGON_Tools\19.0.07"
cmake -DGGML_HEXAGON_DIAG_DEFINES="%~1" "C:\Users\hotschmoe\Documents\llm_inference\llama.cpp-hexagon-decode2\build-decode2\ggml\src\ggml-hexagon\htp-v81-prefix\src\htp-v81-build"
if errorlevel 1 ( echo BUILD_EXIT=9 & exit /b 9 )
call "C:\Users\hotschmoe\Documents\llm_inference\npu_campaign\build-decode2.cmd"
