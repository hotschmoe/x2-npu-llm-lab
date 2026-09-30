@echo off
rem usage: d2-skel.cmd "DEF1=1;DEF2=1"  -- reconfigure + rebuild ONLY the v81 DSP skel with diag defines (host dll untouched)
setlocal EnableExtensions
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=arm64 -host_arch=arm64 >nul
set "PATH=C:\Program Files\LLVM\bin;%PATH%"
set "HEXAGON_SDK_ROOT=C:\Qualcomm\Hexagon_SDK\6.6.0.0"
set "HEXAGON_TOOLS_ROOT=C:\Qualcomm\Hexagon_SDK\6.6.0.0\tools\HEXAGON_Tools\19.0.07"
set "B=C:\Users\hotschmoe\Documents\llm_inference\llama.cpp-hexagon-int4u8\build-int4u8\ggml\src\ggml-hexagon\htp-v81-prefix\src\htp-v81-build"
cmake -DGGML_HEXAGON_DIAG_DEFINES="%~1" "%B%"
if errorlevel 1 ( echo BUILD_EXIT=9 & exit /b 9 )
cmake --build "%B%" --parallel 6
if errorlevel 1 ( echo BUILD_EXIT=4 & exit /b 4 )
echo BUILD_EXIT=0
