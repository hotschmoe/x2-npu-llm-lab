@echo off
rem DSP-only (v81) build into llama.cpp-hexagon-prefill2\build-prefill2-htp with all diagnostic options at default (OFF).
setlocal EnableExtensions
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=arm64 -host_arch=arm64 >nul
set "NINJA=C:\Users\hotschmoe\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe\ninja.exe"
set "SRC=%~dp0..\llama.cpp-hexagon-prefill2\ggml\src\ggml-hexagon\htp"
set "BLD=%~dp0..\llama.cpp-hexagon-prefill2\build-prefill2-htp"
if not "%PF_BLD%"=="" set "BLD=%~dp0..\llama.cpp-hexagon-prefill2\%PF_BLD%"
if not exist "%BLD%\build.ninja" (
  cmake -S "%SRC%" -B "%BLD%" -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA%" -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="%SRC%\cmake-toolchain.cmake" -DHEXAGON_SDK_ROOT=C:/Qualcomm/Hexagon_SDK/6.6.0.0 -DHEXAGON_TOOLS_ROOT=C:/Qualcomm/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07 -DHEXAGON_HTP_DEBUG=OFF -DDSP_VERSION=v81 %PF_CMAKE_EXTRA% -DPREBUILT_LIB_DIR=toolv19_v81
  if errorlevel 1 ( echo BUILD_EXIT=2 & exit /b 2 )
)
cmake --build "%BLD%" --parallel 12 %*
if errorlevel 1 ( echo BUILD_EXIT=3 & exit /b 3 )
echo BUILD_EXIT=0
