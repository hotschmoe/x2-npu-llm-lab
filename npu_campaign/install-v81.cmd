@echo off
setlocal EnableExtensions
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=arm64 -host_arch=arm64
if errorlevel 1 exit /b 1
set "PATH=C:\Program Files\LLVM\bin;%PATH%"
set "HEXAGON_SDK_ROOT=C:\Qualcomm\Hexagon_SDK\6.6.0.0"
set "HEXAGON_TOOLS_ROOT=C:\Qualcomm\Hexagon_SDK\6.6.0.0\tools\HEXAGON_Tools\19.0.07"
set "HEXAGON_HTP_CERT=C:\Users\hotschmoe\Certs\ggml-htp-v1.pfx"
set "WINDOWS_SDK_BIN=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0"
set "SRC=C:\Users\hotschmoe\Documents\llm_inference\llama.cpp-hexagon"
set "HEX=%SRC%\build-wos\ggml\src\ggml-hexagon"
copy /Y "%HEX%\htp-v81-prefix\src\htp-v81-build\libggml-htp-v81.so" "%HEX%\libggml-htp-v81.so"
cd /d "%SRC%"
cmake --build build-wos --target libggml-htp-cat
if errorlevel 1 exit /b 3
copy /Y "%HEX%\libggml-htp-v81.so" "%SRC%\pkg-wos\lib\"
copy /Y "%HEX%\libggml-htp.cat" "%SRC%\pkg-wos\lib\"
copy /Y "C:\Users\hotschmoe\Documents\llm_inference\npu_campaign\ggml-opencl-stub.dll" "%SRC%\pkg-wos\bin\ggml-opencl.dll"
echo BUILD_EXIT=0
exit /b 0
