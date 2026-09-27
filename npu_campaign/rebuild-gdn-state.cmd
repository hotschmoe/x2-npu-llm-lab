@echo off
setlocal EnableExtensions
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=arm64 -host_arch=arm64
if errorlevel 1 (
  echo BUILD_EXIT=1
  exit /b 1
)
set "PATH=C:\Program Files\LLVM\bin;%PATH%"
set "HEXAGON_SDK_ROOT=C:\Qualcomm\Hexagon_SDK\6.6.0.0"
set "HEXAGON_TOOLS_ROOT=C:\Qualcomm\Hexagon_SDK\6.6.0.0\tools\HEXAGON_Tools\19.0.07"
set "HEXAGON_HTP_CERT=C:\Users\hotschmoe\Certs\ggml-htp-v1.pfx"
set "WINDOWS_SDK_BIN=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0"
cd /d "C:\Users\hotschmoe\Documents\llm_inference\llama.cpp-hexagon"
echo REBUILD_START
cmake --build "build-wos\ggml\src\ggml-hexagon\htp-v81-prefix\src\htp-v81-build" --parallel 12
if errorlevel 1 (
  echo BUILD_EXIT=2
  exit /b 2
)
cmake --build build-wos --parallel 12 --target llama-completion
if errorlevel 1 (
  echo BUILD_EXIT=3
  exit /b 3
)
copy /Y "build-wos\ggml\src\ggml-hexagon\htp-v81-prefix\src\htp-v81-build\libggml-htp-v81.so" "pkg-wos\lib\libggml-htp-v81.so"
if errorlevel 1 (
  echo BUILD_EXIT=5
  exit /b 5
)
"%WINDOWS_SDK_BIN%\x86\inf2cat.exe" /driver:"pkg-wos\lib" /os:10_25H2_ARM64
if errorlevel 1 (
  echo BUILD_EXIT=6
  exit /b 6
)
"%WINDOWS_SDK_BIN%\x86\signtool.exe" sign /fd sha256 /f "%HEXAGON_HTP_CERT%" "pkg-wos\lib\libggml-htp.cat"
if errorlevel 1 (
  echo BUILD_EXIT=7
  exit /b 7
)
copy /Y "build-wos\bin\llama-completion.exe" "pkg-wos\bin\"
copy /Y "build-wos\bin\llama.dll" "pkg-wos\bin\"
copy /Y "build-wos\bin\ggml-hexagon.dll" "pkg-wos\bin\"
copy /Y "build-wos\bin\ggml.dll" "pkg-wos\bin\"
copy /Y "C:\Users\hotschmoe\Documents\llm_inference\npu_campaign\ggml-opencl-stub.dll" "pkg-wos\bin\ggml-opencl.dll"
echo BUILD_EXIT=0
exit /b 0
