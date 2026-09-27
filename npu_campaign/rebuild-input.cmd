@echo off
setlocal EnableExtensions
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=arm64 -host_arch=arm64
if errorlevel 1 (
  echo BUILD_EXIT=1
  exit /b 1
)
set "PATH=C:\Program Files\LLVM\bin;%PATH%"
set "HEXAGON_HTP_CERT=C:\Users\hotschmoe\Certs\ggml-htp-v1.pfx"
set "WINDOWS_SDK_BIN=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0"
cd /d "C:\Users\hotschmoe\Documents\llm_inference\llama.cpp-hexagon"
echo REBUILD_START
cmake --build build-wos --parallel 12 --target llama-completion
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
copy /Y "build-wos\ggml\src\ggml-hexagon\htp-v81-prefix\src\htp-v81-build\libggml-htp-v81.so" "pkg-wos\lib\libggml-htp-v81.so"
if errorlevel 1 (
  echo BUILD_EXIT=5
  exit /b 5
)
"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x86\inf2cat.exe" /driver:"pkg-wos\lib" /os:10_25H2_ARM64
if errorlevel 1 (
  echo BUILD_EXIT=6
  exit /b 6
)
"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x86\signtool.exe" sign /fd sha256 /f "C:\Users\hotschmoe\Certs\ggml-htp-v1.pfx" "pkg-wos\lib\libggml-htp.cat"
if errorlevel 1 (
  echo BUILD_EXIT=7
  exit /b 7
)
copy /Y "C:\Users\hotschmoe\Documents\llm_inference\npu_campaign\ggml-opencl-stub.dll" "pkg-wos\bin\ggml-opencl.dll"
if errorlevel 1 (
  echo BUILD_EXIT=8
  exit /b 8
)
echo BUILD_EXIT=0
exit /b 0
