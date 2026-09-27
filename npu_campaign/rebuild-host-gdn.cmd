@echo off
setlocal EnableExtensions
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=arm64 -host_arch=arm64
if errorlevel 1 exit /b 1
set "PATH=C:\Program Files\LLVM\bin;%PATH%"
cd /d "C:\Users\hotschmoe\Documents\llm_inference\llama.cpp-hexagon"
cmake --build build-wos --parallel 12 --target llama-completion
if errorlevel 1 exit /b 3
copy /Y "build-wos\bin\ggml-hexagon.dll" "pkg-wos\bin\"
copy /Y "build-wos\bin\llama-completion.exe" "pkg-wos\bin\"
copy /Y "build-wos\bin\llama.dll" "pkg-wos\bin\"
copy /Y "build-wos\bin\ggml.dll" "pkg-wos\bin\"
copy /Y "C:\Users\hotschmoe\Documents\llm_inference\npu_campaign\ggml-opencl-stub.dll" "pkg-wos\bin\ggml-opencl.dll"
echo BUILD_EXIT=0
exit /b 0
