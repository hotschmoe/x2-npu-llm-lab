@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=arm64 -host_arch=arm64 >nul
cd /d "%~dp0.."
cl /nologo /MD /EHsc /std:c++17 /O2 /I llama.cpp-hexagon-decode2\ggml\include npu_campaign\requant-head.cpp /Fe:npu_campaign\requant-head.exe /Fo:npu_campaign\requant-head.obj /link /LIBPATH:llama.cpp-hexagon-decode2\build-decode2\ggml\src ggml-base.lib
exit /b %errorlevel%
