@echo off
setlocal EnableExtensions
set "CLANG=C:\Qualcomm\Hexagon_SDK\6.6.0.0\tools\HEXAGON_Tools\19.0.07\Tools\bin\hexagon-clang.exe"
set "SDK=C:\Qualcomm\Hexagon_SDK\6.6.0.0"
set "PIC=C:\Qualcomm\Hexagon_SDK\6.6.0.0\tools\HEXAGON_Tools\19.0.07\Tools\target\hexagon\lib\v81\G0\pic"
set "LIB=C:\Qualcomm\Hexagon_SDK\6.6.0.0\tools\HEXAGON_Tools\19.0.07\Tools\target\hexagon\lib"
set "DSP=C:\Users\hotschmoe\Documents\llm_inference\native_nvfp4\island_bw\dsp"
set "SRC=C:\Users\hotschmoe\Documents\llm_inference\npu_campaign\dma_min_skel.c"
"%CLANG%" -mv81 -mcpu=v81 -mhvx=v81 -O2 -G0 -fpic -fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables -fno-use-cxa-atexit -c "%SRC%" -o "%DSP%\dma_min.c.obj" -I"%SDK%\incs" -I"%SDK%\incs\stddef"
if errorlevel 1 exit /b 1
"%CLANG%" -mv81 -mcpu=v81 -mhvx=v81 -G0 -fpic -fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables -fno-use-cxa-atexit -Wl,-Bsymbolic -Wl,-L"%PIC%" -Wl,-L"%LIB%" -Wl,--no-threads -Wl,--no-undefined -shared -Wl,-soname,libdma2_skel.so -o "%DSP%\libdma2_skel.so" "%DSP%\dma_min.c.obj" -lc
if errorlevel 1 exit /b 2
echo MIN_RELINK_OK
exit /b 0
