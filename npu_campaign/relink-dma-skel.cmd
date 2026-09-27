@echo off
setlocal EnableExtensions
set "CLANG=C:\Qualcomm\Hexagon_SDK\6.6.0.0\tools\HEXAGON_Tools\19.0.07\Tools\bin\hexagon-clang.exe"
set "SDK=C:\Qualcomm\Hexagon_SDK\6.6.0.0"
set "PIC=C:\Qualcomm\Hexagon_SDK\6.6.0.0\tools\HEXAGON_Tools\19.0.07\Tools\target\hexagon\lib\v81\G0\pic"
set "LIB=C:\Qualcomm\Hexagon_SDK\6.6.0.0\tools\HEXAGON_Tools\19.0.07\Tools\target\hexagon\lib"
set "DIR=C:\Users\hotschmoe\Documents\llm_inference\native_nvfp4\island_bw\dsp"
cd /d "%DIR%"
"%CLANG%" -mv81 -mcpu=v81 -mhvx=v81 -O2 -G0 -fpic -c dma_bw_skel.c -o dma_bw_skel.c.obj -I"%SDK%\incs" -I"%SDK%\incs\stddef"
if errorlevel 1 exit /b 1
"%CLANG%" -mv81 -mcpu=v81 -mhvx=v81 -G0 -fpic -Wl,-Bsymbolic -Wl,-L"%PIC%" -Wl,-L"%LIB%" -Wl,--no-threads -shared -Wl,-soname,libdma_bw_skel.so -o libdma_bw_skel.so dma_bw_skel.c.obj -lc
if errorlevel 1 exit /b 2
echo RELINK_OK
exit /b 0
