@echo off
setlocal EnableExtensions
set "GGML_HEXAGON_OPBATCH=3072"
set "ADSP_LIBRARY_PATH=C:\Users\hotschmoe\Documents\llm_inference\llama.cpp-hexagon\pkg-wos\lib;"
"C:\Users\hotschmoe\Documents\llm_inference\llama.cpp-hexagon\pkg-wos\bin\llama-bench.exe" -m "C:\Users\hotschmoe\Documents\llm_inference\models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0.gguf" -p 0 -n 64 -d 0,2048,8192 -b 512 -ub 128 -ngl 99 -dev HTP0 -t 4 -r 1 -ctk f16 -ctv f16 -fa on > "C:\Users\hotschmoe\Documents\llm_inference\npu_campaign\bench-depth.log" 2> "C:\Users\hotschmoe\Documents\llm_inference\npu_campaign\bench-depth.log.err"
echo RUN_EXIT=%ERRORLEVEL%
exit /b %ERRORLEVEL%
