@echo off
setlocal EnableExtensions
set "GGML_HEXAGON_OPBATCH=3072"
set "ADSP_LIBRARY_PATH=C:\Users\hotschmoe\Documents\llm_inference\llama.cpp-hexagon\pkg-wos\lib;"
"C:\Users\hotschmoe\Documents\llm_inference\llama.cpp-hexagon\pkg-wos\bin\llama-cli.exe" -m "C:\Users\hotschmoe\Documents\llm_inference\models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0.gguf" -n 64 -c 512 -b 128 -ub 128 -p Hello --temp 0 --device HTP0 -ngl 99 -fit off --spec-type draft-mtp --spec-draft-n-max 1 --single-turn --reasoning off --ignore-eos > "C:\Users\hotschmoe\Documents\llm_inference\npu_campaign\htp-qwen-mtp-keep2.log" 2> "C:\Users\hotschmoe\Documents\llm_inference\npu_campaign\htp-qwen-mtp-keep2.log.err"
echo RUN_EXIT=%ERRORLEVEL%
exit /b %ERRORLEVEL%
