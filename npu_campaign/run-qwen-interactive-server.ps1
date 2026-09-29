# Interactive Qwen3.6-35B-A3B Q4_0 chat on the Hexagon NPU.
# This is the server that held about 26-28 tok/s decode on battery, with the
# diagnostic HMX and cache-flush builds left off. Chat UI: http://127.0.0.1:8080
# Stop it with: Stop-Process -Name llama-server
# Batch sizes: -b 2048 -ub 1024 gives ~1.8x prompt throughput on long prompts (1379 tokens: 3.17 s -> 1.78 s to first token) for ~2% lower decode
# than the old -b 512 -ub 128 (server measurements, 2026-09-29). Use -Ub 128 -B 512 for decode benchmarks.
param([int]$B = 2048, [int]$Ub = 1024)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$env:ADSP_LIBRARY_PATH = "$root\llama.cpp-hexagon\pkg-wos\lib;"
$env:GGML_HEXAGON_OPBATCH = '3072'
foreach ($name in @(
    'LLAMA_LOGITS_TRACE','LLAMA_LOGITS_TRACE_FILE','LLAMA_TENSOR_TRACE_FILE','LLAMA_TENSOR_TRACE_FILTER',
    'LLAMA_RETAIN_TRACE_FILE','LLAMA_RETAIN_TRACE_FILTER','LLAMA_RECURRENT_SNAPSHOT_LAYER','LLAMA_FUSE_MTP_VERIFY',
    'GGML_HEXAGON_VERBOSE','GGML_HEXAGON_PROFILE','GGML_HEXAGON_ALLOC_FILL','GGML_HEXAGON_MOE_TOPK_ALIAS',
    'GGML_HEXAGON_ROUTER_HVX','GGML_HEXAGON_MM_SELECT','GGML_HEXAGON_HEAD_PREFETCH','GGML_HEXAGON_FA_SELECT',
    'GGML_HEXAGON_GDN_SELECT','GGML_HEXAGON_NDEV'
)) { Remove-Item "Env:$name" -ErrorAction SilentlyContinue }
& "$root\llama.cpp-hexagon\pkg-wos\bin\llama-server.exe" `
    -m "$root\models\gguf\qwen36-35b-a3b-q4_0\Qwen_Qwen3.6-35B-A3B-Q4_0.gguf" `
    --host 127.0.0.1 --port 8080 --alias qwen36-35b-a3b `
    --device HTP0 -ngl 99 -fit off --load-mode none `
    -c 8192 -b $B -ub $Ub -t 4 -np 1 `
    --cache-prompt --metrics --jinja --reasoning auto
