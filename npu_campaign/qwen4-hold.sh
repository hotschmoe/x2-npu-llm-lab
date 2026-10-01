#!/bin/bash
# qwen4-hold.sh "<bash commands>": runs the commands inside ONE lockrun.py hold (NPU_LOCK_HELD=1); uses Git bash explicitly (plain 'bash' resolves to WSL under python subprocess).
GB='C:\Program Files\Git\usr\bin\bash.exe'
cd /c/Users/hotschmoe/documents/llm_inference
python npu_campaign/lockrun.py -- "$GB" -c "cd /c/Users/hotschmoe/documents/llm_inference; source npu_campaign/qwen4-lib.sh; $1"
