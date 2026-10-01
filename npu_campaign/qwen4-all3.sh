#!/bin/bash
cd /c/Users/hotschmoe/documents/llm_inference
bash npu_campaign/qwen4-m6.sh > npu_campaign/qwen4-m6.log 2>&1
bash npu_campaign/qwen4-m5.sh > npu_campaign/qwen4-m5.log 2>&1
bash npu_campaign/qwen4-m2.sh > npu_campaign/qwen4-m2.log 2>&1
bash npu_campaign/qwen4-m4.sh > npu_campaign/qwen4-m4.log 2>&1
