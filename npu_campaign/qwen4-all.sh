#!/bin/bash
cd /c/Users/hotschmoe/documents/llm_inference
bash npu_campaign/qwen4-m1.sh > npu_campaign/qwen4-m1.log 2>&1
bash npu_campaign/qwen4-m2.sh > npu_campaign/qwen4-m2.log 2>&1
