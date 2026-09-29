#!/bin/bash
# usage: d2v.sh name "DEFS"  -- build DSP variant + package
cd /c/Users/hotschmoe/documents/llm_inference
powershell -NoProfile -File npu_campaign/d2-variant.ps1 -Name "$1" -Defines "$2" 2>&1 | tail -2
