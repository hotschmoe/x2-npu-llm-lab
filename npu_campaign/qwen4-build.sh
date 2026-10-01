#!/bin/bash
# usage: qwen4-build.sh [first]  -- first = 3-step recipe
D='C:\Users\hotschmoe\Documents\llm_inference\npu_campaign'
if [ "$1" = first ]; then
  cmd.exe //c "$D\build-qwen4.cmd"; cmd.exe //c "$D\build-qwen4-htp.cmd"; fi
cmd.exe //c "$D\build-qwen4.cmd"
