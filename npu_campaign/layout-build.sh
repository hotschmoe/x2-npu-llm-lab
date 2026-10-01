#!/bin/bash
# usage: layout-build.sh [first]  -- first = 3-step recipe
D='C:\Users\hotschmoe\Documents\llm_inference\npu_campaign'
if [ "$1" = first ]; then
  cmd.exe //c "$D\build-layout.cmd"; cmd.exe //c "$D\build-layout-htp.cmd"; fi
cmd.exe //c "$D\build-layout.cmd"
