#!/bin/bash
# usage: d2rs.sh name [n] [extra repeat_server args...] -- runs pkg-decode2-<name>, appends summary to d2_results.txt
cd /c/Users/hotschmoe/documents/llm_inference
name=$1; n=${2:-10}; shift; shift
python npu_campaign/repeat_server.py --pkg npu_campaign/pkg-decode2-$name --n $n --label rs-$name${LBL} --keep "$@" > npu_campaign/mtp-speed/rs-$name${LBL}.out 2>&1
grep -E "RESULT|distinct completions|Traceback|Error" npu_campaign/mtp-speed/rs-$name${LBL}.out | cut -c1-400 | sed "s/^/[$name$LBL] /" | tee -a npu_campaign/d2_results.txt
grep -E "^row" npu_campaign/mtp-speed/rs-$name${LBL}.out | head -1 | cut -c1-150
