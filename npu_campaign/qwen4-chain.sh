#!/bin/bash
# qwen4-chain.sh script1 script2 ...: runs my queue scripts strictly one after the other; refuses to start if another chain of mine is alive (atomic mkdir lock).
cd /c/Users/hotschmoe/documents/llm_inference
LD=npu_campaign/.qwen4-chain.lock
if ! mkdir $LD 2>/dev/null; then echo "another qwen4 chain is running ($(cat $LD/pid 2>/dev/null)); refusing" >&2; exit 1; fi
echo $$ > $LD/pid
trap 'rm -rf $LD' EXIT
for s in "$@"; do bash npu_campaign/$s > npu_campaign/${s%.sh}.log 2>&1; done
