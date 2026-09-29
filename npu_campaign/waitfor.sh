#!/bin/bash
# usage: waitfor.sh maxsec file...   -> waits until any listed file is non-empty (or timeout), then prints the tail of all non-empty ones
cd /c/Users/hotschmoe/documents/llm_inference/npu_campaign
max=$1; shift
end=$((SECONDS+max))
while [ $SECONDS -lt $end ]; do
  for f in "$@"; do [ -s "$f" ] && break 2; done
  sleep 8
done
for f in "$@"; do [ -s "$f" ] && { echo "=== $f"; tail -${TAILN:-14} "$f" | cut -c1-220; }; done
date
