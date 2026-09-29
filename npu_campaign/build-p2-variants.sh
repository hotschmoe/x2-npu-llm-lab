#!/bin/bash
# usage: build-p2-variants.sh "name=nnn ..."  -> sets PF_DBL in pf_probe.h, builds, packages prefill2-cand-<name>
H=/c/Users/hotschmoe/documents/llm_inference/llama.cpp-hexagon-prefill2/ggml/src/ggml-hexagon/htp/pf_probe.h
cd /c/Users/hotschmoe/documents/llm_inference/npu_campaign
for kv in "$@"; do
  n=${kv%%=*}; v=${kv##*=}
  IFS=, read -r a b c d <<< "$v"; c=${c:-0}; d=${d:-3}
  sed -i "s/^#define PF_DBL .*/#define PF_DBL $a/; s/^#define PF_ROUTER_K .*/#define PF_ROUTER_K $b/; s/^#define PF_TOPK_SELECT .*/#define PF_TOPK_SELECT $c/; s/^#define PF_RR .*/#define PF_RR $d/" $H
  powershell -NoProfile -Command "& .\build-prefill2-htp.cmd" 2>&1 | grep -E "BUILD_EXIT|error" 
  powershell -NoProfile -File prefill2-cand.ps1 -Name $n 2>&1 | tail -1
done
sed -i "s/^#define PF_DBL .*/#define PF_DBL 0/; s/^#define PF_ROUTER_K .*/#define PF_ROUTER_K 16/; s/^#define PF_TOPK_SELECT .*/#define PF_TOPK_SELECT 0/; s/^#define PF_RR .*/#define PF_RR 3/" $H
