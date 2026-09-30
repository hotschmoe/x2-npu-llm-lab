#!/bin/bash
# ops650 queue 4: bit-exact plain-logit gates for each opt-in fusion (pkg-ops650-s3), one at a time, vs ref = pkg-merged2-m1 (twice).
cd /c/Users/hotschmoe/documents/llm_inference
P=ops650-g1
run() { # label pkg extraenv
  python npu_campaign/lockrun650.py -- powershell -NoProfile -File npu_campaign/run-mtp-speed.ps1 -Label "$1" -Spec none -Tokens 128 -LogitsTrace -Pkg "$2" -ExtraEnv "$3" -Prompt Hello > "npu_campaign/mtp-speed/$1.runout" 2>&1
}
cmp() { python npu_campaign/compare_logits_traces.py "npu_campaign/mtp-speed/$P-$1.logits.bin" "npu_campaign/mtp-speed/$P-$2.logits.bin" --require-identical > "npu_campaign/mtp-speed/$P-cmp-$1-$2.txt" 2>&1; echo "GATE $1 vs $2 exit=$? :: $(tail -n 2 npu_campaign/mtp-speed/$P-cmp-$1-$2.txt | tr '\n' ' ')" >> npu_campaign/ops650-gates.out; }
run $P-ref1 'npu_campaign\pkg-merged2-m1' "X=1"
run $P-ref2 'npu_campaign\pkg-merged2-m1' "X=1"
cmp ref1 ref2
S3='npu_campaign\pkg-ops650-s3'
run $P-off $S3 "X=1"; cmp ref1 off
run $P-gdn $S3 "GGML_HEXAGON_OPS650_FUSE=4096"; cmp ref1 gdn
run $P-rms $S3 "GGML_HEXAGON_OPS650_FUSE=1024"; cmp ref1 rms
run $P-silu $S3 "GGML_HEXAGON_OPS650_FUSE=131072"; cmp ref1 silu
run $P-shexp $S3 "GGML_HEXAGON_OPS650_FUSE=65536"; cmp ref1 shexp
run $P-topk $S3 "GGML_HEXAGON_MOE_TOPK_ALIAS=1"; cmp ref1 topk
for f in gdn rms silu shexp topk off; do rm -f npu_campaign/mtp-speed/$P-$f.logits.bin; done
