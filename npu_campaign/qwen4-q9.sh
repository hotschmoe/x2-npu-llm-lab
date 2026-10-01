#!/bin/bash
# q9 (item 4): MTP vs plain at agentic sampling (temp 0.6 top-p 0.95 top-k 20) on pkg-final2-f1; per prompt class x seed ONE lockrun hold of 3 interleaved arms (rotated).
# arms: P plain | M mtp d1 (fc+dv65536) | G mtp d1 + first-token p-min 0.5.  256 tokens.  Usage: qwen4-q9.sh [seeds...]  (default: 11 22)
cd /c/Users/hotschmoe/documents/llm_inference
source npu_campaign/qwen4-lib.sh
PK='npu_campaign\pkg-final2-f1'
SEEDS=${@:-11 22}
wait_ac
for seed in $SEEDS; do
  for cls in CODE JSON SKY; do
    pr=${!cls}
    case $(( seed % 3 )) in 0) order="P M G";; 1) order="M G P";; 2) order="G P M";; esac
    cmds=""
    for a in $order; do
      case $a in
        P) cmds="$cmds q4run q4-t06-$cls-s$seed-P '$PK' none 256 \"$pr\" 'X=1' -ExtraArgs '$T06 --seed $seed' >> npu_campaign/qwen4-temp06.out;";;
        M) cmds="$cmds q4run q4-t06-$cls-s$seed-M '$PK' mtp 256 \"$pr\" '$FCDV' -DraftMax 1 -ExtraArgs '$T06 --seed $seed' >> npu_campaign/qwen4-temp06.out;";;
        G) cmds="$cmds q4run q4-t06-$cls-s$seed-G '$PK' mtp 256 \"$pr\" '$FCDV' -DraftMax 1 -PMin 0.5 -ExtraArgs '$T06 --seed $seed' >> npu_campaign/qwen4-temp06.out;";;
      esac
    done
    bash npu_campaign/qwen4-hold.sh "$cmds"
  done
done
