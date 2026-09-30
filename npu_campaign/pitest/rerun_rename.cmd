@echo off
cd /d %~dp0..\..
python npu_campaign\pitest\run_suite.py --model qwen --reps 3 --cases rename_multi --tag qwen-rename > npu_campaign\pitest\logs\console-qwen-rename.txt 2>&1
python npu_campaign\pitest\run_suite.py --model nemotron --reps 3 --cases rename_multi --tag nemotron-rename > npu_campaign\pitest\logs\console-nemotron-rename.txt 2>&1
