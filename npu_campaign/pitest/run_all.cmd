@echo off
rem Full comparison: qwen (all cases), gemma, nemotron; one server at a time, NPU lock taken per model.
cd /d %~dp0..\..
if not exist npu_campaign\pitest\results-qwen-full.md python npu_campaign\pitest\run_suite.py --model qwen --reps 3 --tag qwen-full > npu_campaign\pitest\logs\console-qwen-full.txt 2>&1
python npu_campaign\pitest\run_suite.py --model gemma --reps 3 > npu_campaign\pitest\logs\console-gemma.txt 2>&1
python npu_campaign\pitest\run_suite.py --model nemotron --reps 3 > npu_campaign\pitest\logs\console-nemotron.txt 2>&1
