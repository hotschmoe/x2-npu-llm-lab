# PITEST log (append-only)

## 2026-09-29  setup
- PI 0.80.2 was already present in ~/.bun/install/global (pi.exe shim fails: no bun on PATH). Installed nothing new from npm; copied the existing x64 bun 1.3.12 (from %TEMP%\bun-node-*) to ~\.bun\bin\bun.exe (user scope). Run PI as `bun ...\pi-coding-agent\dist\cli.js` (node >=22 not installed; bun works under emulation, ~4 s startup).
- Isolated PI config: PI_CODING_AGENT_DIR=npu_campaign/pitest/agentdir (own models.json, no user API keys touched). Provider `local`, api openai-completions, compat supportsDeveloperRole=false, supportsReasoningEffort=false.
- Harness: npu_campaign/pitest/run_suite.py (python; takes NPU lock itself, launches llama-server from pkg-wos on port 8095 with -c 16384 -b 2048 -ub 1024 --jinja, runs each case in a fresh temp dir with `pi --mode json -p --no-session`, parses PI events + server log timings).
- Smoke test (create_file, no_tool on Qwen) passed: prefill ~650-700 tok/s, decode ~27 tok/s.
- Sampler is set server-side (PI sends no temperature): Qwen 0.6/top_p 0.95/top_k 20; Gemma 1.0/0.95/64; Nemotron 0.6/0.95.

## 2026-09-29 21:20  incident: leaked server
- Qwen base-9 suite (27/27 PASS, results-qwen-base9.md) finished but `taskkill /T` on the server PID failed ("child process of dead parent"), so my llama-server (pid 1750648, port 8095) stayed alive ~12 min AFTER the lock was released (idle, no compute, but it held the NPU/25 GB while another agent's run-prefill-ppl ran). Killed it by PID with Stop-Process; harness now uses Popen.kill() and waits. Lesson: verify no llama-server of mine remains after each suite. Exact PID killed was one I launched.
- Added cases 10 fix_bug and 11 rename_multi (harder). Full 11-case runs: qwen-full, gemma, nemotron chained by npu_campaign/pitest/run_all.cmd.

## 2026-09-29 22:05  Qwen3.6 full (11 cases x3) done
- results-qwen-full.md: 33/33 after regrading one checker false negative (read_answer r3, correct answer "October 17, 2026" vs ISO-date checker; checker now accepts common date formats). No leaks, no malformed tool calls, no server-side parse warnings. "tool errors" are bash nonzero exits (legit failing tests) plus one edit "could not find exact text" (model retried and succeeded).
- Speed in agentic use: decode 26-27 tok/s; prefill 390-760 tok/s (small batches after cache hits are slower per token; 6k-token tool result: 762 tok/s).
- Lock polling in run_suite.py reduced to 20 ms (1 s polling starved against lockrun.py's 10 ms).

## 2026-09-29 23:45  Nemotron done, Gemma broken on the NPU
- Nemotron 3.5 Lightning full suite: results-nemotron.md (33 runs). Failures: shell_cmd r1 (final answer landed inside reasoning_content: the model closed its thinking with a stray U+27E9 glyph instead of the think-close tag, so llama-server never split content from reasoning; PI saw an empty assistant text), rename_multi 1/3 and 2/3 (case bug, see below), read_answer x1 (model wrote U+2011 non-breaking hyphens in a date; checker now normalizes; regraded).
- CASE BUG found: rename_multi expected "Total: 2" but round(sum*(1+0.0),2) is 2.0 -> "Total: 2.0"; the test was unsatisfiable unless the model edited the test. Qwen "passed" 3/3 in results-qwen-full, presumably by editing test_all.py (not checked). Fixed: expected "Total: 2.0" and checker requires test_all.py unmodified. Re-running rename_multi for qwen and nemotron (tags qwen-rename / nemotron-rename), then merging into the model tables with `run_suite.py --merge <base>,<extra>`.
- GEMMA 4 26B-A4B produces GARBAGE on the installed HTP build (pkg-wos merged2): every output is token soup (`<unused76>`, random subwords), even the 23-token prompt "Say hello in exactly three words." at temp 0. Not PI related. Probe (npu_campaign/pitest/probe_server.py, /v1/chat/completions, same GGUF):
  - CPU (`--device none -ngl 0`): coherent (reasoning text "Constraint 1: Say hello..." ), so the GGUF and chat template are fine.
  - HTP0, -b2048 -ub1024 / -b512 -ub128 / mamba-std env / no env / -fa off / OPFUSION=0 REPLAY=0 / -ngl 15: all garbage (identical garbage for the first two, deterministic).
  - pkg-decode2-base: garbage as well.  (pkg-prefill and pkg-upstream-u1 probes pending in the lock queue.)
  - Nobody had checked Gemma coherence: PREFILL2_LOG only compares greedy output NPU-vs-NPU ("identical"), which is satisfied by identical garbage.
- Unrelated hygiene: a python crash (UnicodeEncodeError printing a Gemma garbage answer to a cp1252 console) killed one queued run; stdout is now UTF-8. Lock polling 4 ms.
- Because Gemma is unusable on HTP, its tool-calling behaviour is being characterised on CPU (`run_suite.py --cpu`, tag gemma-cpu, still under the NPU lock so it cannot perturb other agents' timings).

## 2026-09-30 00:20  coordinator: skip Gemma
- Per coordinator decision: killed my Gemma CPU suite and the pkg-prefill / pkg-upstream-u1 Gemma probe chain (PIDs I launched only), removed the gemma-cpu partial outputs. Gemma finding stays as: garbage output on installed HTP build, CPU coherent, unresolved / out of scope.

## 2026-09-30 00:45  final
- rename_multi fixed and re-run: Qwen 3/3, Nemotron 3/3 (merged into results-qwen.md / results-nemotron.md; results-qwen-base9.md is the first 9-case Qwen run).
- Final: Qwen3.6-35B-A3B Q4_0 33/33, Nemotron-3.5-Lightning Q4_0 32/33 (1 = reasoning/content split failure), Gemma-4-26B-A4B: unusable on HTP (garbage), not run further (coordinator decision; out of scope, unresolved).
- runs/ (raw PI event streams, ~250 MB) is left untracked on purpose. After each suite one llama-server shows in tasklist as an exited zombie handle (HasExited=True); no live server of mine remains.
