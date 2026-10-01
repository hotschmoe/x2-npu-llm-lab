# Local Qwen3.6-35B-A3B (Q4_0) serving on the Hexagon NPU

One consumer, one model process, one port. Everything here lives in `serve\`; nothing touches the llama.cpp fork (`llama.cpp-hexagon\pkg-wos` is only executed).

## Start / stop

```
serve\start-qwen.cmd -Mode chat          # plain decode, http://127.0.0.1:8080
serve\start-qwen.cmd -Mode code          # MTP d1 (+p-min 0.5) for code / JSON / copy-heavy output  (no longer faster than chat on the s14 build: prefer chat)
serve\health.ps1                          # status, mode, lock owner, 1-token round trip
serve\stop-qwen.cmd                       # stops the server, releases the NPU lock
```

`start-qwen` queues for the NPU lock (`npu_campaign\.npu.lock`, same protocol as `lockrun.py`/`npu-lock.ps1`), then loads the model (about 1-3 min), and returns when `/health` is ok.
A hidden `serveockhold.py` process (lockrun.py protocol, 20 ms polling) holds the lock for the whole life of the server (its pid is in the lock's `owner.txt`, label "decode2 lockrun" = lockrun.py);
it releases when the server exits or `stop-qwen` runs. **While the server is up nobody else can benchmark on the NPU** - stop it when you are not using it. `-Foreground` runs in the console (Ctrl-C stops).

Useful options: `-Port`, `-Ctx 32768`, `-Temp 0.6 -TopP 0.95 -TopK 20`, `-Reasoning auto|on|off`, `-ExtraArgs "..."`, `-B 2048 -Ub 1024`.
Logs: `serve\logs\server-<mode>.log(.out/.err)`. Switching mode = `stop-qwen` then `start-qwen -Mode ...` (about 2 min).

## Modes

| mode | model file | speculation | use |
| --- | --- | --- | --- |
| chat | `Qwen_Qwen3.6-35B-A3B-Q4_0.gguf` | none | prose, chat, agents (default) |
| code | `...-Q4_0-mtp-f32.gguf` | `--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-p-min 0.5`, `LLAMA_MTP_FUSED_CATCHUP=1`, `LLAMA_MTP_DRAFT_VOCAB=65536` | code, JSON/tool calls, copy-heavy |
| both | = code | | see below |

**No per-request speculation toggle**: in this llama-server, speculation is a server-level object (`slot.can_speculate()` is `!!spec`); request parameters cannot switch it. So `-Mode both` is just `code`.
MTP at chat-prose acceptance (~70%) is slower than plain (32.6 vs 34.5 tok/s), hence the two modes.

Common flags: `--device HTP0 -ngl 99 -fit off --load-mode none -b 2048 -ub 1024 -t 4 -np 1 --cache-prompt --metrics --jinja --reasoning auto --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 -cram 2048`, plus the frozen decode env
(`GGML_HEXAGON_OPBATCH=3072 REPLAY=1 OPFUSION=1 MM_SELECT=2 FA_SELECT=1 GDN_SELECT=1 ROUTER_HVX=1 ...`, copied from `npu_campaign\run-mtp-speed.ps1`). `GGML_HEXAGON_OPPOLL` is deliberately never set.

## Expected speeds (installed build, AC power; from OVERNIGHT_SUMMARY / MTP3_LOG, 256-token greedy)

- **Installed build 2026-10-01 (qwen4 s14, padded Q4_0 tiles + contiguous weight DMA): plain decode ~40 tok/s (Hello 40.2, code 40.2), prefill pp1024 ~780 tok/s (ub 1024; prefill not re-measured on s14). With s14 plain decode matches or beats MTP everywhere (MTP d1: Hello 35.2, code 40.2, p-min 0.5 code 39.8): USE `-Mode chat` FOR EVERYTHING; `-Mode code` is kept but no longer pays.**
- Older numbers below (36 tok/s plain, MTP +0-8%) are from the previous build (pkg-final2-f1).
- (previous build) plain decode ~36 tok/s (34-36 depending on prompt), prefill pp1024 ~780 tok/s (ub 1024).
- MTP d1: code 37.4 (plain 34.4), +p-min 0.5 37.3; JSON tool call 34.7 -> 35.4 with p-min 0.5 (plain 34.2); copy-heavy 34.2 -> 35.8 (plain 34.0). Gain is 0-8%: use it only for code-like output.
- Server-measured numbers from this setup: see "Measured" at the bottom.

## One port for everything: the decisions shim

```
python serve\decisions_shim.py --listen-port 8090 --upstream http://127.0.0.1:8080
```
Point PI / OpenAI clients at `http://127.0.0.1:8090/v1` (all paths except `/v1/decisions` are passed through, streaming included).
`POST /v1/decisions` follows SGLang's schema (typed questions: `choice` with options -> labels A..Z, `score` with levels -> 0..9, `yes_no`): one prefill, next-token probabilities of the labels, normalised over the labels
(`probabilities`), `label_mass` = full-vocab mass of the labels, `choice`, `score` (probability-weighted mean level index), `usage.completion_tokens = 0`. Extras: `option_probabilities`, `latency_ms`, `timings`.
Several questions about the same `input` share the prefix `[system + input]`, so only the question suffix is prefilled when llama-server's prompt cache can reuse it (see limits).

```
curl http://127.0.0.1:8090/v1/decisions -d '{"input":"Integration keeps failing for 3 days.","questions":[
  {"id":"team","type":"choice","question":"Which team should handle this?","options":[{"name":"billing"},{"name":"technical"}]},
  {"id":"urgent","type":"yes_no","question":"Does the customer need an answer today?"}]}'
```
Tests: `python serve\test_decisions.py` (against a running server).

## Pointing PI (pi.dev)

Working config: `npu_campaign\pitest\agentdir\models.json` (provider `local`, `api: openai-completions`, `baseUrl http://127.0.0.1:<port>/v1`, `apiKey none`, compat `supportsDeveloperRole:false, supportsReasoningEffort:false, supportsUsageInStreaming:true, maxTokensField:max_tokens`,
model id `local-model`; `contextWindow` there is 16384 for the suite, raise it to 32768 for this server) and `settings.json` (`defaultProvider local`, `defaultThinkingLevel off`). Copy both into `~\.pi\agent\` (or set `PI_CODING_AGENT_DIR`) and change the port to 8080 (or 8090 via the shim).
llama-server ignores the model id (alias `qwen36-35b-a3b`).

## Memory budget (48 GB machine)

Model ~19.4 GiB resident. `-c 32768`: only 10 of 40 layers are full attention (small KV, well under 1 GiB at f16) plus a fixed recurrent state, so 32K costs little (measured below). Prompt cache capped with `-cram 2048`.
**One model process at a time**: a second llama process (even a hung llama-cli) can leave ~0 GB free; check `Get-Process llama-*` before starting. Other campaign agents benchmark on the same NPU; the lock queues you behind them.

## Troubleshooting

- `start-qwen` waits forever: someone holds the lock (`type npu_campaign\.npu.lock\owner.txt`); a dead owner's lock is taken over after 2 min.
- Server dies at load / extreme slowness: not enough free RAM (another llama process alive). Look at `serve\logs\server-*.log.err`.
- Stuck NPU process at 100% CPU: that is the `OPPOLL=1` hang; never set it. Stop with `stop-qwen`.
- Garbage text: you are running Gemma 4 on the NPU (known broken; use Qwen only).
- A mode switch needs stop + start. Lock left behind after a crash: `stop-qwen` cleans a stale `serve-qwen` lock.

## Known limits

- Gemma 4 gives garbage on the NPU; `GGML_HEXAGON_OPPOLL=1` can hang (holds the NPU); Q4_0 MTP head (headq40 gguf) costs +1% PPL, not used.
- Single slot (`-np 1`): requests are serialised, a second client waits.
- Qwen3.6 is a hybrid (recurrent) model: llama-server can only reuse a cached prefix at a checkpoint position, so the decisions shim's prefix reuse may degrade to a full re-prefill (~780 tok/s) in some cases.
- Server holds the NPU for its whole life; do not leave it running unattended if other agents need the NPU.

## Measured (2026-10-01 00:00-00:16, installed build pkg-wos, AC, -c 32768 -b 2048 -ub 1024)

- chat mode: decode 36.0 tok/s (200-token story); RAM available fell from ~38 GB to ~17 GB with the server loaded (about 21 GB, ctx 32768 included).
- code mode (MTP d1, p-min 0.5): 37.3 tok/s on a Python-function prompt (acceptance 124/129), 37.2 on a JSON prompt (54/55). Short chat-style prompts see ~110 tok/s prefill (small batches); long prompts reach ~780 tok/s.
- `smoke.ps1`: create_file, read_answer, shell_cmd PASS via PI; server metrics: 223 tok/s average prefill over the agent turns (many short and cached turns), 33.5 tok/s average decode.
- decisions shim (`test_decisions.py`, 12/12 checks): yes/no, 3-way choice and 5-level score examples all correct; probabilities sum to 1; label_mass 0.96-0.9997; reversing the option order keeps the choice (technical 0.99996 vs 0.99995). Latency per question 0.45-0.7 s (100-145 prompt tokens, one prefill, 0 generated); a 3-question call took 2.0 s wall.
  `cache_n` was 0 for every question: llama-server could not reuse the shared prefix (hybrid recurrent model), so each question re-prefills the whole prompt. Keep inputs short or batch questions about long inputs knowing the full input is re-prefilled per question (~780 tok/s on long inputs).
- Server stop: the llama-server process lingers ~15 s after exit while memory is torn down; `stop-qwen` and the supervisor wait for that before releasing the NPU lock.

## Context size and the prefill-speed gotcha (2026-10-01)

- Server context is `-c 32768` (PI's `contextWindow` in `~\.pigent\models.json` must match). The model's native maximum is 262144. Only 10 of its 41 layers use full attention (2 KV heads, head dim 256), so the f16 KV cache costs ~20 KB/token:
  32768 -> 0.6 GiB, 131072 -> 2.5 GiB, 262144 -> 5.0 GiB (plus a small fixed recurrent state). RAM is not the limit (model 19.4 GiB; ~37 GB free on an idle machine). Untested on the NPU: speed and correctness of attention at very long depth
  (measured so far: decode 40.3 tok/s at depth 0, 35.9 tok/s at depth 2048, prefill 774 -> 745 t/s). Use `-Ctx 131072` and validate with a long prompt before trusting more.
- **`GGML_HEXAGON_FA_SELECT` / `GDN_SELECT` must be 2 for a server** (set in start-qwen.ps1 since 2026-10-01). The decode-tuned value 1/1 (copied from the MTP benchmark script) made prompt processing ~20 tok/s (a 1623-token PI prompt took 82 s);
  with 2/2 the same kind of prompt runs at ~700 tok/s and decode stays 40 tok/s.
- PI is installed under Node LTS (`npm -g @earendil-works/pi-coding-agent`, 1.0.0) from 2026-10-01; the bun copy and the old standalone shim (`~\.localin\pi.cmd.bak`, pointing at pi-v0.85.1) were retired. Update with `npm update -g @earendil-works/pi-coding-agent`.

### Update 2026-10-01: default context is now 262144 (the model's native maximum)
`start-qwen.ps1 -Ctx` defaults to 262144; PI's `hexagon-npu` entry has `contextWindow 262144` and `maxTokens 65536` (the earlier 4096 cap truncated long file writes; the server itself has no output cap, `n_predict = -1`).
Validated with `serve\long_ctx_test.py <port> <tokens>` (needle-in-haystack: a secret code planted at the start of a long prompt, asked at the end), Qwen3.6-35B-A3B Q4_0 on HTP0, plain decode, ub1024, AC:
| prompt tokens | needle | prefill | decode |
| ---: | --- | ---: | ---: |
| ~1.4K | n/a | 700 t/s | 40.0 tok/s |
| 30.5K | found | 644 t/s | 24.8 tok/s |
| 125.8K | found | 416 t/s | 12.8 tok/s |
RAM available after loading at 262144: ~10.4 GB of 47.6 (KV cache is allocated up front: 5 GiB). A full 250K prompt was not run (about 10+ minutes of prefill, decode ~7 tok/s expected); deep contexts are usable but slow, mostly because attention cost grows with depth.
