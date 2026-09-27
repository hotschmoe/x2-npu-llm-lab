# Campaign status

Updated: 2026-09-26 19:13 local. The Qwen Q4_0 GGUF on disk has had its 60 Q5_0 shared-expert tensors requantized to Q5_K in place. Restore from `npu_campaign/q5_0_shexp_backup.bin` with `requant-shexp.exe --restore` before treating that file as the original quant. Best measured decode is still 28.4 tok/s over 128 tokens: 28.38 and 28.40 (`htp-qwen-replay.log`, `htp-qwen-replay2.log`), both the same PHP text as the 27.6 runs. The GDN state-gather fuse did run (1391 ops, 28.59 tok/s, `htp-qwen-baseline2.log`) and printed the aws-cdk fragment, so the matcher is not called. Do not enable it. The expert-weight mul is now folded into `ADD+NW` and the text is still PHP (`htp-qwen-addnw.log`, 28.37 tok/s, 35.24 ms, 1381 ops). That is not under the 35.2 ms pair, so 28.4 stands. `SHEXP+EP` (SwiGLU + shared-expert down + the f32 gate dot folded into the sigmoid epilogue) printed a different INSERT (`htp-qwen-shexpep.log`, email column, 28.93 tok/s, 34.57 ms, 1261 ops) and is not called. The installed skel is the restored PHP file, `pkg-wos/lib/libggml-htp-v81.so`, 844176 bytes, with the catalog that verifies it (`npu_campaign/libggml-htp-php-844176.cat`). Confirm run `htp-qwen-shexpep-off.log` is PHP through "You have an", 28.62 tok/s, 34.94 ms, 1381 ops. One run, not a pair. 28.4 stands. The v81 prefix so is the unused 848432-byte `SHEXP+EP` link. The previous PHP skel is still `npu_campaign/libggml-htp-v81-php-839952.so` (839952 bytes). Fuse-off rerun `htp-qwen-gdn-off.log` is PHP at 27.76 tok/s and does not replace 28.4. With f16 KV and a prefix already cached, tg64 is 28.34 at depth 0, 26.41 at 2048, and 25.11 at 8192 (`bench-depth.log`). Graph replay (`GGML_HEXAGON_REPLAY`, default on) submits the captured single-flush command again. The previous 27.6 pair was 36.21 and 36.36 ms; this pair is 35.23 and 35.21 ms. Router top-k is now fused (`MOE+TOPK`) and the 32-token text is still the PHP PDO fragment: 27.23 tok/s, `htp-qwen-topk.log`. Two 128-token repeats are 27.71 and 27.57 (`htp-qwen-topk-n128.log`, `htp-qwen-topk-n128b.log`). They overlap the 27.6 band, so they are not a new best. The 26.92 tok/s figure is the best 32-token average from before the router-norm fuse and includes two cold graphs. The router-norm fuse is on and still prints PHP: 26.60, 27.86, and 26.37 tok/s (`htp-qwen-wnorm.log`, `htp-qwen-wnorm2.log`, `htp-qwen-wnorm3.log`). 27.86 did not repeat. The RMS matcher stays off. The repro pair with that matcher off was 26.88 and 26.68 (`htp-qwen-repro2.log`, `htp-qwen-repro.log`). The ADD+N number was 26.41, log `htp-qwen-addn2.log`. The conv-window number was 25.79, log `htp-qwen-convwin-final.log`. The embedding is on HTP0 (`splits = 1`). MTP verify is slower than plain decode: 18.94 tok/s over 64 tokens, acceptance 0.75, log `htp-qwen-mtp3.log.err`. `LLAMA_FUSE_MTP_VERIFY=1` appended blk.40 attention to the target graph and aliased that KV into the draft context. Two temp-0 Hello runs disagreed: `htp-qwen-mtp-fuse.log` keeps "a question" at 22.2 tok/s, `htp-qwen-mtp-fuse2.log` says "questions" at 21.3 tok/s. Leave the variable unset. Flag-off PHP is `htp-qwen-fuse-off.log`, "You have an", 28.56 tok/s, 35.01 ms, 1381 ops. One run. 28.4 stands.

## Step 1 — signed Hexagon build

- State: done (`BUILD_EXIT=0`)
- Source: `llm_inference/llama.cpp-hexagon` at `08618ff`. August tree `llm_inference/llama.cpp` was not modified.
- Install: `llama.cpp-hexagon/pkg-wos`
- `pkg-wos/bin/ggml-opencl.dll` is the zero-device stub. The real DLL is `npu_campaign/disabled-for-smoke/ggml-opencl-real.dll`. The Adreno driver fast-fails in `CB.dll` if the real DLL loads.

## Step 1b — HTP0 smoke

- State: done
- Q4_0 4096×14336 GEMV, one op: 547 µs, about 60.5 GB/s
- Llama 3.2 1B Q4_0, 17/17 layers on HTP0: decode 108.68 tok/s (9 tokens, 9.20 ms/token). Log: `npu_campaign/htp-llama1b-completion.log`
- Smoke model: `models/gguf/llama32-1b-instruct-q4_0/Llama-3.2-1B-Instruct-Q4_0.gguf` (773025920 bytes)

## Step 2 — Qwen3.6 Q4_0 trace

- Profile-1 log: `npu_campaign/htp-qwen-profile1.log`. 42/42 layers on HTP0, 18720 MiB. Prompt 43.0 tok/s over 9 tokens.
- Decode: `npu_campaign/htp-qwen-decode.log`. 32 tokens, 59.43 ms/token, **16.83 tok/s**, graphs reused 30/32. No second copy is running.

## Step 3 — fewer batch flushes

- Gap is measured. Batch-1 device time is 44.1 ms across 21 OPBATCHes. Wall decode is 59.43 ms. The 15 ms is the host round-trip at each `fit_op` vmem flush. See FINDINGS.md 08:50.
- 8 sessions failed: the driver opened four (session ids 0–3) and `FASTRPC_RESERVE_NEW_SESSION` for the fifth returned `0x200`. Log `htp-qwen-ndev8.log`.
- 4 sessions finished slower: 14.66 tok/s, 68.20 ms/token. Log `htp-qwen-ndev4.log`. Each session buffer is still larger than the 3158 MiB map.
- 64-buffer host cap is in and measured. One session, 32 tokens: **20.37 tok/s**, 49.10 ms/token. Log `htp-qwen-bufs64.log`. No flush-limit line fired. Do not restore `pkg-wos/bin/ggml-opencl.dll` from the build tree. The stub must stay, or the process fast-fails in `CB.dll`.

## Step 4 — the 42 scheduler splits

- State: removed, except the embedding
- The 60 Q5_0 shared-expert weights were requantized in place to Q5_K (same byte size, already a Hexagon matmul). No DSP rebuild. Backup `q5_0_shexp_backup.bin`.
- Log `htp-qwen-q5k-b128.log`. CPU model buffer 272.81 MiB. HTP0 model buffer 18765.11 MiB. **splits = 2**. The remaining CPU split is `token_embd` `GET_ROWS`.
- Default op cap 1280, one flush per token, buffers 17/64: **22.98 tok/s**, 43.52 ms/token.
- `GGML_HEXAGON_OPBATCH=3072` (env only, source default stays 1280). No flush. Log `htp-qwen-opbatch3072.log`: **23.47 tok/s**, 42.60 ms/token. Graphs reused 30/32.
- `pkg-wos/bin/ggml-opencl.dll` is still the stub. `HTP_OP_MAX_BUFS` is still 64.

## Step 5 — warm batch-1 profile at OPBATCH 3072

- State: done. Log `htp-qwen-profile1-warm.log.err`. `-n 4 -lv 5`, so the printed 245 ms/token is the debug log, not a new rate.
- The reused token is one OPBATCH, 1923 ops, **38.00 ms** on the DSP. Op sum 36.90 ms. Queue inside the batch 1.10 ms. No flush.
- 42.60 ms wall − 38.00 ms device = 4.60 ms host (embedding `GET_ROWS` plus one submit). See FINDINGS.md 10:04.
- The output projection is 3.00 ms (139 GB/s on 417 MB). Routed expert GEMVs are 7.13 ms at 73–84 GB/s. Neither is the bulk.
- 1231 ops under 8 µs add to 6.67 ms. The concentrated case is the GDN conv-state `CONCAT`+`CPY`: 4.10 ms for 6.7 MB, about 1.6 GB/s. The 2 MB recurrent state read is fine (77 GB/s).
- Q5_K shared experts are slightly faster than the Q8_0 ones (28/14 µs vs 31/16). Do not requant them again.

## Step 6 — fuse the conv-state window

- State: measured. `GGML_HEXAGON_FUSE_SSM_CONV_WIN` folds the 30-layer `CONCAT`+`CPY`+`SSM_CONV` into one `SSM_CONV`. The 4-wide window is not stored.
- Scalar gathers in VTCM are exact and slower: 23.09 tok/s, log `htp-qwen-convwin-scalar.log`.
- The installed kernel HVX-copies each 32-channel block to cached scratch, gathers there, and stores the window back to VTCM for the existing mac. Log `htp-qwen-convwin-final.log`: **25.79 tok/s**, 38.78 ms/token. Graphs reused 30/32. Greedy text matches the 23.47 run.
- Embedding-on-HTP first aborted (`htp-qwen-embd-q4.log`, `NO-SUPPORT` on the above-4GB index). The skel now DMAs that index into VTCM. Log `htp-qwen-embd-idx.log`: HTP buffer 19037.92 MiB, no CPU model buffer, **splits = 1**, 25.74 tok/s, 38.85 ms/token, PHP text matches. Not faster than 25.79. Leave `dev_input` on HTP0.
- A repeat of that gather was 26.10 tok/s (`htp-qwen-convwin-bounce3.log`). Same text. Treat 25.79 as the number for this skel.
- `pkg-wos/bin/ggml-opencl.dll` is still the stub. `HTP_OP_MAX_BUFS` is still 64. `cmake --install` does not publish the new `libggml-htp-v81.so`; it was copied from the v81 prefix build and the catalog was re-signed. Install also restores the real OpenCL DLL.

## Step 7 — fuse the expert ADD chain

- State: measured. `GGML_HEXAGON_FUSE_MOE_ADD` replaces each layer's 7 expert-row ADDs with one `ADD+N`. The 8 packed f32 rows are one VTCM left-fold. The running sum is not stored.
- Log `htp-qwen-addn.log` (`-lv 4`): splits = 1, HTP buffer 19037.92 MiB, n-ops 3072, no flush, **26.20 tok/s**, 38.17 ms/token. Graphs reused 30/32. The fuse line is `n_src 8 n_elem 2048`.
- Log `htp-qwen-addn2.log`: **26.41 tok/s**, 37.87 ms/token. Graphs reused 30/32. Greedy text matches the PHP PDO fragment on both runs.
- 38.78 − 37.87 = 0.91 ms on the faster run. The pair is about 0.6–0.9 ms under the 25.79 wall, roughly half of the old 7-ADD chain, not the rest of the short ops.
- `pkg-wos/bin/ggml-opencl.dll` is still the stub. `HTP_OP_MAX_BUFS` is still 64. The Q5_0 backup was not restored. The conv-window fuse and the Q4_0 extended-index path were not reverted.

## Step 8 — fuse the shared-expert gate

- State: measured. `GGML_HEXAGON_FUSE_SHEXP_GATE` replaces sigmoid(gate) * shexp + moe + residual with one `ADD+SG`. The scalar gate matmul stays. Same HVX ops, same order. The running products are not stored.
- No new profile. The 10:04 token puts that chain at 887 µs (5 + 6 + 5.6 + 5.4 µs, ×40).
- `-n 1 -lv 4` log `htp-qwen-sg-lv4.log.err` prints conv window, ADD+N, and `ADD+SG (n_elem 2048)`. The 42.16 ms there is one cold token.
- Log `htp-qwen-sg.log`: **26.83 tok/s**, 37.27 ms/token. Graphs reused 30/32.
- Log `htp-qwen-sg2.log`: **26.92 tok/s**, 37.14 ms/token. Graphs reused 30/32. Greedy text matches the PHP PDO fragment on both.
- 37.87 − 37.14 = 0.73 ms on the faster run. The pair is 0.60–0.73 ms under the 26.41 wall, most of the 0.89 ms chain.
- `pkg-wos/bin/ggml-opencl.dll` is still the stub. `HTP_OP_MAX_BUFS` is still 64. The Q5_0 backup was not restored. The conv-window fuse, the ADD+N fuse, and the Q4_0 extended-index path were not reverted.

## Step 9 — rebuilt skel, matcher off, PHP is back

- State: checked. No source change. The Q4_0 `GET_ROWS` dequant is the ggml nibble order (`(q & 0x0F) - 8`, high nibble in the upper half) over 18-byte blocks. `sizeof(block_q4_0)` is asserted. fp16 to fp32 is the exact software conversion. That path was already in the 26.92 skel. It is not what flips greedy tokens.
- The RMS matcher is still disabled. `try_fuse` returns before `try_fuse_rms_norm_scale`. Do not enable it. The only run that fused it (`htp-qwen-rms.log`, scale 0.0883883) printed the React Native fragment, 26.52 tok/s.
- Same installed skel (`pkg-wos/lib/libggml-htp-v81.so`, 835400 bytes, 14:13) and the same host, matcher off, canonical flags (`GGML_HEXAGON_OPBATCH=3072`, `-b 128 -ub 128 -p Hello --temp 0`):
  - `htp-qwen-repro.log`: PHP PDO fragment, seed 2818050408, **26.68 tok/s**, 37.49 ms/token. Graphs reused 30/32. Fuses logged: conv window, ADD+N, ADD+SG. No RMS line. No flush. splits = 1.
  - `htp-qwen-repro2.log`: same PHP text, seed 2502661726, **26.88 tok/s**, 37.20 ms/token.
- Default `-b`/`-ub` (context batch clamped to 256, generate log still says 2048) is also PHP. `htp-qwen-bdefault.log`: **26.26 tok/s**, 38.08 ms/token. Slower, so keep `-b 128 -ub 128`.
- `htp-qwen-rms-off.log` (aws-cdk, generate n_batch 2048, 26.94 tok/s) was not reproduced. The dequant formula does not explain it. Do not treat 26.94 as the best.
- `pkg-wos/bin/ggml-opencl.dll` is still the stub (102400 bytes). `HTP_OP_MAX_BUFS` is still 64. The Q5_0 backup was not restored. ADD+SG, ADD+N, and the conv-window fuse were not reverted.

## Step 10 — fuse the router weight norm

- State: measured. `GGML_HEXAGON_FUSE_MOE_WNORM` replaces each layer's `SUM_ROWS`+`CLAMP`+`DIV` of the 8 selected expert weights with one `MOE+WNORM`. The sum is not stored. Same HVX sum, same HVX clamp, then the same reciprocal multiply the f32 scalar DIV uses.
- The 10:04 token spends about 11 µs on that chain, ~0.44 ms across 40 layers. One launch instead of three is about 0.3 ms if the launch floor is the whole cost.
- `-n 1 -lv 4` log `htp-qwen-wnorm-lv4.log.err` prints conv window, `MOE+WNORM (n_elem 8)`, ADD+N, and ADD+SG. No RMS line. splits = 1. HTP buffer 19037.92 MiB. n-ops 3072. No flush.
- Same 32-token command (`GGML_HEXAGON_OPBATCH=3072`, profile off, `-n 32 -b 128 -ub 128 -p Hello --temp 0 --device HTP0 -ngl 99 -fit off --load-mode none`):
  - `htp-qwen-wnorm.log`: PHP PDO fragment, seed 3922511455, **26.60 tok/s**, 37.59 ms/token. Graphs reused 30/32.
  - `htp-qwen-wnorm2.log`: same PHP text, seed 1639016804, **27.86 tok/s**, 35.90 ms/token. Graphs reused 30/32.
  - `htp-qwen-wnorm3.log`: same PHP text, seed 2551100566, **26.37 tok/s**, 37.92 ms/token. Graphs reused 30/32.
- Two of the three sit on the 26.6 repro band. 27.86 did not repeat. Do not replace 26.92. The matcher stays on because the text is PHP. The expert-weight MUL was not fused.
- Installed skel `pkg-wos/lib/libggml-htp-v81.so` is 835496 bytes at this step, copied from the v81 prefix build. The copy `build-wos/ggml/src/ggml-hexagon/libggml-htp-v81.so` is still the stale 835400-byte file. The catalog in `pkg-wos/lib` was signed after the prefix copy. `pkg-wos/bin/ggml-opencl.dll` is the stub, 102400 bytes. `HTP_OP_MAX_BUFS` is still 64. The Q5_0 backup was not restored. ADD+SG, ADD+N, the conv-window fuse, and the Q4_0 extended-index path were not reverted. `try_fuse_rms_norm_scale` is still not called.

## Step 11 — fuse the router top-k

- State: measured. `HTP_OP_MOE_TOPK` replaces softmax + desc argsort + top-8 gather + the weight norm on a 256-wide decode row. The probs and the unnormalized weights are not stored. The full index row is stored. A row that is not 256 wide keeps `MOE+WNORM`.
- `-n 1 -lv 4` log `htp-qwen-topk-lv4.log.err` prints conv window, `MOE+WNORM`, ADD+N, ADD+SG, and `MOE+TOPK (n_exp 256 k 8)`. One chain logged `moe-topk skip: logits`. No RMS line.
- Log `htp-qwen-topk.log`: PHP PDO fragment, **27.23 tok/s**, 36.72 ms/token. Graphs reused 30/32.
- Log `htp-qwen-topk-n128.log`: **27.71 tok/s**, 36.09 ms/token. Graphs reused 126/128. PHP through the INSERT, then "However".
- Log `htp-qwen-topk-n128b.log`: **27.57 tok/s**, 36.27 ms/token. Graphs reused 126/128. Same long text as `htp-qwen-n128.log` through "You have an".
- 36.09 and 36.27 overlap the earlier 36.21 and 36.36. Not a new best. The fuse stays on. The expert-weight MUL was not fused.
- Installed skel `pkg-wos/lib/libggml-htp-v81.so` is 839952 bytes, the v81 prefix build. The catalog was signed after that copy. `pkg-wos/bin/ggml-opencl.dll` is the stub, 102400 bytes. `HTP_OP_MAX_BUFS` is still 64. The Q5_0 backup was not restored. ADD+SG, ADD+N, the conv-window fuse, MOE+WNORM, and the Q4_0 extended-index path were not reverted. `try_fuse_rms_norm_scale` is still not called.

## Step 12 — GDN state gather, disabled

- State: measured, off. `try_fuse_gdn_state` replaces the 30 cache `GET_ROWS` with a row stride in `op_params[3]` and an i32 index on `src[6]`. The steady graph went from 1421 ops to 1391.
- Log `htp-qwen-baseline2.log`: **28.59 tok/s**, 34.98 ms/token, graphs reused 126/128, replay hit. Text is the aws-cdk VPC fragment, not PHP. Seed 1623861709.
- The matcher is not called. Do not enable `GGML_HEXAGON_FUSE_GDN_STATE`. The RMS matcher stays off.
- Installed skel is the previous PHP file, `pkg-wos/lib/libggml-htp-v81.so`, 839952 bytes. The catalog was signed after that copy. `pkg-wos/bin/ggml-opencl.dll` is the stub, 102400 bytes.
- Log `htp-qwen-gdn-off.log`: PHP PDO through "You have an", seed 2334271453, **27.76 tok/s**, 36.02 ms/token, replay hit 1421 ops. Not a new best. 28.4 stands.

## Step 13 — expert weight mul into the add

- State: measured, on. `HTP_OP_ADD_NW` scales the eight packed `ffn_moe_down` rows with `hvx_mul_scalar_f32_aa`, then left-folds them with `hvx_add_f32_aaa`. The weighted rows are not stored. The 17:40 token spends 278 µs on that mul.
- Log `htp-qwen-addnw.log`: PHP PDO through "You have an", **28.37 tok/s**, 35.24 ms/token, graphs reused 126/128, replay hit 1381 ops. The fuse line is `ADD+NW (n_src 8 n_elem 2048)`. No skip line. No GDN-state line. No RMS line.
- 35.24 does not beat 35.21 and 35.23. Not a pair. 28.4 stands. The fuse stays on because the text is PHP.
- Installed skel `pkg-wos/lib/libggml-htp-v81.so` is 844176 bytes, copied from the v81 prefix build. The catalog was signed after that copy. `build-wos/ggml/src/ggml-hexagon/libggml-htp-v81.so` is still the stale 839952-byte file. Previous PHP skel: `npu_campaign/libggml-htp-v81-php-839952.so`.
- `pkg-wos/bin/ggml-opencl.dll` is the stub, 102400 bytes. `HTP_OP_MAX_BUFS` is 64. The Q5_0 backup was not restored. `try_fuse_rms_norm_scale` is still not called. `try_fuse_gdn_state` is still not called.

## Step 14 — shared-expert down epilogue, disabled

- State: measured, off. `HTP_OP_SHEXP_EP` replaces SwiGLU, the shared-expert down `MUL_MAT`, and the f32 gate dot with one op. The down GEMV is the existing `op_matmul` (same kernel params). The gate dot is `vec_dot_f32_f32_aa_1x1`. The sigmoid-scale-add is the `ADD+SG` sequence. The 17:40 token spends 0.66 ms on the down GEMV, 0.25 ms on the SwiGLU, and 0.27 ms on the gate dot.
- Log `htp-qwen-shexpep.log`: the fuse line fired, `SHEXP+EP (k 512 n 2048)`. Steady graph 1261 ops (1381 minus the 40 SwiGLUs, 40 downs, and 40 gate dots). Replay hit. Graphs reused 126/128. Seed 2302241197. **28.93 tok/s**, 34.57 ms/token.
- The text is still a PHP PDO fragment, but the INSERT is `username, password, email`, and it does not reach "You have an". Not the addnw text.
- The matcher is not called. Do not enable it. The installed skel is the 844176-byte PHP file again. Catalog `npu_campaign/libggml-htp-php-844176.cat` verifies that so. A copy of this skel is `npu_campaign/libggml-htp-v81-php-844176.so`.
- Log `htp-qwen-shexpep-off.log`: PHP PDO through "You have an", no email column, no `SHEXP+EP` line, `ADD+SG` still logged. Seed 1241911316. **28.62 tok/s**, 34.94 ms/token, replay hit 1381 ops, graphs reused 126/128. One run. Not a new pair. 28.4 stands.
- `pkg-wos/bin/ggml-opencl.dll` is the stub, 102400 bytes. `HTP_OP_MAX_BUFS` is 64. The Q5_0 backup was not restored. `try_fuse_rms_norm_scale` is still not called. `try_fuse_gdn_state` is still not called. `absorb_shexp_down` is still not called.

## Still open

- Moving the embedding onto HTP removed the CPU split and did not change the wall by itself. The remaining host time is the one sched copy (logits) plus the submit, not `GET_ROWS`. Do not turn profile on for a rate run.
- The 0.96 ms GDN state `GET_ROWS` is still in the token. Fusing it changed the greedy text. Do not turn that matcher back on without a PHP match.
- Router norm, router top-k, and the expert-weight mul are fused and did not move the 28.4 tok/s wall. The shared-expert down epilogue (`SHEXP+EP`) is faster and changes the INSERT. Do not call it. Still open from the 17:40 token: the q/k `RMS_NORM`+`SCALE` pairs (634 µs, matcher stays off) and the 2.96 ms output projection. Do not requant `output.weight`. The sub-8 µs remainder is about 4 ms and is scattered.
- MTP accepts drafts (0.75, mean length 1.75) but a 2-token verify costs about two single-token forwards. Plain decode is faster. Log `htp-qwen-mtp3.log.err`, 18.94 tok/s. `LLAMA_FUSE_MTP_VERIFY=1` drifted (see Step 15). Leave it unset.

## Step 15 — fused MTP verify tail, off

- State: measured, off. `LLAMA_FUSE_MTP_VERIFY=1` appends blk.40 attention to the target graph and aliases that layer's KV into the draft context. The catch-up `llama_decode` is skipped because `ctx_other` is set. The MoE and the MTP LM head stay on `draft()`.
- `htp-qwen-mtp-fuse.log`: temp-0 Hello keeps "a question". Banner Generation 22.2 t/s.
- `htp-qwen-mtp-fuse2.log`: the same command says "questions". Banner Generation 21.3 t/s.
- The pair does not agree. Do not set the variable. `run-mtp-fuse.cmd` is the reproducer.
- Flag off, log `htp-qwen-fuse-off.log`: PHP through "You have an", replay hit 1381 ops, 28.56 tok/s, 35.01 ms. One run. 28.4 stands. Installed skel remains 844176 bytes. `absorb_shexp_down` is not called.
- The token embedding is on HTP0. `splits = 1`. One sched copy remains (logits).
- DMA skel open returns `512` (`AEE_ERPC`) once the library path is visible. No DSP GB/s yet.
