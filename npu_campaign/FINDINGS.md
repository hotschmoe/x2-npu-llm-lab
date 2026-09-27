# NPU campaign findings

Machine: ASUS Zenbook A16, Snapdragon X2 Elite Extreme, Hexagon v81, one NSP, 8 MB VTCM. Goal file: `../NPU_MOE_20TPS.md`.

## 2026-09-26 00:45 — source and runtime contract

The Hexagon build is from llama.cpp `08618ff` (2026-09-26), not the local August tree. That tip contains `src/models/qwen35moe.cpp`, Q4_K / Q5_K matmul, MXFP4, and the HMX gated-delta path. The August tree's Hexagon ops stop at Q4_0, Q4_1, Q8_0, IQ4_NL, and MXFP4.

Upstream README (`docs/backend/snapdragon/README.md` at this commit) changes the session plan:

- One session still has about 3.5 GB of virtual address space.
- Models larger than that can stay on one session. The backend maps and unmaps weight buffers during the graph. That remap is overhead, and it is the first thing to measure on the 20 GB Q4_0 file.
- `GGML_HEXAGON_DEVICES=HTP0:0,HTP0:1,...` creates virtual sessions on one physical NPU so each session keeps its weights mapped. The GPT-OSS-20B example uses four virtual sessions. `GGML_HEXAGON_NDEV` is the deprecated spelling of the same idea.
- Device selection on the command line is `--device HTP0` plus `-ngl 99`. `ADSP_LIBRARY_PATH` must contain both `libggml-htp-v81.so` and `libggml-htp.cat`.
- `GGML_HEXAGON_DMA64` defaults on for v81.
- `GGML_HEXAGON_PROFILE=1` prints `profile-op` with usec and cycles. `=2` adds eight PMU counters. `=3` is still in `ggml-hexagon.cpp` even though the README stops at mode 2: it prints `trace-evt` start/stop lines (DMA, HVX, HMX, GDN stages) with cycle stamps. Implied GB/s still has to be computed from tensor bytes and those times. The log line is `GGML_LOG_DEBUG`, so the process needs a debug log level or the backend's own verbose switch before those lines appear. Confirm that on the first smoke.

## 2026-09-26 00:55 — HTP0 is open and a Q4_0 GEMV ran

Signed build of llama.cpp `08618ff` finished (`BUILD_EXIT=0`). `pkg-wos/lib` holds `libggml-htp-v81.so` (818248 bytes) and `libggml-htp.cat`, signed by `CN=GGML.HTP.v1`. `signtool verify /pa` accepts the catalog.

`test-backend-ops` could not start with the real `ggml-opencl.dll`. The process died `0xC0000409` (the same Adreno `CB.dll` fast-fail as the September bandwidth probes) while the OpenCL backend probed devices, before Hexagon ran. `ggml.dll` imports that DLL, so it cannot be deleted. `npu_campaign/opencl-stub.c` is a registry that reports zero OpenCL devices. The real DLL is saved at `npu_campaign/disabled-for-smoke/ggml-opencl-real.dll` and the stub is what `pkg-wos/bin/ggml-opencl.dll` is right now.

With `ADSP_LIBRARY_PATH` pointed at `pkg-wos\lib` (trailing semicolon) the session came up:

- Hexagon arch v81, DMA64 on
- `FASTRPC_GET_DOMAINS` returned `0x6c`; the backend fell back to a static CDSP domain and that fallback worked
- HTP0: 8 threads, 8 HVX, 1 HMX, 8 MB VTCM
- URI `file:///libggml-htp-v81.so?htp_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp&_session=0` opened

`support -b HTP0 -o MUL_MAT` returned `Backend HTP0: OK`. bf16 matmul is not supported. f32, f16, and q4_0 are.

`test` mode then executed a Q4_0 GEMV, `256 x 16` by `256 x 1`, and it matched the CPU (`OK`). Two profile lines: 258 µs and 234 µs at 2.12 GHz. The bytes are about 3.4 KB, so the implied rate is about 0.014 GB/s. That 230 µs is a launch floor, not a memory rate.

`perf` mode on the large case in the perf list, Q4_0 `m=4096, n=1, k=14336` (33,030,144 weight bytes, row stride 8064, plus a 57,344-byte f32 activation and a 16,384-byte output):

- one op, not fused: 547 µs, 1,158,295 cycles, 2117 MHz
- implied rate on those bytes: 33.10 MB / 547 µs = 60.5 GB/s

That is the DMA-engine band from Hao et al., and it is the first measurement on this laptop that is not stuck at the 28 GB/s HVX load. The harness average over 3408 runs was 363.66 µs/run and 322.94 GFLOPS, but those runs were fused into `MUL_MAT_NX` batches (one logged batch was 213 ops in 282 ms). The fused average is not a per-token weight-read rate. The 547 µs line is.

A ~230 µs floor means an expert GEMV of a few hundred kilobytes does not get to use the 60 GB/s. At that floor, a few hundred serial expert launches per token would land near 5 tok/s before any draft. `MUL_MAT_ID` fusion is the row that has to collapse those launches. That is what the Qwen layer trace measures.

Profile lines are `GGML_LOG_DEBUG`. `test-backend-ops` prints them. `llama-cli` hides debug until `-lv 5` or `-v`.

## 2026-09-26 01:05 — Llama 3.2 1B Q4_0 decoded on HTP0

`llama-completion` with `--device HTP0 -ngl 99 -c 256 -lv 4` loaded the 729.75 MiB file and offloaded 17/17 layers. HTP0 held 743.45 MiB of weights. The host held another 205 MiB. KV cache for 256 tokens was 8 MiB on HTP0.

`common_perf_print`:

- prompt: 19.86 ms / 11 tokens = 553.85 tok/s
- decode: 82.81 ms / 9 runs = 9.20 ms/token = 108.68 tok/s

Generation stopped at end-of-turn, so the run is 9 tokens, not 16. The prompt is too short for the 554 tok/s figure to mean much. The decode number is the one to keep.

Hexagon reports 0 MiB free, so `--fit` warns and would rather not place the model there. With `-ngl 99` it continued anyway. A 20 GB load needs `-fit off` so that warning cannot shrink the offload.

The earlier `llama-cli` chat run printed 99.4 tok/s and then spun on EOF, reprinting the prompt until the process was stopped. It had grown to about 17 GB resident. Use `llama-completion` for measurements.

The homemade DMA skel was still reported as `0x80000406` here. That code is what `libcdsprpc` returns when `getenv("ADSP_LIBRARY_PATH")` does not see the directory. `SetEnvironmentVariable` after process start does not update the C runtime copy. `_putenv_s` does. After that fix the error is not `0x80000406`. See the 01:17 note.

## 2026-09-26 00:50 — what "20 tok/s" looks like on this chip family

Published llama.cpp Hexagon numbers on X2 Elite, from the Qualcomm author's PRs, are for small models. They are the comparison point, not a measurement of this laptop.

| Source | Model | X2 Elite prefill | X2 Elite decode |
|---|---|---:|---:|
| PR 24954 (matmul rework, 2026-06) | gemma-4-E2B Q4_0 | 1688 tok/s | 34.15 tok/s |
| PR 25085 (FA rework, merged 2026-07-01, commit 13e6738) | gemma-4-E2B Q4_0 | 1966 tok/s | 43.00 tok/s |
| PR 24954 | Qwen3.5-2B Q4_0 | 1258 tok/s | 37.17 tok/s |
| PR 25085 | Qwen3.5-2B Q4_0 | 1305 tok/s | 35.41 tok/s |
| PR 24954 | LFM2.5-8B-A1B Q4_0 | 1186 tok/s | 41.39 tok/s |

The FA rework raised gemma-4-E2B decode and slightly lowered Qwen3.5-2B decode. Prefill moved up in both. Our tree includes both PRs plus the 2026-09-21 HMX gated-delta PR (#29199). Qwen3.6-35B-A3B is 30 gated-delta layers and 10 full-attention layers, so #29199 is the op that has to be fast. That PR's body describes the kernel; a public X2 Elite tok/s table for Qwen3.6 was not in the fetched page.

LFM2.5-8B-A1B activates about 1B parameters and decodes at 41 tok/s on X2 Elite. Qwen3.6-35B-A3B activates about 3B. If decode stays proportional to active bytes, plain decode lands near 14 tok/s, and one accepted draft token lands near 28. That is the same shape as the 28 GB/s arithmetic in the goal doc. The per-op trace is what decides whether expert GEMVs are actually at that rate or are stuck in launch and gather.

Phone-class Hexagon (8 Gen 3 / 8 Elite, older backend) loses decode to the CPU on 0.6B–4B dense models because the GEMV is tiny. On 8 Elite Gen 5 (v81, SM8850) a community profile of Qwen3.5-9B saw HTP decode around 6–7 tok/s, with prefill several times the GPU. This laptop's NPU is the same v81 generation and a wider memory system. Those phone numbers are a floor, not a forecast.

Local measured paths from the earlier probes: CPU read ~154 GB/s, NPU copied QNN buffer ~15 GB/s, NPU rpcmem shared ~28 GB/s. Hao et al. (EuroSys 2026, arXiv 2509.23324) put HVX loads at 20–30 GB/s and the DMA engine near 60 GB/s on large regular blocks. The 547 µs GEMV above lands on that DMA number. The homemade DMA skel does not. See the 01:17 note.

## 2026-09-26 01:08 — full Qwen3.6-35B-A3B Q4_0 on one HTP0 session

`llama-completion -n 1 -c 128 -ngl 99 --device HTP0 -fit off` with `GGML_HEXAGON_PROFILE=1` loaded the 20.8 GB file onto one session. Log: `npu_campaign/htp-qwen-profile1.log`. Exit 0.

- 42/42 layers offloaded. HTP0 model buffer 18720 MiB, context 67 MiB, compute 125 MiB. Host model buffer 314 MiB. Load 19.2 s.
- The tool then waited for stdin, got EOF, and did not run a decode loop. `eval time` is 0. Prompt eval is real: 209.11 ms / 9 tokens = 23.23 ms/token = 43.0 tok/s wall clock.
- Profile of the batch-1 shaped graph for the last full-attention layer, blk.39:
  - fused top-8 gate+up `MUL_MAT_ID_NX`, Q4_0, 116 µs
  - top-8 down `MUL_MAT_ID`, Q4_0, 63 µs
  - seven serial `ADD`s that fold the eight expert outputs, about 6 µs each
  - shared expert is Q8_0, about 31 µs gate+up and 17 µs down
  - Q projection Q6_K 2048×8192, 107 µs
  - flash attention, 58 µs
  - the layer's device ops add to about 0.70 ms
- The output projection is Q6_K, 2048×248320, 2945 µs. That one op is about four of those layer-times.
- Router, softmax, argsort, and the top-8 gather are 4–20 µs, and they ran on HTP.

A decode loop is the next run (`-no-cnv -n 32`, profile off) so `eval time` is a real token rate. The 43 tok/s figure is prompt wall time over nine tokens, with graphs not reused.

## 2026-09-26 01:15 — Qwen3.6 decode, 32 tokens

Same file, same device flags (`--device HTP0 -ngl 99 -fit off -no-cnv --ignore-eos -n 32 -c 128`), profile off, log level 3. Log: `npu_campaign/htp-qwen-decode.log`. This log level does not print the offload line. The previous invocation of the same flags placed 42/42 layers and 18720 MiB on HTP0.

- load 25.6 s
- decode: 1901.61 ms / 32 runs = **59.43 ms/token = 16.83 tok/s**
- graphs reused on 30 of the 32 runs
- sampling 6.07 ms for the whole run, unaccounted 0.44 ms
- prompt eval recorded as 0 because the prompt was one token and the timer folded it into the decode runs
- the text continued from "Hello" for 32 tokens and stayed coherent
- `blk.40.nextn.*` tensors are in the GGUF (the MTP head, including `eh_proj`) and this loader skips them as unused. The draft weights are already in the file. They are not executed by this command.

16.83 tok/s is the plain single-token rate. One extra accepted draft token at the same forward cost would be about 34 tok/s. The 20 tok/s goal needs the forward to fall from 59 ms to 50 ms, or about 0.2 extra accepted tokens per forward. The batch-1 layer sum from the profile (about 0.70 ms times 40, plus the 2.95 ms output projection) is about 31 ms of device work, so roughly half of the 59 ms is still gap between those ops.

## 2026-09-26 01:17 — DMA open is AEE_ERPC, not the catalog

`island_bw/dma_bw_host.cpp` called `SetEnvironmentVariableA("ADSP_LIBRARY_PATH", ...)` after start. `libcdsprpc.dll` reads that variable with `getenv`, which keeps the copy from process creation. With the directory missing from that copy, open returns `0x80000406` (`AEE_EUNABLETOLOAD`). `_putenv_s` before `LoadLibrary` makes the same host open `libggml-htp-v81.so` (handle non-zero) from `pkg-wos/lib` and from `npu_campaign/dsp-try`.

After that fix, `libdma_bw_skel.so` and a minimal skel (`npu_campaign/dma_min_skel.c`, symbols `dma_bw_skel_handle_invoke`, `_uri`, `qaic_version` 10051, `dma_bw_slim`) still fail. The return is 512, `AEE_ERPC` (`0x80000400+0x200`). Opening the working ggml skel with a symbol name that is not in the file returns the same 512. A missing file still returns `0x80000406`. `signtool verify /pa /c` accepts both catalogs. The DSP loader is not treating this ELF as that interface. No DMA GB/s.

One open without `_modver` returned 0 and the invoke returned `AEE_EBADPARM` in about 4 ms. That is a rejected call, not a 16 MiB copy.

## 2026-09-26 01:21 — profile 3, one decode token, layer 0

`llama-completion -n 1 -c 128 -ngl 99 --device HTP0 -fit off -no-cnv --ignore-eos -lv 5` with `GGML_HEXAGON_PROFILE=3`. Log: `npu_campaign/htp-qwen-profile3.err` (76 MB, 914,730 `trace-evt` lines). Exit 0. 42/42 layers offloaded. The printed eval time is 50828 ms for that one token. That is the cost of writing the trace, not the device time. The 16.83 tok/s figure from the unprofiled 32-token run still stands.

Layer 0 is a gated-delta layer. The rows below are the batch-1 graph (the second copy of each op in the log). Clock is the `mhz` on that line, about 2130. Bytes are the active set: Q4_0 is 18 bytes per 32 weights, Q4_1 is 20, Q8_0 is 34. Top-8, not all 256 experts. Implied GB/s is those bytes divided by the op's wall microseconds.

| op | usec | bytes | GB/s |
|---|---:|---:|---:|
| router `ffn_gate_inp` f32 2048×256 | 19 | 2.10 MB | 110 |
| top-8 gate+up `MUL_MAT_ID_NX` Q4_0 | 122 | 9.46 MB | 77.6 |
| top-8 down `MUL_MAT_ID` Q4_1 | 85 | 5.32 MB | 62.6 |
| shared gate+up `MUL_MAT_NX` Q8_0 | 35 | 2.24 MB | 64.0 |
| shared down `MUL_MAT` Q8_0 | 17 | 1.12 MB | 66.1 |
| gated-delta + state copy | 68 | 2.10 MB state |  |
| seven expert `ADD`s | 5–6 each | 8 KB | launch |

Inner stages for the two expert GEMVs, from `trace-evt` start/stop whose cycle stamps sit inside that op. The sums count all HVX threads, so they are larger than the wall time. Eight threads overlap. Dividing the DMA sum by 8 lands on the wall time, which means each thread is in DMA for essentially the whole op.

| op | stage | intervals | sum usec | usec / 8 |
|---|---|---:|---:|---:|
| gate+up 122 us | DMA | 257 | 905 | 113 |
| gate+up | HVX_COMP | 256 | 504 | 63 |
| gate+up | HVX_A_QUANT | 8 | 2.9 |  |
| down 85 us | DMA | 513 | 611 | 76 |
| down | HVX_COMP | 512 | 337 | 42 |
| down | HVX_A_QUANT | 8 | 31.3 | 3.9 |
| down | L2FLUSH | 8 | 0.5 |  |

The expert GEMVs are on the DMA engine, in the same 60–80 GB/s band as the 547 µs Q4_0 GEMV. They are not the 28 GB/s HVX-load path. The router line above 100 GB/s is a 2 MB weight that the op says it staged through 1 MB of VTCM; treat it as a small-op timer, not a new ceiling. The seven 5 µs adds are the launch floor, about 40 µs per layer of serial cleanup.

## 2026-09-26 08:50 — the 59 ms token is 21 DSP batches

`parse_profile.py` over `htp-qwen-profile3.err` splits the log into 42 OPBATCHes: one forward of a 2-token prompt (batches 0–20) and one batch-1 forward (batches 21–41). Inside a batch the idle time between ops is about 0.1 ms for the whole token. The DSP is packed.

The batch-1 forward:

- sum of op usec: 42.1 ms
- sum of OPBATCH usec: 44.1 ms
- the 2 ms between those two is inside the DSP queue

The unprofiled 32-token run was 59.43 ms/token. 59.43 − 44.1 = 15.3 ms that is not inside any OPBATCH. That is one host round-trip per batch boundary, about 0.8 ms, times 20 boundaries.

`fit_op` flushes the batch when the next tensor would exceed `b_vmem_max` (the session's ~3.3 GB map). The biggest decode batch is layers 0–5 (294 ops, 6.1 ms). Later batches are one or two layers. The op cap is 1280 and was not hit. The 18.7 GB weight buffer does not stay mapped for the whole token.

`GGML_HEXAGON_DEVICES=8` opened four sessions and then failed. HTP0, HTP1, HTP2, and HTP7 came up (session ids 0–3, domains 3, 7, 11, 15, each with the same 3.31 GB vmem). HTP3's reserve of virtual index 3 returned `0x200` (`AEE_ERPC`). Log: `npu_campaign/htp-qwen-ndev8.log`. This driver has four session slots on the one NSP, not eight.

Four sessions still cut the flush count on paper. The measurement does not. `htp-qwen-ndev4.log`: all four sessions opened, 42/42 layers offloaded, buffer sizes 5100 / 4540 / 5000 / 4080 MiB. Every one of those is larger than the 3158 MiB map. Decode was 2182 ms / 32 tokens = **68.20 ms/token = 14.66 tok/s**, slower than the single session. The activation handoff between sessions cost more than the flushes it removed.

The batch seams are always the shared-expert gate (`ffn_gate_inp_shexp`), which is just the next op after the batch fills. `HTP_OP_MAX_BUFS` was 16, and that cap is only on the host. The DSP walks `n_bufs` from the request.

## 2026-09-26 09:05 — 64 buffers, 20.37 tok/s

Host cap raised to 64 in `llama.cpp-hexagon/ggml/src/ggml-hexagon/htp/htp-ops.h`. `enqueue_op` logs the counters when a batch flushes. Rebuild log `rebuild-bufs.log`, catalog re-signed. `cmake --install` put the real `ggml-opencl.dll` back and the process fast-failed `0xC0000409` until `npu_campaign/ggml-opencl-stub.dll` was copied over it again.

One session, same 32-token command as the 16.83 run. Log `htp-qwen-bufs64.log`.

- startup line: `n-bufs 64 n-tensors 8192 n-ops 1280`
- the flush log did not fire. No batch hit the 64-buffer cap.
- decode: 1571.31 ms / 32 runs = **49.10 ms/token = 20.37 tok/s**
- graphs reused 30/32
- the text continued from "Hello" and stayed coherent
- scheduler still reports `splits = 42` on a 3787-node graph. Those splits are llama's, above the Hexagon batcher.

49.10 ms against the 44.1 ms of device OPBATCH time leaves about 5 ms. The 15 ms of flush waits is gone.

## 2026-09-26 09:20 — the 42 splits are the Q5_0 shared experts

`GGML_SCHED_DEBUG=2` on one reserve of the same graph. Log `htp-qwen-splits.log`. The 42 splits are not 42 layers and they are not the Hexagon op batcher. They alternate CPU and HTP0:

- split 0, CPU, no copies: `GET_ROWS` of `token_embd.weight`. The input layer is hard-wired to the CPU in `llama_model::load_tensors`.
- split 1, HTP0, 7 copies, including the 8 KB embedding row.
- then 20 pairs. Each CPU split copies one `attn_post_norm-<L>` (8 KB) and runs four ops: `MUL_MAT` gate, `MUL_MAT` up, `SWIGLU`, `MUL_MAT` down. Each HTP split copies the 8 KB `ffn_shexp-<L>` back and continues. The last HTP split also copies `out_ids`.

The 20 CPU layers are 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29, 30, 32, 33. Those are exactly the layers whose shared-expert weights are Q5_0. The GGUF has 60 Q5_0 tensors, three per layer (`ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp`), each 2048×512, 22 bytes per 32 weights, 704 KiB. 20 × 3 × 704 KiB = 41.25 MiB. `token_embd` Q4_0 is 272.81 MiB. Sum 314.06 MiB, which is the whole CPU model buffer. Hexagon `MUL_MAT` accepts Q4_0, Q4_1, Q8_0, IQ4_NL, MXFP4, Q4_K, Q5_K, Q6_K, f16, and f32. It does not accept Q5_0, so `select_weight_buft` leaves those weights on the CPU and the scheduler follows the weight.

Profile-1 already showed the handoff. On layer 5 the HTP batch ends at `ffn_moe_out-5` and the next batch starts with `ffn_gate_inp_shexp`, whose mul reads `HTP0#ffn_shexp-5#0`. That name is the scheduler copy. Layer 0's shared expert is Q8_0 and stays inside the HTP batch (34 µs gate+up, 16 µs down, batch-1).

A temporary timer around `ggml_backend_sched_compute_splits` (added, measured, then reverted) on a warm 12-token run, log `htp-qwen-splits-time12.log`:

- 21 CPU splits and 21 HTP splits, every token
- CPU `graph_compute` settles near 1.0–2.6 ms, with spikes to about 4.6 ms. That is the 60 Q5_0 GEMVs plus 20 SWIGLUs.
- the CPU-split copy time is about 37 ms. That is the host blocked in `ggml_backend_synchronize` on the previous HTP chunk, so it is mostly the DSP time, not a new cost.
- HTP `graph_compute` is enqueue only, about 1.2 ms. HTP copy of the 8 KB result is 0.03 ms.

The same binary without that log, 32 tokens, `-lv 3`, log `htp-qwen-splits-time.log`: 1581.63 ms / 32 = **49.43 ms/token = 20.23 tok/s**. Same rate as the 49.10 ms run. The timer was not the rate.

So the 5 ms is those 20 boundaries. About 1–2.5 ms is the CPU math. About 1.2 ms is walking 21 HTP subgraphs. The rest is the RPC at each sync, on top of device time that is already inside the 44.1 ms. Putting the 60 Q5_0 weights on HTP removes the boundaries. The same shape in Q8_0 costs about 50 µs on HTP, so 20 layers add about 1 ms of device time and delete the CPU math and the 20 syncs. Wall should fall from about 49 ms to about 45 ms, which is 22 tok/s, against the 44.1 ms device floor. It closes most of the 5 ms. It does not open another 10 tok/s. Q5_0 is not a host-side flag. The repack set in `ggml_hexagon_is_repack_type` has no Q5_0 entry, and the DSP matmul tiles do not either.

## 2026-09-26 09:41 — Q5_0 shared experts are Q5_K on HTP, 23.47 tok/s

A Q5_0 DSP kernel was not written. Q5_K is already a Hexagon matmul type and it is the same size as Q5_0 (176 bytes per 256 weights, 22 bytes per 32). The 60 tensors are `blk.{5,6,8,9,11,12,14,15,17,18,20,21,23,24,26,27,29,30,32,33}.ffn_{gate,up,down}_shexp.weight`. Gate and up are 2048×512, down is 512×2048, so `ne[0]` is divisible by 256 on every one. Each blob is 720896 bytes. `npu_campaign/requant-shexp.cpp` dequantizes with ggml and writes `ggml_quantize_chunk(Q5_K)` back over the same bytes, then changes the GGUF type field from 6 to 13. Offsets and file size do not move. Worst tensor was `blk.24.ffn_down_shexp`: rmse 0.00035, max abs 0.0175. The original blobs are in `npu_campaign/q5_0_shexp_backup.bin` (43257796 bytes). Restore with `requant-shexp.exe --restore <gguf> q5_0_shexp_backup.bin`. The file still says `general.file_type = Q4_0`.

Same 32-token command as the 20.37 run (`-no-cnv --ignore-eos -n 32 -c 256 -b 128 -ub 128 -p Hello --temp 0 --device HTP0 -ngl 99 -fit off --load-mode none`). Log `htp-qwen-q5k-b128.log`.

- loader: q5_0 is gone, q5_K is 60 tensors. q4_0 208, q4_1 5, q8_0 105, q6_K 5, unchanged
- CPU model buffer 272.81 MiB (was 314.06). That is `token_embd` alone
- HTP0 model buffer 18765.11 MiB (was 18720.11). The repacked Q5_K tiles added 45.00 MiB
- graph nodes still 3787. **splits = 2** (was 42). The leftover CPU split is the token-embedding `GET_ROWS`, which `load_tensors` pins to the host
- HTP compute buffer 124.32 MiB, CPU compute 2.07 MiB, same shape as the 125.25 / 2.57 of the 20.37 run
- decode: 1392.79 ms / 32 = **43.52 ms/token = 22.98 tok/s**. Graphs reused 30/32
- greedy text changed (the requant is not bit-exact) and stayed coherent

The 1280-op host cap is now the flush. `fit_op` logs once per token: `n-ops 1280 n-bufs 17/64 n-tens 678/8192 vmem 205520896/3311808512`. Buffers and vmem are not the limit. The DSP sizes the batch from `req->n_ops`, so the cap is the host `GGML_HEXAGON_OPBATCH` (default 1280), not a skel rebuild. One extra submit per token.

`GGML_HEXAGON_OPBATCH=3072` on the same command, log `htp-qwen-opbatch3072.log`. Startup prints `n-ops 3072`. The flush line does not fire. Splits stay 2. Decode: 1363.34 ms / 32 = **42.60 ms/token = 23.47 tok/s**. Graphs reused 30/32. That is 0.9 ms under the 1280-cap run, which is the removed submit. It is also under the old 44.1 ms device sum, because that sum was a different token and still had the shared-expert copies at the CPU boundary. `pkg-wos/bin/ggml-opencl.dll` is still the stub. `HTP_OP_MAX_BUFS` is still 64. The default op cap was not edited.

## 2026-09-26 10:04 — one warm batch-1 token is 38.00 ms on the DSP

`GGML_HEXAGON_PROFILE=1` and `GGML_HEXAGON_OPBATCH=3072`, same flags as the 23.47 tok/s run except `-n 4 -lv 5`. Log `htp-qwen-profile1-warm.log.err` (the profile lines are stderr). Exit 0. Startup still prints `n-ops 3072`. Splits stay 2. 42/42 layers. The printed eval time is 981.98 ms / 4 = 245.50 ms/token. That is the cost of writing 1923 debug lines per token, not the rate. The 42.60 ms figure stands.

Five OPBATCHes, one each, so the 3072 cap did not flush:

| batch | what | n-ops | OPBATCH µs |
|---|---|---:|---:|
| 0 | warmup | 2053 | 56705 |
| 1 | first decode, graph not reused | 1983 | 40727 |
| 2 | same shape as the warm tokens | 1923 | 38349 |
| 3 | reused | 1923 | 38154 |
| 4 | reused, the warm token | 1923 | 38003 |

`graphs reused = 2` of the 4 decode runs. The last token is the one below. Its two neighbors with the same 1923-op graph are 38.35 ms and 38.15 ms, so the shape is stable. Inside that batch the op times sum to 36904 µs and the cycle-stamp idle between ops is 37 µs. The other 1099 µs is the DSP queue, not a host flush. 42.60 − 38.00 = 4.60 ms of the wall token is outside this OPBATCH: the CPU embedding `GET_ROWS` and the one host submit/sync. This run cannot time that 4.6 ms, because the log itself is the wall.

The output projection is not the token. It is the only op over 128 µs, and it is fast. Routed-expert GEMVs are the next block, and they are on the DMA band. A third of the batch is ops shorter than 30 µs.

| piece | µs | % of 38.00 ms | note |
|---|---:|---:|---|
| Q6_K `output.weight` 2048×248320 | 2997 | 7.9 | 417.2 MB, 139 GB/s, `hvx-tiled` |
| top-8 gate+up `MUL_MAT_ID_NX` Q4_0, ×40 | 4506 | 11.9 | 9.44 MB/layer, 113 µs, 84 GB/s |
| top-8 down `MUL_MAT_ID`, ×40 | 2622 | 6.9 | 35× Q4_0 at 65 µs (73 GB/s), 5× Q4_1 at 70 µs |
| seven expert `ADD`s ×40 | 1596 | 4.2 | 5–6 µs each, the launch floor |
| router `ffn_gate_inp` f32 ×40, plus softmax/argsort/norm | 2001 | 5.3 | the matmul is 20 µs; the rest is ~5 µs ops |
| GDN `attn_qkv` Q4_0 2048×8192, ×30 | 2825 | 7.4 | 9.44 MB, 94 µs, 100 GB/s |
| GDN `attn_gate` Q4_0 2048×4096, ×30 | 1546 | 4.1 | 4.72 MB, 52 µs, 92 GB/s |
| GDN `ssm_out`, ×30 | 2070 | 5.4 | 14× Q8_0 at 86 µs (~99 GB/s), 16× Q4_0 at 54 µs (~87 GB/s) |
| full-attn Q + V + out + flash, ×10 | 2246 | 5.9 | Q is 95–106 µs, V 31 µs, out 54 µs, flash 41 µs |
| shared-expert matmuls, ×40 | 2029 | 5.3 | Q5_K gate+up 28 µs / down 14 µs, Q8_0 31 / 16. Q5_K is the faster of the two |
| conv-state `CONCAT` ×30 | 2889 | 7.6 | 128 KB f32, 96 µs, 1.4 GB/s |
| conv-state `CPY` ×30 | 1206 | 3.2 | 96 KB, 40 µs, 2.5 GB/s |
| `GATED_DELTA_NET+CPY` ×30 | 1596 | 4.2 | 53 µs, includes the 2 MB state writeback |
| state `GET_ROWS` ×30 | 1000 | 2.6 | 2.10 MB `cache_s` at 27 µs (77 GB/s) plus the 96 KB conv read |
| `ssm_alpha` + `ssm_beta` + `SSM_CONV` | 855 | 2.2 | f32 projections at 9 µs, conv at 11 µs |
| DSP queue inside the one batch | 1099 | 2.9 | OPBATCH minus the sum of op µs |
| norms, mul, silu, sigmoid, swiglu, residual adds | 4920 | 12.9 | largest class is `MUL` at 987 µs across 110 ops; no single hotspot |

The rows above sum to the 38003 µs batch.

1231 of the 1923 ops are under 8 µs and add to 6.67 ms. 1383 ops under 10 µs add to 7.97 ms. That is a launch tax spread across the token, not a bandwidth miss. The GEMVs above are already at 73–139 GB/s. Fusing them further does not move 42 ms.

The conv-state pair is the concentrated form of that tax. `CONCAT` builds a 4×8192 f32 window and `CPY` writes the 3×8192 tail back, 60 serial kernels, 4.10 ms, about 1.6 GB/s on 6.7 MB. The 2 MB recurrent state itself is fine: the `GET_ROWS` is at 77 GB/s and the gated-delta kernel is 1.60 ms for 30 layers. Shared-expert Q5_K is not the rate. The output projection would have to disappear entirely to buy 3 ms, which is 25.3 tok/s if nothing else changed, and it is already the widest read in the token.

## 2026-09-26 12:20 — fused conv window, 25.79 tok/s

The 30 gated-delta layers build the depthwise conv as `CONCAT` of the packed 3-tap history with the new token, `CPY` of that tail back into the cache, then `SSM_CONV`. That was 60 kernels, 4.10 ms, about 1.6 GB/s on 6.7 MB. See the 10:04 table.

`GGML_HEXAGON_FUSE_SSM_CONV_WIN` is on when op fusion is 1. On a decode token it replaces those three with one `SSM_CONV`. Sources are the packed history, the f32 kernel, and the new token. Destinations are the conv output and the shifted history. The 4-wide window is not written. It only matches d_conv 4, one token, one sequence.

The mac is the existing qf32 d_conv==4 kernel. The gather was the cost. Doing that gather with scalar loads and stores in VTCM is exact and slower than the three kernels it replaces. Log `htp-qwen-convwin-scalar.log`: 1385.85 ms / 32 = **43.31 ms/token = 23.09 tok/s**. The greedy text is the same PHP fragment as the 23.47 run. An in-register permute of the 3-wide history was not bit-exact, so it is not in the kernel.

The gather that is both exact and faster copies each 32-channel block out of VTCM with HVX, invalidates those cache lines, gathers the window in cached scratch, and stores it back to VTCM for the existing unpack and mac. Log `htp-qwen-convwin-final.log`. Same command as the 23.47 run (`GGML_HEXAGON_OPBATCH=3072`, profile off, `-n 32`).

- decode: 1240.82 ms / 32 = **38.78 ms/token = 25.79 tok/s**
- graphs reused 30/32
- greedy text matches `htp-qwen-opbatch3072.log` (`Hello,` then the PHP PDO fragment)
- the same gather before unused tables were deleted was 26.10 tok/s in `htp-qwen-convwin-bounce3.log`, same text. The 0.3 tok/s is run noise
- `pkg-wos/bin/ggml-opencl.dll` is still the stub. `HTP_OP_MAX_BUFS` is still 64. The Q5_0 backup was not restored

42.60 − 38.78 = 3.82 ms off the wall token. That is most of the 4.10 ms of `CONCAT`+`CPY`. No new device profile was taken. The output projection and the sub-8 µs ops were not touched.

## 2026-09-26 13:10 — embedding on HTP aborts the graph

The last CPU split is `token_embd` `GET_ROWS`. Pointing the input layer at HTP0 did place the table there: the HTP buffer grew from 18765.11 MiB to 19037.92 MiB, and `splits` went from 2 to 1. Log `htp-qwen-embd-q4.log`. The warmup then aborted: `get-tensor-async failed : dsp-error NO-SUPPORT`.

Hexagon `GET_ROWS` accepts f32, f16, q8_0, and i32. The embedding is Q4_0, and the kernel also rejects an index tensor whose address is a 64-bit buffer offset (`htp_tensor_is_extended`). A Q4_0 row dequant is now in the skel (`get_rows_thread_q4_0_*`). The lookup returned no-support because the index address is above 4GB, not because the row type is missing. The next section stages that index and keeps the table on HTP.

A 32-token rerun with that revert, same `OPBATCH=3072` and the conv-window skel, is `htp-qwen-embd-reverted.log`: CPU buffer 272.81 MiB, HTP 18765.11 MiB, splits 2, **38.57 ms/token = 25.92 tok/s**. That matches 25.79 within run noise. The installed rate to keep is 25.8 tok/s.

## 2026-09-26 12:40 — embedding on HTP, splits = 1, same rate

The no-support abort was the index, not the Q4_0 row. `op_get_rows` loads the index with a 32-bit pointer and used to reject any tensor whose DSP address is above 4GB. The row itself already uses the 64-bit DMA path. The index is at most a few hundred `i32`s. When that address is extended, the skel now DMAs those bytes into VTCM past the row scratch and the threads read the copy. `token_embd` stays on the first HTP device.

Log `htp-qwen-embd-idx.log`. Same command as the 25.79 run (`GGML_HEXAGON_OPBATCH=3072`, profile off, `-n 32 -b 128 -ub 128 -p Hello --temp 0 --device HTP0`), plus `-lv 4` so the split line is visible.

- `load_tensors: input layer assigned to HTP0`
- no CPU model buffer. HTP0 model buffer **19037.92 MiB** (18765.11 + 272.81)
- `sched_reserve: graph: nodes = 3787, splits = 1`, sched copies = 1, CPU compute buffer 1.07 MiB
- decode: 1243.24 ms / 32 = **38.85 ms/token = 25.74 tok/s**
- graphs reused 30/32
- greedy text matches the PHP PDO fragment
- host model memory in the breakdown is 0 MiB

25.74 does not beat 25.79. Moving the lookup onto the DSP removed the CPU split and did not move the wall time. The remaining sched copy is the logits, not the embedding. `pkg-wos/bin/ggml-opencl.dll` is still the stub. `HTP_OP_MAX_BUFS` is still 64. The Q5_0 backup was not restored.

`cmake --install` does not copy `htp-v81-prefix/src/htp-v81-build/libggml-htp-v81.so` onto the skel the catalog hashes. The skel that ran was copied from that prefix build and the catalog was signed again. Install also puts the real OpenCL DLL back; the stub was copied over it again.

## 2026-09-26 13:09 — expert ADD chain fused, 26.41 tok/s

Before the conv fuse, 1231 ops under 8 µs were 6.67 ms. After that fuse the concentrated remainder is the MoE expert sum. Each of the 40 layers adds eight views of `ffn_moe_weighted` with seven serial ADDs. In the 10:04 profile those runs are about 40 µs, ~1.6 ms on the token. The rows are packed f32, 2048 elements, and adjacent.

`GGML_HEXAGON_FUSE_MOE_ADD` is on when op fusion is 1. On a decode token it replaces that chain with one `ADD+N`. Sources are the eight views. The destination is the last ADD. The running sum is not written. The skel DMAs the packed block into VTCM and left-folds it with the same HVX add as `HTP_OP_ADD`, so the greedy text stays exact. It only matches one token, f32, `n_elem` a multiple of 32, and rows laid out in order.

Same command as the 25.74 run (`GGML_HEXAGON_OPBATCH=3072`, profile off, `-n 32 -b 128 -ub 128 -p Hello --temp 0 --device HTP0 -ngl 99 -fit off`). The embedding stays on HTP0.

- `htp-qwen-addn.log` (`-lv 4`): `fused expert ADD chain into ADD+N (n_src 8 n_elem 2048)`. splits = 1, HTP buffer 19037.92 MiB, sched copies = 1, n-ops 3072, no flush. Decode: 1221.47 ms / 32 = **38.17 ms/token = 26.20 tok/s**. Graphs reused 30/32.
- `htp-qwen-addn2.log`: 1211.85 ms / 32 = **37.87 ms/token = 26.41 tok/s**. Graphs reused 30/32.
- Greedy text matches the PHP PDO fragment on both. It starts `Hello,` and ends at `$pdo = new PDO('mysql:host=localhost`.
- `pkg-wos/bin/ggml-opencl.dll` is still the stub. `HTP_OP_MAX_BUFS` is still 64. The Q5_0 backup was not restored. The conv-window fuse and the Q4_0 extended-index `GET_ROWS` path were not reverted.

38.78 − 37.87 = 0.91 ms on the faster run. The pair is 0.61 ms and 0.91 ms under the 25.79 wall. That is about half of the old seven-ADD chain, not the rest of the 6.67 ms. The output projection was not touched. `cmake --install` still does not publish the v81 skel the catalog hashes; the skel was copied from the prefix build and the catalog was signed again. Install also puts the real OpenCL DLL back; the stub was copied over it again.

## 2026-09-26 13:34 — shared-expert gate fused, 26.92 tok/s

The warm profile was not rerun. After the conv window and the expert ADD+N, the largest leftover short chain is the shared-expert epilogue, 40 times. From the 10:04 token, per layer: sigmoid of one f32 (5 µs), mul of the 2048-wide shared expert by that scalar (6 µs), add of `ffn_moe_out` (5.6 µs), add of the attention residual (5.4 µs). Sum 887 µs. The scalar gate matmul stays; it is an 8 KB dot, and a new dot would not match the GEMV rounding.

`GGML_HEXAGON_FUSE_SHEXP_GATE` is on when op fusion is 1. On a decode token it replaces those four with one `ADD+SG`. Sources are the moe row, the residual, the shared-expert row, and the pre-sigmoid gate. The destination is `l_out`. The running products are not stored. The skel uses the same HVX sigmoid, scalar mul, and two adds, in that order. It only matches one token, f32, `n_elem` a multiple of 32.

`-n 1 -lv 4` prints all three fuses. Log `htp-qwen-sg-lv4.log.err`: conv window, `ADD+N (n_src 8 n_elem 2048)`, and `ADD+SG (n_elem 2048)`. The 42.16 ms on that one token is graph setup, not the rate.

Same 32-token command as the 26.41 run (`GGML_HEXAGON_OPBATCH=3072`, profile off, `-n 32 -b 128 -ub 128 -p Hello --temp 0 --device HTP0 -ngl 99 -fit off --load-mode none`).

- `htp-qwen-sg.log`: 1192.68 ms / 32 = **37.27 ms/token = 26.83 tok/s**. Graphs reused 30/32.
- `htp-qwen-sg2.log`: 1188.59 ms / 32 = **37.14 ms/token = 26.92 tok/s**. Graphs reused 30/32.
- Greedy text matches `htp-qwen-addn2.log` on both. It starts `Hello,` and ends at `$pdo = new PDO('mysql:host=localhost`.
- `pkg-wos/bin/ggml-opencl.dll` is still the stub. `HTP_OP_MAX_BUFS` is still 64. The Q5_0 backup was not restored. The conv-window fuse, the ADD+N fuse, and the Q4_0 extended-index `GET_ROWS` path were not reverted.

37.87 − 37.14 = 0.73 ms on the faster run. The pair is 0.60 ms and 0.73 ms under the 26.41 wall. That is most of the 0.89 ms chain, not the router norm or the q/k RMS+SCALE pairs. The output projection was not touched.

## 2026-09-26 14:20 — RMS_NORM+SCALE changed the tokens

Q/K is `RMS_NORM` of a 128-wide head followed by `SCALE` by `1/sqrt(128)` (0.088388). A kernel `HTP_OP_RMS_NORM_SCALE` does the norm, scales into the source scratch, and copies back. The matcher was enabled for one run.

- `htp-qwen-rms.log`: the log line fired (`n_elem 128 scale 0.0883883`). 1206.69 ms / 32 = **37.71 ms/token = 26.52 tok/s**. The text is a React Native async-storage fragment, not the PHP PDO fragment.
- The matcher is disabled again. `htp-qwen-rms-off.log`: 1187.79 ms / 32 = **37.12 ms/token = 26.94 tok/s**. The text is an aws-cdk VPC fragment, still not PHP.

Earlier PHP runs used different sampler seeds (1421681837, 2784298388, 3677408291) and the same text, so temp 0 is deterministic. These two texts differ from that and from each other. The skel rebuilt for this pass is not bit-exact with the 26.92 skel while the matcher is on. The rate is in the same band. Do not call 26.94 the new best. The matcher stays off. The next note shows the matcher-off skel does print PHP.

## 2026-09-26 14:48 — matcher off, the PHP fragment is back

The Q4_0 embedding dequant was the first suspect and it is not the token change. `dequant_row_q4_0_f32` is the ggml order: low nibble then high nibble, bias 8, scale `GGML_FP16_TO_FP32`. `block_q4_0` is 18 bytes (`static_assert` in `ggml-common.h`). fp16→fp32 is the integer software conversion, which is exact for every finite fp16. `use_dma` stays off because the destination is f32. One embedding row is 2048 weights, one chunk, so the `ne00` length matches the DMA'd row. `get-rows-ops.c` was last edited at 12:36, before `htp-qwen-sg2.log`. The 14:12 v81 rebuild recompiled it only because `htp-ops.h` gained `HTP_OP_RMS_NORM_SCALE`.

The matcher-on run is the one that leaves PHP. `build_gdn_l2_norm` is `SCALE(RMS_NORM(x, eps/n), 1/sqrt(n))` with n = 128, so the logged scale 0.0883883 is the real op, not a bad match. `htp-qwen-rms.log` fused it and printed the React Native fragment. `try_fuse` still returns before `try_fuse_rms_norm_scale`. Leave it that way. The kernel does call the same HVX rms and the same scale, but it scales into the source scratch and `memcpy`s back, and that run was not the PHP text.

With that matcher left off, the installed skel (835400 bytes, identical in `pkg-wos/lib` and both v81 build copies) prints PHP again. Canonical command: `GGML_HEXAGON_OPBATCH=3072`, `-no-cnv --ignore-eos -n 32 -c 256 -b 128 -ub 128 -p Hello --temp 0 --device HTP0 -ngl 99 -fit off --load-mode none`.

| log | seed | tok/s | ms/token | text |
|---|---:|---:|---:|---|
| `htp-qwen-repro.log` | 2818050408 | 26.68 | 37.49 | PHP PDO |
| `htp-qwen-repro2.log` | 2502661726 | 26.88 | 37.20 | PHP PDO |
| `htp-qwen-bdefault.log` | 2143315491 | 26.26 | 38.08 | PHP PDO |

The third row omits `-b`/`-ub`. Context batch clamps to 256 (`min(n_ctx, 2048)`). The generate line still prints the unclamped logical batch, 2048. That is the same generate line as `htp-qwen-rms-off.log`, and this time the text is PHP, at a slower 38.08 ms. Keep `-b 128 -ub 128`. Both canonical runs reuse 30/32 graphs, log conv window, ADD+N, and ADD+SG, and do not log RMS or a flush. splits = 1. q5_K is still 60 tensors.

`htp-qwen-rms-off.log` (aws-cdk, 26.94 tok/s, generate n_batch 2048) did not come back. The dequant source does not explain a deterministic aws-cdk result: the same skel and the same default-batch command now print PHP. Do not replace 26.92 with 26.94. 26.88 is the same band as 26.92, not a new best.

`pkg-wos/bin/ggml-opencl.dll` is the stub, 102400 bytes, same size as `npu_campaign/ggml-opencl-stub.dll`. `HTP_OP_MAX_BUFS` is 64. The Q5_0 backup was not restored.

## 2026-09-26 15:15 — router norm fused, PHP held, wall did not

The open chain was `SUM_ROWS` of the 8 selected expert weights, `CLAMP` of that scalar at the f16 minimum (`6.103515625e-5`), then `DIV`. On the warm token each layer is about 3 + 4 + 4 µs, so 40 layers are about 0.44 ms. The f32 DIV is not a divide. `compute_scalar_dma_div_f32` multiplies by `1.0f / val`.

`GGML_HEXAGON_FUSE_MOE_WNORM` is on when op fusion is 1. On a decode token it replaces those three with one `MOE+WNORM`. The source is the weight row. The destination is the DIV output. The sum is not stored. The skel calls `hvx_reduce_sum_f32`, then `hvx_clamp_scalar_f32_aa` of that one float, then `hvx_mul_scalar_f32` by the reciprocal. It only matches one row, f32, `n_elem` from 1 to 256, and a positive clamp minimum under 1e-3 with a huge maximum. `try_fuse_rms_norm_scale` is still not called.

`-n 1 -lv 4` prints the line. Log `htp-qwen-wnorm-lv4.log.err`: conv window, `MOE+WNORM (n_elem 8)`, `ADD+N (n_src 8 n_elem 2048)`, `ADD+SG (n_elem 2048)`. No RMS line. `n-ops 3072`. No flush. `splits = 1`. HTP0 model buffer 19037.92 MiB. The one-token time is graph setup, not the rate.

Same 32-token command as the 26.88 run (`GGML_HEXAGON_OPBATCH=3072`, profile off, `-n 32 -b 128 -ub 128 -p Hello --temp 0 --device HTP0 -ngl 99 -fit off --load-mode none`).

| log | seed | tok/s | ms/token | text |
|---|---:|---:|---:|---|
| `htp-qwen-wnorm.log` | 3922511455 | 26.60 | 37.59 | PHP PDO |
| `htp-qwen-wnorm2.log` | 1639016804 | 27.86 | 35.90 | PHP PDO |
| `htp-qwen-wnorm3.log` | 2551100566 | 26.37 | 37.92 | PHP PDO |

All three reuse 30/32 graphs and end at `$pdo = new PDO('mysql:host=localhost`. The third run was only to check the 27.86 outlier. It landed on the slow side. 26.60 and 26.37 are the 26.68 repro band, not under the 37.14 ms of `htp-qwen-sg2.log`. A 0.3 ms launch save is smaller than this spread. Do not replace 26.92. Leave the matcher on. The text did not change, so the fuse was not disabled.

The installed skel is `pkg-wos/lib/libggml-htp-v81.so`, 835496 bytes, the v81 prefix build. `build-wos/ggml/src/ggml-hexagon/libggml-htp-v81.so` is still the stale 835400-byte file from 14:13. The catalog was signed after the prefix copy. `pkg-wos/bin/ggml-opencl.dll` is the stub, 102400 bytes. `HTP_OP_MAX_BUFS` is 64. The Q5_0 backup was not restored. ADD+SG, ADD+N, the conv-window fuse, and the Q4_0 `GET_ROWS` path were not reverted.

The expert-weight `MUL` (`ffn_moe_down` times the normed weights, ~7 µs, ~0.28 ms) was not fused. The q/k `RMS_NORM`+`SCALE` matcher stays off.

## 2026-09-26 15:45 — 128 tokens, 27.6 tok/s, PHP held

The 32-token average counts two tokens whose graph is not reused. A 128-token run dilutes those. Same stack as the router-norm note: conv window, ADD+N, ADD+SG, MOE+WNORM, embedding on HTP, `OPBATCH=3072`, `-b 128 -ub 128`, `--temp 0`.

| log | tok/s | ms/token | reused | text |
|---|---:|---:|---:|---|
| `htp-qwen-n128.log` | 27.61 | 36.21 | 126/128 | PHP PDO, through the INSERT |
| `htp-qwen-n128b.log` | 27.51 | 36.36 | 126/128 | PHP PDO |

The two runs differ by 0.15 ms. The 27.86 tok/s 32-token outlier (35.90 ms) is the warm-token rate; the 32-token averages near 37 ms were the two cold graphs. **27.6 tok/s** over 128 tokens is the rate to keep. The expert-weight multiply is 0.28 ms, inside that 0.15 ms spread, so it stays unfused.

## 2026-09-26 16:05 — MTP accepts well and the verify step is not free

The Q4 file's `blk.40` nextn tensors load when llama-cli is started with `--spec-type draft-mtp --spec-draft-n-max 1`. `llama-completion` does not take that flag. `-no-cnv` is completion-only; the cli uses `--single-turn`. The draft context is a second schedule on the same HTP0. The model buffer grows from 19038 MiB to 19893 MiB. The fuses still fire (conv window, ADD+N, ADD+SG, MOE+WNORM).

A 10-token chat reply (`--reasoning off`, no `--ignore-eos`) accepted every draft: 5/5, mean length 2.00. Draft time was 66.6 ms for those 5 calls, 13.3 ms each. Eval was 461 ms / 10 tokens.

Forcing 64 tokens (`--ignore-eos`, log `htp-qwen-mtp3.log.err`):

- eval 3325.61 ms / 64 tokens = **52.79 ms/token = 18.94 tok/s**
- draft acceptance 0.75 (27 accepted / 36 generated), mean length **1.75**, acc at position 0 is the 0.75
- 36 verify steps for 64 output tokens

On the short run the draft clock was separate: 66.6 ms for 5 drafts, and the eval line was 461 ms for 10 tokens. That is about 79 ms of verify plus 13 ms of draft per step. The 64-token run did not print the draft clock (`-lv 3`). Its 52.79 ms/token is the whole step.

## 2026-09-26 16:25 — a plain 2-token forward already shares the weights

`llama-bench` on the current stack, one repetition, `OPBATCH=3072`, `-dev HTP0 -ngl 99 -t 4`. Log `bench-batch.log`.

| test | tok/s | ms for the whole forward | vs one token |
|---|---:|---:|---:|
| pp1 | 26.51 | 37.7 | 1.00 |
| pp2 | 36.23 | 55.2 | 1.46 |
| pp4 | 51.90 | 77.0 | 2.04 |
| pp8 | 46.56 | 171.8 | 4.56 |
| tg32 | 26.80 | 37.3 per token | decode |

Two tokens cost 1.46× one token, not 2×. Four tokens cost about 2× one token, which is 52 tok/s. The GEMV is already a small GEMM. pp8 slips back to 47 tok/s, so the sweet spot on this stack is around 4, not "as wide as possible."

The MTP verify at ~79 ms for two tokens is slower than this plain pp2 at 55 ms. That gap is the speculative schedule (KV already filled, a second context, graph shape), not a missing fuse. No further elementwise kernel is large enough to change the draft decision: the expert-weight multiply is 0.28 ms, inside the 0.15 ms spread of the 128-token runs.

## 2026-09-26 16:40 — wider drafts accept more and run slower

Same cli path, `--ignore-eos -n 64 --reasoning off`. One MTP layer, so extra draft tokens reuse that head serially.

| n_max | tok/s | ms/token | mean accepted length | per-position accept |
|---:|---:|---:|---:|---|
| 1 | 18.94 | 52.79 | 1.75 | 0.75 |
| 2 | 19.72 | 50.70 | 2.38 | 0.81, 0.58 |
| 4 | 13.81 | 72.39 | 3.00 | 0.81, 0.62, 0.38, 0.19 |

`n_max=2` is the least bad and still well under 27.6. Draft time scales with the serial loop: 13 ms per drafted token (658 ms / 26 calls for two tokens, 1062 ms / 21 calls for four). The verify batch grows with them and does not pick up the bench's 52 tok/s at width 4. Acceptance is not the problem. Each extra draft token is another full MTP decode, and the target then re-decodes the whole set to catch the hidden state up. That second context is the cost. Plain prefill already shares the weights.

## 2026-09-26 16:16 — router top-k fused, PHP held, wall did not

The 128-token pair was already on disk (`htp-qwen-n128.log` 27.61 tok/s, `htp-qwen-n128b.log` 27.51). Both print the PHP PDO fragment. They are not the same string after the INSERT: one adds an email column. That spread is 0.15 ms (36.21 vs 36.36). The expert-weight MUL is still one 6–7 µs op, about 0.28 ms across 40 layers, inside that spread. It was not fused.

The chain still outside that MUL is softmax of the 256 router logits, the desc argsort, and the top-8 gather, then the weight norm. On the 10:04 token that prefix is about 4 + 8 + 5 µs, ~0.7 ms across 40 layers. `HTP_OP_MOE_TOPK` does the same HVX softmax (scale 1, no mask), the same 256-wide desc bitonic argsort, and the same clamp-and-reciprocal on the 8 sorted values. The full index row is still stored, so the top-8 view stays at offset 0. A chain that is not one 256-wide row keeps `MOE+WNORM`. `try_fuse_rms_norm_scale` is still not called.

`-n 1 -lv 4` log `htp-qwen-topk-lv4.log.err` prints conv window, `MOE+WNORM`, ADD+N, ADD+SG, and `MOE+TOPK (n_exp 256 k 8)`. One earlier chain logged `moe-topk skip: logits` and did not take the new op. No RMS line. No `NO-SUPPORT`.

Same 32-token command (`GGML_HEXAGON_OPBATCH=3072`, profile off, `-n 32 -b 128 -ub 128 -p Hello --temp 0 --device HTP0 -ngl 99 -fit off --load-mode none`):

- `htp-qwen-topk.log`: PHP PDO fragment, ends at `$pdo = new PDO('mysql:host=localhost`, seed 11345361, **27.23 tok/s**, 36.72 ms/token. Graphs reused 30/32.

Same command with `-n 128`:

| log | seed | tok/s | ms/token | reused | text |
|---|---:|---:|---:|---:|---|
| `htp-qwen-topk-n128.log` | 2496835793 | 27.71 | 36.09 | 126/128 | PHP through the INSERT, then "However" |
| `htp-qwen-topk-n128b.log` | 2811515198 | 27.57 | 36.27 | 126/128 | matches `htp-qwen-n128.log` through "You have an" |

36.09 and 36.27 overlap 36.21 and 36.36. The mean moved 0.1 ms. That is not a new best. Do not replace 27.61. Leave the fuse on: the 32-token fragment is the PHP PDO text, and one of the two long runs reproduced the previous long text. The expert-weight MUL stays unfused. The 128-token spread is 0.18 ms, still not enough to see a 0.28 ms change.

Installed skel `pkg-wos/lib/libggml-htp-v81.so` is 839952 bytes, the v81 prefix build, catalog signed after the copy. `pkg-wos/bin/ggml-opencl.dll` is the stub, 102400 bytes. `HTP_OP_MAX_BUFS` is 64. The Q5_0 backup was not restored. ADD+SG, ADD+N, the conv-window fuse, MOE+WNORM, and the Q4_0 extended-index path were not reverted.

## 2026-09-26 17:05 — graph replay, 28.4 tok/s

The warm token was already one DSP batch. llama reuses the ggml graph (126/128) and Hexagon already caches the op list and the shape-derived kernel parameters on `graph->uid`. What still ran every token was `enqueue_op`: fuse matching, descriptor fill, then a copy of the op-node vector into the queue slot. Positions, the token id, and the KV index are written into the same buffers between tokens. `op_params` are fixed when the graph is built.

`GGML_HEXAGON_REPLAY` defaults on. After a graph flushes once, the host keeps the packed buffer, tensor, and op descriptors and submits that blob on the next compute with the same uid. A second flush in the same graph turns replay off for that uid. Profiling turns it off too, because the profile dump needs the op-node names. The non-profile queue push no longer copies the op-node vector.

Same 128-token command (`GGML_HEXAGON_OPBATCH=3072`, `-n 128 -b 128 -ub 128 -p Hello --temp 0 --device HTP0 -ngl 99 -fit off --load-mode none -lv 4`). Text matches `htp-qwen-n128.log` through "You have an".

| log | tok/s | ms/token | replay |
|---|---:|---:|---|
| `htp-qwen-replay.log` | 28.38 | 35.23 | captured uid 20, 1421 ops, 24 bufs, 963 tensors, then hit |
| `htp-qwen-replay2.log` | 28.40 | 35.21 | same capture, then hit |

35.23 and 35.21 sit about 1.0 ms under 36.21 and 36.36. That is the new 128-token number: **28.4 tok/s**. The warmup graph (uid 16, 2054 ops) and one 1481-op graph are captured and not replayed into the steady 1421-op decode graph. Host-only rebuild. The skel was not relinked. `pkg-wos/bin/ggml-opencl.dll` is still the stub.

## 2026-09-26 17:40 — warm-token map, f16 KV, prefix depth

Profile 1 on the current stack (`GGML_HEXAGON_PROFILE=1`, replay off while profiling, `-n 4 -lv 5`). Log `htp-qwen-profile-replay.log.err`. The last batch is the steady decode graph: 1421 ops, device 34214 µs, op sum 33364 µs, queue slack 850 µs, inter-op gap 27 µs. The DSP is packed. Wall on that run is 47 ms because the debug log is on; the unprofiled replay pair stays at 35.2 ms.

Where the 33.4 ms of op time sits:

| piece | ms | count | what it is |
|---|---:|---:|---|
| routed gate+up `MUL_MAT_ID_NX` | 4.95 | 40 | ~124 µs, about 76 GB/s on the active experts |
| GDN `attn_qkv` | 3.04 | 30 | ~101 µs |
| output `Q6_K` | 2.96 | 1 | 2048×248320, about 141 GB/s |
| routed down `MUL_MAT_ID` | 2.77 | 40 | ~69 µs |
| `ssm_out` | 2.17 | 30 | ~72 µs |
| `attn_gate` | 1.65 | 30 | ~55 µs |
| `GATED_DELTA_NET+CPY` | 1.63 | 30 | state update |
| conv window | 1.51 | 30 | already fused |
| shared-expert gate/up | 1.21 | 40 | ~30 µs |
| full-attn Q `Q6_K` | 1.03 | 10 | ~103 µs |
| GDN state `GET_ROWS` | 0.96 | 30 | ~32 µs, 2 MB memcpy of `cache_s` |
| router | 0.81 | 40 | ~20 µs |
| ops under 8 µs | 4.23 | 736 | launch |
| flash-attn, short context | 0.39 | 10 | grows once a prefix is cached |

Token embedding `GET_ROWS` is 59 µs. It is not a pile.

`llama-bench` on the same binary, f16 K and V, flash attention on, one stream, `OPBATCH=3072`, `-p 0 -n 64 -b 512 -ub 128 -t 4`. Log `bench-depth.log`.

| prefix already in cache | tok/s | ms/token |
|---:|---:|---:|
| 0 | 28.34 | 35.29 |
| 2048 | 26.41 | 37.86 |
| 8192 | 25.11 | 39.82 |

Empty-cache tg64 matches the 28.4 completion pair. A filled prefix costs about 2.6 ms at 2K and 4.5 ms at 8K. KV is already f16 (`llama-context` default, and `-ctk f16 -ctv f16` here). llama-server `cache_prompt` defaults on, which is the serving form of this depth test. Quantized KV is the wrong next step: on the Intel B70 Q4 lane, q8_0 KV is a few percent slower when the cache is empty and about half speed at 32K, because every token dequantizes the whole cache (neural.download KV precision guide, 2026-08-22 sweep). This machine is not short of memory for an f16 cache.

Single stream stays the target. Four virtual sessions on this one NSP decoded at 14.66 tok/s. Steve's multi-user win on Arc is one weight read shared by a batch, and his XPU graphs are captured only at specific batch sizes; an uncaptured batch is slower than one user. The captured graph here is the batch-1 decode graph.

## 2026-09-26 18:07 — GDN state gather changed the tokens

The 17:40 token spends 0.96 ms on 30 `GET_ROWS` of the 2 MB GDN `cache_s`. `try_fuse_gdn_state` drops that gather and points `GATED_DELTA_NET` at the cache plus the i32 index. `op_params[3]` is the row stride (2097152). The decode kernel adds `index * stride` to the state DMA. Address equality with `SSM_CONV+WIN` is not a use; the compute buffer reuses that slot. The matcher only rejects a real view of the gather.

One 128-token run, same command as the 28.4 pair (`GGML_HEXAGON_OPBATCH=3072`, `-n 128 -b 128 -ub 128 -p Hello --temp 0 --device HTP0 -ngl 99 -fit off --load-mode none`). Log `htp-qwen-baseline2.log`.

- the fuse line fired: `fused GDN state gather (row 2097152 bytes)`
- steady graph is 1391 ops (1421 minus the 30 gathers). Replay hit. Graphs reused 126/128
- decode: 4477.73 ms / 128 = **34.98 ms/token = 28.59 tok/s**
- the text is the aws-cdk VPC fragment, not the PHP PDO fragment. Seed 1623861709

28.59 is faster than 28.4 and is not the rate to keep. The matcher is not called. `GGML_HEXAGON_FUSE_GDN_STATE` stays defined and unused. The installed skel is the previous PHP skel again, `pkg-wos/lib/libggml-htp-v81.so`, 839952 bytes, catalog signed after that copy. A 128-token rerun on that stack, log `htp-qwen-gdn-off.log`: PHP PDO through "You have an", seed 2334271453, **27.76 tok/s**, 36.02 ms/token, replay hit 1421 ops, graphs reused 126/128. No GDN-state line. That is slower than the 35.2 ms pair and does not replace 28.4. `pkg-wos/bin/ggml-opencl.dll` is the stub, 102400 bytes. `HTP_OP_MAX_BUFS` is 64. The Q5_0 backup was not restored. The RMS matcher stays off.

## 2026-09-26 18:00 — GDN state gather did not ship

The 0.96 ms gather is a real copy: 30 `GET_ROWS` of `cache_s`, about 32 µs each, one 2 MB row. The GDN kernel already DMAs that row. Dropping the gather is the right combine, and it did not land.

The host matcher found the gather, then refused to drop it because `SSM_CONV+WIN` in the same batch has an input at the same data address. That is the compute buffer recycling a dead tensor's slot, not a second reader of the state. A pointer compare is the wrong test. The next attempt has to match the tensor object.

A kernel that selected the row from `op_params[3]` never ran, because the matcher returned first. An earlier version of that kernel moved a field inside `htp_gdn_kernel_params` and the one-token graph came back `NO-SUPPORT`. That field is reverted. `gated-delta-net-ops.c` and `.h` match the previous good skel.

Installing the skel has a trap. `cmake --build` of the v81 prefix writes `htp-v81-prefix\src\htp-v81-build\libggml-htp-v81.so`. The parent step can print `ninja: no work to do` and leave `build-wos\ggml\src\ggml-hexagon\libggml-htp-v81.so` at an older file (this time 835400 bytes from 14:13, which does not know the later opcodes). Copy the prefix so, then resign `libggml-htp.cat`, then copy both into `pkg-wos\lib`. The restored so is 839952 bytes. `htp-qwen-baseline2.log` was later overwritten by the fuse-on aws-cdk run. The PHP confirm after the fuse was turned back off is `htp-qwen-gdn-off.log`. The 28.4 pair stands.

## 2026-09-26 18:40 — expert weight mul folded into the add, PHP held, wall did not

The 17:40 token's next legal pile after the GDN state gather is the expert-weight mul. Forty `MUL`s of `ffn_moe_down` by the eight normed weights are 278 µs, and 256 µs of that is under 8 µs. The q/k `RMS_NORM`+`SCALE` pair is larger (634 µs) and stays off. The state gather stays off.

`HTP_OP_ADD_NW` runs when the eight-row `ADD+N` parent is that mul. Sources are the packed down rows and the `[1, 8]` weight row. The skel does `hvx_mul_scalar_f32_aa` per row, then the same `hvx_add_f32_aaa` left fold as `ADD+N`. The weighted rows are not stored. Decode only: `ne[2] == 1`.

Same 128-token command (`GGML_HEXAGON_OPBATCH=3072`, `-n 128 -b 128 -ub 128 -p Hello --temp 0 --device HTP0 -ngl 99 -fit off --load-mode none`). Log `htp-qwen-addnw.log`.

- the fuse line fired: `fused expert weight mul into ADD+NW (n_src 8 n_elem 2048)`. No `add-nw skip`. No GDN-state line. No RMS line
- steady graph is 1381 ops (1421 minus the 40 muls). Replay captured uid 20, then hit. Graphs reused 126/128
- decode: 4511.31 ms / 128 = **35.24 ms/token = 28.37 tok/s**
- the text matches `htp-qwen-replay.log` through `You have an` (the PHP PDO fragment)

35.24 does not beat 35.21 and 35.23. One run is not a pair. **28.4 stands.** The matcher stays on because the text is PHP.

## 2026-09-26 19:10 — MTP catch-up skip changed the second sentence

Codex (read-only, gpt-5.6-sol) on `common_speculative_impl_draft_mtp`: `process()` row k is `(h[k-1], x[k])`, logits for `x[k+1]`. The next `draft()` seed is `(h[last], id_last)` with `id_last == x[last+1]`, logits for `x[last+2]`. Sampling `process()`'s last-row MTP logits is the wrong draft. The plain decoder graph must not grow a blk.40 tail just because the GGUF contains MTP weights. The step that would actually share the HTP submit is an opt-in verification graph that appends the MTP block to the target forward, then skips the separate catch-up `llama_decode(ctx_dft)`.

A smaller host shortcut was measured and reverted. `draft()` already decodes `id_last`. The server then `seq_rm`s that position and `process()` decodes it again. Keeping that position and skipping it in the catch-up fired: `mtp keep pos 13 id 9419`, then `skipped 1, keeping 1`. Two temp-0 `Hello` runs (`htp-qwen-mtp-keep.log`, `htp-qwen-mtp-keep2.log`) printed "questions" where `htp-qwen-mtp3.log` printed "a question", and generation stayed 19.4–19.5 tok/s against the old short-reply 18.9–19.5. The source is back to `npu-htp-decode` without that skip, and `llama-cli` was rebuilt from it. Do not sample the catch-up logits. Do not keep a draft-context write that `process()` would have recomputed from the target hidden state. The installed skel is the v81 prefix link, `pkg-wos/lib/libggml-htp-v81.so`, 844176 bytes, catalog signed after that copy. The file beside the build-tree catalog is still the stale 839952-byte so from 17:52. A copy of the previous PHP skel is `npu_campaign/libggml-htp-v81-php-839952.so`. `pkg-wos/bin/ggml-opencl.dll` is the stub, 102400 bytes. `HTP_OP_MAX_BUFS` is 64. The Q5_0 backup was not restored. Output weight was not requantized.
