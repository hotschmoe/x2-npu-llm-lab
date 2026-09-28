# Bounded MTP revisit: matched target, intermittent plain baseline

Date: 2026-09-27. This is a diagnostic milestone, not production MTP qualification or a speed promotion. No kernel, model, or installed binary was changed.

## Frozen protocol

- Installed `llama-cli.exe` SHA256 `DC30C4365E3BCC3757E88B863A5F5FE454E5E9C9E8D85456565D416F1DA06C4F`; `llama.dll` `359708332567BB4BDB2A0CCD1170261CF3BB6062ED9847E174A4A42AA0088FF8`; `ggml-hexagon.dll` `C9E4D8ADE6884DA62DF7A597DCEAC6582AC036236C06134231A405BAF8E6399E`; DSP `libggml-htp-v81.so` `D2D98EDEA1BDFD8652EE0766C4B7707AE1209E335164E6115478091C304B68CD`; catalog `libggml-htp.cat` `5CB7F7EF8F6D6E88C5247D4EB5AC43CA527D7DF15BA66C547CBC25A3507CE09F`. These are the restored later campaign baseline, **not** the experimental EFC DSP used by earlier MTP studies. Source worktree remains dirty; installed-file hashes are the run provenance.
- Losslessly expanded MTP GGUF SHA256 `E8E27D9475056E03B5D663C032B0966E00E68DB6314C07A883F791B9F6297746`, size 20,837,295,744 bytes. Plain and MTP both load this exact file. It is a derived model with two BF16 draft gate/router tensors expanded to value-preserving F32, not the untouched default GGUF.
- Five separate process launches, serial and without an overlapping inference/build: `Hello`, fixed seed 1234, temp 0, 64 generated tokens, HTP0, context 512, batch 128, ubatch 128, op batch 3072, replay and fusion on, router HVX, GDN selector 1, attention selector 1, MTP draft max 1 when enabled. `LLAMA_FUSE_MTP_VERIFY` and experimental KV-only catch-up were off. All five CLI runs returned 0 with complete 64-row target logits and selected-token records. Logs are `mtp-revisit-0927-{plain-a64,plain-b64,real-a64,real-b64,plain-c64}.{detail.log,logits.bin,log,log.err}` in this directory; raw captures are local artifacts, not committed.
- Compared the complete target full-vocabulary logit rows and selected tokens using `compare_logits_traces.py --require-identical`. Its MTP comparison reads the first/target context only; draft-context logits and internal rollback state are **not** qualified by this gate.

## Results

| Pair | Complete target-logit and selected-token result |
| --- | --- |
| plain A vs plain C | All 64 rows bit-identical; all selected tokens equal. |
| real MTP A vs real MTP B | All 64 target rows bit-identical; all selected tokens equal. Both accept 27/36 drafts, including 9 rejections. |
| plain A/C vs real MTP A | All 64 target rows bit-identical; all selected tokens equal. Real verification, acceptance, rejection and catch-up were exercised. |
| plain B vs plain A/C or real MTP A | First logit-bit difference at zero-based step 10 (plain A vs B NMSE `0.00054645482`, max absolute difference `0.223294973`); first selected-token difference at step 57, `198` versus `271`. The two streams need not share input prefixes after step 57, so later differences are not interpreted. |

Reported traced-run decode rates were plain A/B/C `27.37/27.06/27.10` tok/s and real A/B `24.07/24.02` tok/s. Logging and a five-run non-ABBA order make these **diagnostic only**, not an accepted throughput comparison. They do not suggest that MTP is faster here. Prompt times were roughly 266-301 ms and are separate from decode.

## Decision and next gate

For this exact model/binary/policy/prompt, real MTP has demonstrated 64-token target-logit equality against two matching plain runs and an independent MTP repeat. The one differing plain run shows an intermittent **plain-path** numerical failure before its selected-token split. Therefore, a plain/MTP mismatch on this policy cannot be attributed to speculative verification without first controlling that baseline; nor does equality on this one prompt establish global MTP correctness. This reproduces the earlier pattern under the later D2D baseline rather than proving that the older EFC observations transferred unchanged.

Stop the bounded MTP revisit here. The next correctness investment is an allocation-preserving first-divergence probe for the plain step-10 logit variation, with a same-run/identical-layout plain repeat gate. Existing tensor-retention probes alter graph lifetime and can suppress the failure, so any new probe must show that the uninstrumented plain failure still occurs. Only then extend matched plain/MTP target gates to 128 tokens and multiple prompts, separately audit draft logits plus KV/recurrent accepted/rejected-prefix rollback, and finally measure untraced paired speed. Do not enable fused MTP verify or KV-only catch-up based on this result.
