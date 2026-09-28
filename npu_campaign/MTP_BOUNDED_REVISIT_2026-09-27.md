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

## Deeper continuation: plain failure survives fixed graph descriptions

The user resumed work after the bounded checkpoint. On the same D2D baseline, a second plain64 pair with fusion and replay both off differed already at the prompt logit row (NMSE `0.00880110055`, max `0.750018716`) and first selected token step 19. This shows neither fusion nor replay is necessary for a plain-path discrepancy, but changing both policies also changes the graph and trajectory; it does not isolate a root cause.

Another plain64 pair with `GGML_HEXAGON_VERBOSE=1` kept fusion/replay on and retained no tensors. Its first bit difference moved to step 31 and its first selected-token split to step 61, so logging did **not** suppress the failure. The two logs have exactly the same 7,604 `execute-op` descriptions and 3,922 `add-tensor` descriptors after normalizing process-specific absolute addresses; all logged tensor names, buffer IDs, offsets, sizes, flags and shapes match. Graph replay op counts also match. The mismatch is therefore not explained by a *visible* op sequence or tensor-layout change. This does not rule out cache state, unlogged descriptor content, async timing, or an arithmetic race. Logs: `mtp-deep-plain-fr0-{a,b}64` and `mtp-deep-verbose-{a,b}64`.

## Whole-cache diagnostic gates, not a production fix

`run-mtp-wholecache-revisit.ps1` serially installs the archived signed DSP `E9560F297EFB470BC70A61615BE6130F0E35CD6366FDE52A84F2C2D588FC5160` with its matching catalog, verifies the frozen C9E4 host and expanded GGUF hashes, runs plain A/B then real MTP A/B, and restores the installed D2D DSP and 5CB7 catalog in `finally`, checking both restored hashes. This archived diagnostic flushes/invalidates the **whole** DSP data cache and clears its dirty tracker before every operation. It does not change model values or graph source, but it also changes barriers and timing, so it is not proof of a particular faulty cacheline. All runs exited 0 and the baseline was independently hash-checked after each four-arm session.

| Prompt and length | Plain A/B target rows | MTP A/B target rows | Plain A vs MTP A | Draft acceptance |
| --- | --- | --- | --- | --- |
| `Hello`, 64 | All 64 bit-identical | All 64 bit-identical | All 64 bit-identical | 27/36 in both MTP runs |
| `Hello`, 128 | All 128 bit-identical | All 128 bit-identical | All 128 bit-identical | 52/74 in both MTP runs |
| `Explain why the sky appears blue in three clear sentences.`, 128 | All 128 bit-identical | All 128 bit-identical | All 128 bit-identical | 59/68 in both MTP runs |

These are full target-vocabulary logit bits **and** selected-token checks, not transcript-only matches. The `Hello` cache-control trajectory also matches the stable D2D plain64 trajectory bit-for-bit. The 128-token cases exercise acceptance and rejection under two distinct prompts. No CPU equivalence, arbitrary-context rollback-state equivalence or broad-corpus guarantee follows. Traced cache-control decode rates ranged about 16.85-18.94 tok/s and are not performance qualification; per-op whole-cache flushing is deliberately expensive. Raw artifacts are local under `mtp-deep-wc64-*`, `mtp-deep-wc128-*`, and `mtp-deep-wc-prose128-*` and are not committed.

### Draft trace correction

The MTP draft context writes top-10 candidate rows to the logit trace. Its `logits-trace-selected` event records the return value of `common_sampler_sample`, but `common_speculative_impl_draft_mtp::draft()` ignores that return and instead drafts `common_sampler_get_candidates(smpl, true)->data[0].id`, the highest-probability candidate after sorting. The selected trace can therefore differ between runs even when the draft actually used and the candidate/logit row are identical. A naive application of the target comparator to draft context reported a false selected-token divergence at draft row 6 for `Hello` and row 4 for prose, although the candidate lists and logits were bit-identical. Source: `common/speculative.cpp` MTP constructor and `draft()`; `common/sampling.cpp` sampling and `get_candidates()`.

`compare_logits_traces.py --context-index 1 --draft-greedy --require-identical` now checks the draft candidate IDs, their logit bits, and the top candidate while deliberately ignoring that unused sampled-ID event. It passes all 74 draft rows across the `Hello` 128 MTP pair and all 68 across the prose 128 pair; it also passes the 36-row original D2D MTP64 repeat. This is an exact *draft-candidate trace* repeat gate, not a direct capture of every hidden/KV state cell or a proof that all draft arithmetic is deterministic on other prompts.

Next: distinguish the whole-cache call's flush/invalidate, implicit synchronization, and dirty-tracker reset effects with narrower OFF-default signed interventions, each requiring the same full target/draft gates. Reproduce the failure on the matching untreated baseline before calling any narrow arm a rescue. Do not adopt the all-cache diagnostic as a speed path.
