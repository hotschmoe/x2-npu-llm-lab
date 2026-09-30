# decode2 campaign log (append-only)

Agent decode2. Worktree `llama.cpp-hexagon-decode2` (branch `decode2` from `npu-htp-decode` ee4381b). Build: `build-decode2.cmd` (host, then
`build-decode2-htp.cmd` for the v81 skel: the first configure leaves the ExternalProject unconfigured), package: `pkg-decode2.ps1 -Name <x>` -> `pkg-decode2-<x>`.

## 2026-09-29 (start)  Repeat protocols for the plain-path instability

Two scripts, both hash the FULL-vocab logits of each row:
- `repeat-plain.ps1` (+ `repeat_rows.py`): N separate llama-cli processes (~55 s each, mostly load).
- `repeat_server.py`: ONE llama-server process, N identical `/completion` requests with `cache_prompt=false`, temp 0 (server up ~40 s, then seconds per
  request). Prints per-row distinct-hash counts. Takes the NPU lock.
Baseline (pkg-decode2-base = unmodified npu-htp-decode source, host 327D96FA, skel 8874861A):
- llama-cli, Hello, 2 runs: all 3 rows differ between the two runs.
- server, 15-token prompt, n_predict 3, 6 requests in ONE process: 5 distinct traces out of 6 (rows 0,1,2 all vary; row 0 = the prompt row).
=> the instability is not an address-layout effect: it varies within a process for the identical request (a timing race / cache-state race), and it is
   cheap to measure (~1.5 min for N=10).

## Sweep 1 (server repeat, 15-token prompt, n_predict 3, N=10 requests in one process, baseline build)
| knob | distinct request traces / 10 |
| --- | ---: |
| none (baseline, N=6 earlier) | 5/6 |
| GGML_HEXAGON_OPBATCH=1 (one op per batch: whole-cache flush+invalidate between ALL ops) | **1/10 (stable)** |
| GGML_HEXAGON_NHVX=1 (single HVX worker) | 10/10 |
| GGML_HEXAGON_NHMX=1 | 10/10 |
| GGML_HEXAGON_OPFUSION=0 | 10/10 |
| GGML_HEXAGON_REPLAY=0 | 10/10 |
| GGML_HEXAGON_ALLOC_FILL=1 | 10/10 |
Conclusion: not an inter-thread race among HVX workers, not fusion/replay related, not uninitialized-allocation contents; it disappears when the whole cache is
flushed+invalidated between every op => inter-op cache-state hazard (or timing that a batch boundary resets).

## Diagnostic DSP builds (all OFF by default; `GGML_HEXAGON_DIAG_DEFINES` cache var in htp/CMakeLists.txt, `d2-skel.cmd`, `d2-variant.ps1`, `d2v.sh name "DEFS"`, `d2rs.sh name n`)
`GGML_HEXAGON_OPHASH_DIAG` (per-op dst hash returned in the profile record, needs GGML_HEXAGON_PROFILE=1): ophash package with profile: **6/6 identical traces (stable!)**.
Controls (10 requests each, server repeat, 15-token prompt):
| DSP variant | distinct traces / 10 |
| --- | ---: |
| ctrl (source unchanged, diag code compiled out) | 10 |
| ctrl + GGML_HEXAGON_PROFILE=1 (host verbose log, replay off) | 9 |
| DST_FULL_AFTER: clean+invalidate every dst range after every op | 10 |
| ... only dst <= 4096 B | 10 |
| ... only dst > 4096 B | 10 |
So flushing+invalidating outputs after each op does NOT stabilize; the ophash diag additionally reads every dst through the cache (slow, ~MBs per op) and IS stable, as is OPBATCH=1.
Working hypothesis: a timing race (something asynchronous outlives an op: DMA, HMX queue, ...) that op-to-op slack hides. Next: DELAY_AFTER_US probes.

## Key finding: READING the outputs after each op stabilizes it (DELAY does not)
| DSP variant (15-token prompt, N=10 requests, one server) | distinct traces / 10 |
| --- | ---: |
| DELAY_AFTER 5 us / 50 us spin after every op | 10 / 7 |
| TOUCH_AFTER=7: clean+inv dst, read every word through the cache, clean+inv dst (after every op) | **1 (stable)** |
| TOUCH_AFTER=2: read every word of every dst through the cache after every op (no flush at all) | **1 (stable)** |
Reading the freshly written outputs through the cache is enough. (flush-only 'afterall' stays unstable.) Bisecting by dst size class next.

## Size-class bisect of the TOUCH_AFTER=2 cure, and negative controls (N=10, one server each)
| variant | distinct traces / 10 |
| --- | ---: |
| touch dst <= 4096 B only (t2s) | **1 (stable)** |
| touch dst 4097..65536 B only (t2m) | 10 |
| touch dst 64K..1M only (t2l) | 7 |
| touch dst > 1M only (t2x) | 8 |
| touch dst <= 512 B only (t2a) | 10 |
| `syncht` at the end of every hex_l2flush (flushsync) | 9 |
So the cure lives in dst tensors of 513..4096 B (the router/gating chain: softmax(256xT f32 = 4 KB), argsort ids, top-k weights ...). Note repeated 'attractor'
hashes across unstable runs (e.g. row0 d5291acc02 in two independent runs) vs a different stable value (touch: e322ed5861).

## CORRECTION (13:15): the "stable" touch/ophash variants were all-NaN outputs; a valid reference exists
The variants I called stable (TOUCH_AFTER=2/6/7, ophash+profile, t2b) produce **all-NaN logits** deterministically (repeat_analyze.py now prints NaN counts and
top tokens) - reading small dsts after each op leaves stale lines that the consumers then read: itself a proof that consumers read small tensors through the
cache without a sufficient invalidate. Not a cure. (Also fixed: repeat_server hung on p.wait after terminate -> Traceback before the analysis; now taskkill.)
The valid reference: GGML_HEXAGON_OPBATCH=1 (whole-cache flush+invalidate between every op) is stable, non-NaN, and gives row hashes
row0 d5291acc02 (top 13), row1 073f8ae694 (top 271), row2 edd4c6a51f (top 248068) 4/4. These are exactly the 'attractor' hashes that recur in the unstable
baseline runs (e.g. flushsync run 5 and 10, baseline run 4 and 6), so **the reference hashes = the correct deterministic result; instability = deviations from it**.
The acceptance test for any fix is therefore: every request's rows equal d5291acc02 / 073f8ae694 / edd4c6a51f (prompt: 'The quick brown fox jumps over the lazy dog near the river bank today', -c 1024 -b/-ub 128).
Next: COHDET_DIAG (before each op, hash every activation src <=1MB through the cache, clean+invalidate, hash again; A!=B with tracker-clean = a hole in the dirty tracker).

## 14:10  ROOT CAUSE of the plain-path instability: the cache-maintenance granule is 64 B, `hex_l2flush` steps 128 B
Evidence chain (all N=10 requests in one server, 15-token prompt, valid non-NaN logits):
1. Reference: OPBATCH=1 (whole-cache flush+invalidate between ops) -> 4/4 identical, rows d5291acc02/073f8ae694/edd4c6a51f. Unstable baseline runs often land ON these hashes
   (they are the correct deterministic result) or on nearby noisy variants.
2. COHDET_DIAG (before each op compare the cached view of every activation source <= 1 MB with the DDR view after clean+invalidate): a handful of mismatches per 9 batches
   that the dirty tracker did not know about (all MUL_MAT_ID_NX / MUL_MAT_NX / GET_ROWS activation sources), and the mismatching bytes are always ONE 64-byte HALF of a 128-byte
   pair (diff offsets 3392, 5056, 192, 2368 = 128k+64, 16 words each; cached value != DDR value, both plausible activations = stale previous-layer data at the recycled address).
3. => the hardware invalidates/cleans in 64-byte units per `dccleaninva`, but `hex_l2flush` (and the tracker's `HEX_L2_LINE_SIZE 128`) issue one instruction per 128 bytes,
   leaving every second 64 B granule untouched. Stale/dirty second halves survive a 'flush'. Whether they matter depends on what the (allocator-reused) address held before,
   hence the intermittency.
4. TEST: `GGML_HEXAGON_DIAG_FLUSH64` (dccleaninva every 64 B in hex_l2flush): **10/10 requests bit-identical and equal to the OPBATCH=1 reference on all three rows** (baseline: 10/10 distinct).
Everything else measured earlier is now explained: OPBATCH=1 hides it (whole-cache ops), flush-after-dst 'afterall' did not help (same 128 B stride), syncht no help, delays no help, and
the previous campaign's 'actsrc-small' rescue used qurt_mem_cache_clean(FLUSH_INVALIDATE) which flushes every line (correct granule) whereas its inline `hex_l2flush` variant failed.

## 14:35 Fix verification (production hex_l2flush with HEX_CACHE_GRANULE=64; commit 84c8650 on branch decode2)
| build | prompt | requests | distinct traces | == OPBATCH=1 reference |
| --- | --- | ---: | ---: | --- |
| fix64 (64 B steps) | 15-token 'quick brown fox' | 20 in one process | **1** | yes, all 3 rows |
| fix32 (32 B steps, HEX_CACHE_GRANULE=32) | same | 10 | 1 | yes |
| fix64 | ~50-token 'hash table' prompt | 10 | **1** | yes, all 3 rows (29d68b3683 / a832f76bf0 / ecbbbdc9a4) |
| unmodified base (ctrl) | quick brown fox | 10 | 10 | rarely |
32 B and 64 B steps behave identically here; 64 B is the default (dccleaninva granule on this core; ssm-conv.c already uses 32 B dcinva steps on its scratch).
Queued: ABBA speed base vs fix64 (plain and MTP), prefill PPL 12.1307/13.3331, MTP gates (now plain == MTP is checkable).

## 15:45 Gates on the fix (fix64 = decode2 84c8650 host+skel)
- Prefill PPL (run-prefill-ppl.ps1, 4x512): **12.1307 +/- 1.11278** (exact); 16 chunks: **13.3331 +/- 0.63451** (exact). The prefill path is bit-identical to before the fix.
- NPU-lock fairness: run scripts polled the lock every 5 s and were starved for tens of minutes by back-to-back holders; `npu-lock.ps1` now polls every 200 ms and honours
  NPU_LOCK_HELD=1; `lockrun.py -- cmd` takes the lock with 10 ms polling and runs one command under it (used by qq.sh).
- The `npu-htp-decode` branch has not moved (ee4381b), so 'merge npu-htp-decode into decode2' is a no-op: decode2 = ee4381b + 84c8650.

## 16:10 MTP/plain logit gates on fix64 (Hello, 128 tokens, full-vocab rows, compare_logits_traces.py --require-identical)
- ref-plain vs cand-plain (two independent plain runs of the same build): **128/128 rows bit-identical** (this is the check that failed at the prompt row before the fix).
- ref-mtp (plain MTP) vs cand-mtp (fused catch-up + draft vocab 65536): 128/128 identical.
- **cand-plain vs cand-mtp: 128/128 identical; ref-plain vs ref-mtp: 128/128 identical** (plain==MTP target identity, previously only guaranteed 'when plain is stable').
- mtp-gate.ps1's 3rd comparison (--context-index 1 --draft-greedy between the non-fc/non-dv ref arm and the fc/dv cand arm) aborts the script (different draft row counts 74 vs 72,
  by construction of the draft-vocab cap) so the trace files were left behind; compared manually and deleted.

## 16:20 MTP/plain full-vocabulary logit gates, 3 prompts x 128 rows (fix64, all comparisons --require-identical): ALL PASS
| prompt | plain A vs plain B | MTP(plain cfg) vs MTP(fc+dv65536) | plain vs MTP (fc+dv) | plain vs MTP (plain cfg) | acceptance ref / cand |
| --- | --- | --- | --- | --- | --- |
| Hello | identical | identical | identical | identical | 52/74, 54/72 |
| sky prose | identical | identical | identical | identical | 59/68, 59/68 |
| hash-table | identical | identical | identical | identical | 54/72, 55/71 |
i.e. the plain path is now run-to-run deterministic and the MTP target logits equal the plain logits bit for bit, on every row.

## 17:15 Speed runs ABORTED (laptop moved to battery); no speed numbers are recorded
The four plain runs that completed before the switch (base 29.15 on AC earlier; then fix64 27.05, fix64 21.21, base 21.13 on battery) are NOT valid A/B data and must not be quoted.
Also note run-mtp-speed refuses existing labels, so a re-run of a queue script needs fresh labels (qq_pending.sh uses d2s2/d2h2/d2b2).
PENDING (need AC power, `nohup ./qq_pending.sh > qq_pending.out &`, every run takes the NPU lock via lockrun.py):
 1. interleaved untraced ABBA, plain and MTP (fc + dv65536), base vs fix64 (Hello 256): cost of the 2x flush instructions (expected to be tiny: flushes are per-op ranges, 64 B steps).
 2. item D: PPL (already scripted in qq_pending.sh section 4: Q6_K head vs Q4_0 head model `..-mtp-f32-headq40.gguf`, made by requant-head.exe) + decode ABBA plain/MTP with the Q4_0 head.
 3. item B garbage hunt, base vs fix64, slots=4 (sections 5).
 Not attempted: item C (recurrent-state snapshot DMA, 2-query FA, small-op overhead, OPPOLL power policy with npu-watt): all need timing measurements.
