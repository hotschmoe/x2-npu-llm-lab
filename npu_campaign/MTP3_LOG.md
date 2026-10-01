# MTP3 log (append-only) - agent mtp3
Goal: Qwen3.6-35B-A3B MTP decode from ~30 tok/s (dmax 1, acceptance 71%) to >35 tok/s, and robustness.
Worktree `llama.cpp-hexagon-mtp3` (branch `mtp3` from npu-htp-decode ce75353), build dir `build-mtp3`, packages `npu_campaign/pkg-mtp3-*`.
Baseline package for all base arms: installed `llama.cpp-hexagon/pkg-wos` (== `pkg-merged2-m1`, host 47D07A15, skel 27ADA976).

## 2026-09-29 21:30 Plan / analytic ceiling (from MTP_SPEEDUP_LOG numbers, to be replaced by fresh measurements)
tok/s = (1 + a) / (V + D + H), a = mean accepted drafts per step (dmax 1: acceptance rate).
Known: V(T=2) DSP ~49 ms (wall ~52), D (draft+folded catch-up) wall ~3.3, H ~1.7 => step ~57 ms; plain step 35.6 ms.
- a = 0.71: 1.71/57 = 30.0 tok/s (matches measurement). To reach 35 at a = 0.71 the step must be <= 48.9 ms (-8 ms).
- a = 1.00 at the same step cost: 2/57 = 35.1 tok/s: at dmax 1 the ceiling is ~35 even with perfect acceptance. So acceptance alone
  cannot deliver >35; step cost must drop (spin poll: -6% NPU time; T=2 verify) or dmax>1 with good acceptance chains.
- Break-even draft acceptance for dmax 1 vs plain: (1+a)/(V+D+H) > 1/35.6 => a > 0.52.

## 2026-09-29 23:30 Measurements so far (installed pkg-wos == pkg-merged2-m1; fresh profile on this build, Hello, dmax 1, fc + dv65536)
Device profile (GGML_HEXAGON_PROFILE=1, m3p-plain / m3p-mtp): plain M=1 batch 34.7 ms (33.8 sum of ops, 1384 ops); MTP verify M=2 49.7 ms (48.9, 1414 ops);
MTP draft graph 2.58 ms (40 ops: head MUL_MAT 0.72 with dv65536, TOP_K 0.25, experts/attn ~1.4); MTP-profile: draft decode 3.1-3.5 ms wall.
Verify M=2 minus M=1 = +15.0 ms (per class): experts gate/up NX +3.8 and down +2.0 (5.8 total: 16 distinct experts, ~77 GB/s effective vs 141 GB/s streaming
on the 418 MB head: expert kernels are the biggest non-fundamental cost), GDN+CPY +2.1, dense MUL_MAT +2.7 (attn_qkv +0.75, head +0.6, attn_gate +0.7),
conv window +0.5, FA +0.6, RMS/SILU/SWIGLU/ROPE/SOFT_MAX etc. ~+3 (M=2 costs ~1.4-2x per small op).
Head (output.weight Q6_K, 418 MB): 2.95 ms at M=1, 3.53 ms at M=2 => a Q4_0 head (286 MB) would save ~1.0-1.1 ms per verify.

SPOOF (not a result): `--spec-synth-rates 1.0` (built-in synthetic acceptance = 100%, dmax 1, block poll, fc+dv): 35.34 tok/s (acc 127/127).
This is the ceiling of dmax 1 at today's step cost: even perfect acceptance barely reaches 35. Measured acceptance 0.71 => ~30.
Step cost model (wall): step = V + D + H with V(T=2) ~52 ms (DSP 49 + ~3 wake), D ~3.3 (DSP 2.4-2.6), H ~1.7; tokens/step = 1 + a.
=> the lever that matters is step cost (V), not acceptance: d(tok/s)/d(a) = +0.5 tok/s per +0.01... e.g. a 0.71 -> 0.85 gives only +5 tok/s...
   (1.85/57 = 32.5), and a step cost 57 -> 50 ms at a = 0.71 gives 34.2.

## 2026-09-29 23:45 Offline n-gram (prompt lookup) study on recorded agentic sessions (npu_campaign/mtp3_tools/ngram_sim.py, pitest/runs/qwen-full, 33 sessions, 19k generated pseudo-tokens)
Regex pseudo-tokens (not the model tokenizer); wall-cost model V(T) = llama-bench pp1..8 (35.3/53.5/65.5/76/82/88/94/100.5 ms). Fallback to MTP (a = 0.71) when no n-gram match.
N-gram key length N / draft length M -> ngram draft acceptance and modeled tok/s (plain 28.1, MTP-only 29.2 in this model):
N=4: M=1 78% 30.3, M=3 63% 31.7, M=5 51% 32.3, M=8 40% 32.0; N=6: M=5 59% 32.0; N=8: M=5 67% 31.6; N=12: M=5 71% 30.8.
=> on tool-call/thinking traffic n-gram+MTP is worth about +8-10% over MTP alone (before the extra costs of T>3 unfused verify and MTP catch-up after n-gram steps).
Current fusions support T <= 3 in the conv window (3 snapshot outputs, HTP_OP_MAX_OUTPUTS = 4), so K <= 3 (n_rs_seq <= 2): n-gram draft length 2 is directly testable.

## 2026-09-30 06:00 Results batch 1 (all runs interleaved inside one NPU hold per block, untraced, 256 tokens, installed pkg-wos unless noted; machine ~6% slower than yesterday's session: plain 26.4-26.5 vs 28.1)
Note: the shared lock queue was heavily contended (upstream/ops650/pitest holds of 20-60 min); I run blocks of 4-5 runs per lock hold (lockrun3.py, 10 ms polling).
1. Baseline ABBA block (Hello): plain 26.46, MTP dmax1 fc+dv65536 28.94 (+9.4%, acc 106/149), plain+OPPOLL=1 28.64, MTP+OPPOLL=1 31.65 (+19.6% vs block plain; +10.5% vs spin plain).
2. fastrpc QoS (new env GGML_HEXAGON_QOS=<enable>,<latency_us>; the existing code set enable=1 with an UNINITIALIZED latency): def 26.44, pm-qos(1,100) 26.28, adaptive(2,100) 26.43,
   poll-qos(3,10000) 26.57, OPPOLL=1 28.60 (plain decode, one round). => no FastRPC QoS mode reproduces the spin gain (the dspqueue path is not governed by RPC QoS);
   only keeping the host thread awake does (+8%). Decision: QoS code stays env-gated, default unchanged (init latency to 0 instead of garbage).
3. Acceptance depends heavily on content (greedy, dmax 1, fc+dv): Hello 71% (106/149), sky prose 85%, JSON tool-call 83% (116/139), code (interval merging) 95% (124/131),
   copy-heavy (repeat a config 3 times) 97% (125/129). MTP-vs-plain: code 32.57 vs 26.37 (+23.5%), copy-heavy 32.94 vs 26.63 (+23.7%).
4. SPOOF ceiling (--spec-synth-rates): acceptance 100% -> 35.34 tok/s; 85% -> 32.92 (actual 0.83) at today's step cost (block poll). Matches the analytic ceiling (2/57 ms).
5. n-gram+MTP (ngram-simple n=4, m=2, need_n_rs_seq raised to 2): upstream common_ngram_simple_draft never drafted when size_m < size_n (copy_max < n_draft_min) -> fixed to copy_max < 1 (pkg c).
   With the broken matcher (pkg a, 0 ngram drafts) rep 32.85 vs MTP 32.94; code (pkg c matcher) mng 32.21 vs MTP 32.57: n-gram adds nothing when MTP already accepts 95-97%.
6. Draft-max 2 gated by draft-head confidence for the 2nd token (LLAMA_MTP_PMIN_NEXT, fc auto-off for dmax>1), single rounds, interleaved in one block per prompt:
   copy-heavy: d1 33.03 | d2 pmin-next 0.8 35.18 | 0.9 34.88 | none 34.75 ; Hello: d1 29.21 | 0.8 29.33 | 0.9 28.69 | none 28.74 ;
   JSON: d1 30.99 | 0.8 32.17 | 0.9 31.40 | none 32.64.  => gated dmax 2 is neutral on chat prose and +4-6% on structured/copy-heavy output (more with fc, see below).

## 2026-09-30 12:30 COORDINATOR: ops650 finished (branch ops650 b7f0f4e, plain decode 27.0 -> 35.1 tok/s). Merged into mtp3 (clean, no conflicts): merge commit on branch `mtp3`
(worktree llama.cpp-hexagon-mtp3); package npu_campaign/pkg-mtp3-e (host ggml-hexagon.dll 3A3970BF, skel ECCE2654). All cost-model/ceiling numbers above were measured on the OLD step cost.
Laptop was on BATTERY from ~13:05 (PowerOnline False): every speed number recorded after that is invalid and NOT recorded (a partial m3y ceil1 run was killed and deleted).
run() now tags each run with power=AC/BATT and the speed queues (m3_allac.sh) wait for AC; bit-exactness and garbage-hunt runs do not depend on speed.

### Gates on pkg-mtp3-e (ref = pkg-merged2-m1; mtp-gate.ps1, 128 traced tokens, full-vocab logit rows, --require-identical, cand uses LLAMA_MTP_FUSED_CATCHUP=1)
- Hello: ref-plain vs cand-plain, ref-mtp vs cand-mtp, draft rows (74/74), cand-plain vs cand-mtp, ref-plain vs ref-mtp: all exit 0, bits_equal=True.
- sky prose: all five comparisons exit 0, bits_equal=True (draft rows compared 539/539 top).
- (code prompt, PPL 4x512 = 12.1307 exact; PPL 16 chunks, edge check: see below when done)
- code prompt (interval merge): all five comparisons exit 0, bits_equal=True (target rows 128/128; draft rows 1103/1103 top).
- Qwen PPL (run-prefill-ppl.ps1, pkg-mtp3-e): 4x512 = 12.1307 +/- 1.11278 (exact), 16 chunks = 13.3331 +/- 0.63451 (exact).
- Fusion edge check (run-fusion-edge-check.ps1, OPFUSION=1 vs 8062, 6/40/130/260/520/900 words): 6/6 identical.
=> GATES PASS on pkg-mtp3-e (merge of mtp3 + ops650). Logit identity plain == MTP also holds (cand-plain vs cand-mtp, all 3 prompts).

### 14:00 Garbage hunt (timing independent, battery runs are fine for this)
Config = the not-recommended one: MTP draft-max 2, --spec-draft-p-min 0.7, GGML_HEXAGON_REPLAY_SLOTS=4 (multi-slot replay), sky prompt, 256 tokens, fc auto-off.
| build | runs | result |
| --- | ---: | --- |
| pkg-decode2-base (before the 64 B flush fix) | 3 | 1 clean (150/160), 2 GARBAGE ('////', 68/350 and 67/355 accepted) |
| pkg-mtp3-e (flush fix + ops650) | 3 | 3/3 clean, 150/160 |
Reproduction on the old build, absence on the fixed build: the flush fix cured it for this configuration (more runs queued: d1/p0.7/Hello, and N=6 per arm).
Forced fused catch-up with draft-max 2 (LLAMA_MTP_FC_FORCE=1, the original failing mechanism) on pkg-mtp3-e: 3/3 clean but acceptance 132/160 (vs 150/160 unfused): fc for draft-max 2 is a *draft-quality* issue (drafts differ), not garbage.

### 15:45 New-base ceilings (pkg-mtp3-e = ops650 merged; AC; Hello 256; block poll; ceil1 block)
plain 34.50 tok/s | MTP d1 spoof acceptance 100%: 38.60 | 85% (actual 0.83): 35.08 | 71% (actual 105/149 = 0.70): 32.57 | spin + 100%: 41.54.
=> on the new base MTP at chat-prose acceptance (0.70) is SLOWER than plain (32.6 vs 34.5): step cost is ~52 ms (plain 29 ms/token), break-even acceptance ~0.79.
   (old base: step 57 vs plain 35.6, break-even 0.52). ops650 sped up M=1 more than the M=2 verify. >35 at dmax 1 needs acceptance >= ~0.85 (code 0.95, copy 0.97, JSON 0.83, prose 0.7-0.85).
### Spin-poll (GGML_HEXAGON_OPPOLL=1) HANG: 2 of ~14 spin runs hung forever at 100% CPU (12:11 pkg-wos MTP after graph-replay capture; 15:18 pkg-mtp3-e plain at startup), holding the NPU lock for 14-28 min
until killed. The dspqueue_read loop in flush_pending `continue`s forever on EWOULDBLOCK/EEXPIRED, in both poll modes (no timeout/abort): a lost/late DSP response = infinite loop.
Spin is NOT safe to recommend. run-mtp-speed.ps1 got -TimeoutSec (kills a hung run; blocks use 420 s) so a hang can no longer hold the shared lock.
### Fused catch-up at draft-max 2 (LLAMA_MTP_FC_FORCE=1, experiment): target rows bit-identical (96/96) but acceptance 54/80 vs 58/74 unfused. Step-by-step accepted/drafted sequences (LLAMA_TRACE=1) are identical for
the first 3 verify steps, then diverge at the step right after a fully accepted draft (2/2: two deferred catch-up rows + current row in one 3-row draft batch); single deferred rows are fine. Root cause not found
(state bookkeeping reads correct); fc stays auto-disabled for draft-max > 1.

### 16:30 Garbage hunt final (AC): pkg-mtp3-e 12/12 clean (draft-max 2 p-min 0.7 slots=4 sky: 6 x 150/160; draft-max 1 p-min 0.7 slots=4 Hello: 6 x 89/106, words 187/262 every time) vs
pkg-decode2-base (before flush fix): d2: 2 of 3 garbage (68/350, 67/355), d1: garbage too (35/203 acceptance, '////'). Fc forced at draft-max 2: 3/3 clean. => the 64 B flush fix removed the garbage in the
not-recommended configs (multi-slot replay, p-min, draft-max 2).
### 16:35 First-token confidence gate (p-min, d1 fc+dv, new base, single runs, one block): Hello plain 33.53 | pm0 32.10 | 0.5 32.96 | 0.7 31.69 | 0.85 30.97;
JSON plain 34.08 | pm0 34.91 | 0.5 35.49 | 0.7 34.11 | 0.85 33.66. => p-min 0.5 is the best gate (a rejected draft costs ~20 ms on the new base, but a skipped draft still costs the draft graph 3.3 ms); used as arm E in the final matrix.
### Profile of the new base (DSP ms per batch): plain M=1 1114 ops 26.7 | MTP verify M=2 1354 ops 43.7 (+17.0: experts +5.8, dense MUL_MAT +3.3 (SHEXP+EP and RMS_NORM+SCALE fusions are M=1 only),
GDN +2.5, GET_ROWS +1.0 (72 -> 102 ops), unfused small ops +~3 (ADD+SG, SCALE, RMS_NORM, SWIGLU x2), FA +0.6) | draft 2.5.

## 2026-09-30 18:25 FINAL MATRIX (AC only; Hello-class prompts as named; 256 tokens, greedy; interleaved inside one NPU hold per block; rotated order per round; battery-window runs discarded)
Arms: A plain installed pkg-wos | B plain pkg-mtp3-e | C MTP d1 installed (fc+dv65536) | D MTP d1 pkg-mtp3-e | E D + first-token p-min 0.5 | F MTP d2, 2nd token gated by head confidence 0.8 (LLAMA_MTP_PMIN_NEXT, dv65536, fc auto-off).
Rounds completed: code 2, JSON 1, copy-heavy 1, prose(sky) 0 valid (round-1 block was killed by a battery switch; DSP errors 0x72 at the transition) => thin arms are indicative only. User asked to stop extra rounds.
--- code
A plain installed                  n=2 mean  26.92  runs 26.79 27.05  acc   vs installed plain +0.0%
B plain mtp3-e                     n=2 mean  34.40  runs 33.99 34.81  acc   vs installed plain +27.8%
C MTP d1 installed                 n=2 mean  33.42  runs 33.18 33.67  acc 124/131  vs installed plain +24.2%
D MTP d1 mtp3-e                    n=2 mean  37.40  runs 37.01 37.79  acc 124/131  vs installed plain +38.9%
E MTP d1 + p-min                   n=2 mean  37.34  runs 36.90 37.77  acc 123/129  vs installed plain +38.7%
F MTP d2 gated (2nd tok p>=0.8)    n=2 mean  37.91  runs 37.40 38.43  acc 159/169  vs installed plain +40.8%
--- json
A plain installed                  n=1 mean  26.53  runs 26.53  acc   vs installed plain +0.0%
B plain mtp3-e                     n=1 mean  34.17  runs 34.17  acc   vs installed plain +28.8%
C MTP d1 installed                 n=1 mean  31.16  runs 31.16  acc 116/139  vs installed plain +17.5%
D MTP d1 mtp3-e                    n=1 mean  34.73  runs 34.73  acc 116/139  vs installed plain +30.9%
E MTP d1 + p-min                   n=1 mean  35.39  runs 35.39  acc 116/130  vs installed plain +33.4%
F MTP d2 gated (2nd tok p>=0.8)    n=1 mean  32.52  runs 32.52  acc 151/173  vs installed plain +22.6%
--- sky
A plain installed                  n=1 mean  26.62  runs 26.62  acc   vs installed plain +0.0%
--- rep
A plain installed                  n=1 mean  26.63  runs 26.63  acc   vs installed plain +0.0%
B plain mtp3-e                     n=1 mean  33.98  runs 33.98  acc   vs installed plain +27.6%
C MTP d1 installed                 n=1 mean  33.00  runs 33.00  acc 125/129  vs installed plain +23.9%
D MTP d1 mtp3-e                    n=1 mean  34.16  runs 34.16  acc 125/129  vs installed plain +28.3%
E MTP d1 + p-min                   n=1 mean  35.82  runs 35.82  acc 125/126  vs installed plain +34.5%
F MTP d2 gated (2nd tok p>=0.8)    n=1 mean  37.22  runs 37.22  acc 166/173  vs installed plain +39.8%
Prose/chat (Hello, single block, pm1): plain 33.53 | MTP d1 32.10 | d1 + p-min 0.5 32.96 (acc 101/126) -> MTP does NOT beat plain on chat prose at acceptance 0.70 on the new base.
Q4_0 MTP head (headq40 GGUF): PPL 4x512 12.2553 vs 12.1307 (+1.0%, beyond the NPU-vs-CPU spread 0.07) -> not recommended; speed ABBA not run (user cut extra tests).
Next lever (documented, not done): extend ops650's SHEXP+EP and RMS_NORM+SCALE fusions and the GET_ROWS split to M=2..3 (est. -2 ms of the 17 ms excess verify cost, +4% MTP tok/s), gated by mtp-gate + PPL + edge.
