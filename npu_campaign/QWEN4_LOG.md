# QWEN4 log (append-only) - agent qwen4
Worktree `llama.cpp-hexagon-qwen4` (branch `qwen4` from npu-htp-decode 56a66d1 = installed pkg-final2-f1), build dir `build-qwen4`, packages `npu_campaign/pkg-qwen4-*`.
Build: `npu_campaign/qwen4-build.sh [first]` (build-qwen4.cmd / build-qwen4-htp.cmd / pkg-qwen4.ps1; cmd.exe needs full paths). Target model Qwen3.6-35B-A3B Q4_0.

## 2026-09-30 21:50 Setup + code reading (hypotheses before any measurement)
- Q4_0 tile (repack_q4_0_tiled): 32 rows x 32 k = 512 B quants (byte [cp*32+row] = q[row][2cp] | q[row][2cp+1]<<4) + 64 B fp16 scales = 576 B, padded to 640 B in VTCM.
  One DMA = one tile-row: 64 (K=2048) rows of 576 B each, dst stride 640 (2D descriptor, row_size 576 -> per-row DMA overhead + 64 B pad write).
- HVX Q4_0 dot (tiled_vec_dot_q4_0_32x1): per 4 vectors: and, lsr, vshuff, 2x valign, 2x vshuff (unpack_and_interleave_4bit_x2), 2x vsub 8, 2x vrmpy => ~11 ops / 128 B + scale math ~10 => ~55-60 ops per 576 B tile.
  Q8_0 dot: vror+vshuff+vrmpy = 3 ops/128 B => ~35 per 1088 B tile. Hypothesis H1: Q4 decode matmul is HVX-instruction (permute/shift) bound, not DMA bound.
- Hypothesis H2: expert (MUL_MAT_ID(_NX), M=1) kernels loop experts x weights with a full DMA fill/drain per (expert, weight): rows/thread = 96 = 3 tile-rows per (expert,p) => pipeline never reaches steady state, 16 drains per op.
- Spoofs added (OFF by default; GGML_HEXAGON_OPS650_SPOOF=14/15/16): matmul kernels (mv_2d_repacked, mm_nx_2d_repacked, mv_id, mv_id_nx) skip the dot (14, DMA only), skip DMA+pop (15, compute only), both (16).
- OPPOLL fix (host): bounded spin (GGML_HEXAGON_OPPOLL_SPIN_MS, default 400 ms) then blocking-read fallback with a logged rescue message; watchdog GGML_HEXAGON_RSP_TIMEOUT_S (120 s) aborts with a diagnostic instead of hanging.
- Package pkg-qwen4-s1 = spoofs + bounded spin (host ggml-hexagon.dll D47E1418, skel 7122CFDB). Profile-mode anatomy queue: qwen4-q1.sh (runs q4-anat-m0/14/15/16).

## 2026-09-30 22:10 Q4 matmul anatomy on Qwen decode (profile mode, 38 warm batches of 1113 ops, run q4-anat-m*; batch ms / per-op class ms per token)
Spoofs (garbage outputs, research only): m0 real | m14 dot skipped (DMA only) | m15 DMA skipped (compute only) | m16 both skipped (framework+quant+fork/join floor).
| class (ops/tok, weight MB/tok) | real | DMA only | compute only | floor |
| dense MUL_MAT (141, 1045 MB) | 9.05 | 8.93 | 6.58 | 1.24 |
| MUL_MAT+ADD (70, 255 MB) | 2.74 | 2.61 | 1.54 | 0.38 |
| MUL_MAT_ID_NX gate+up (40, 378 MB, Q4_0) | 4.44 | 4.06 | 2.78 | 0.26 |
| MUL_MAT_ID down (40, 191 MB, Q4_1 weights!) | 2.55 | (kernel hvx_mm_id not spoofed) | | 2.44 incl. all |
| MUL_MAT_NX (50, 96 MB) | 1.35 | 1.11 | 0.67 | 0.23 |
| whole batch | 26.40 | 25.56 | 20.30 | 10.88 |
=> H1 (Q4 matmul is HVX-instruction bound) is REFUTED: removing the dots changes the batch by <1 ms; the Q4 matmuls are DMA-bound (dense ops reach ~136 GB/s DMA-only incl. the 418 MB head at 142 GB/s). The "70-98 GB/s" of ops650 is the DMA-bound rate of SHORT ops (per-op ramp/tail) and, for the expert ops, imbalance + bubbles.
Per-op framework floor (m16) is only 5-6 us/op, not 15.
Expert ops: gate/up 512 rows = 16 tile rows over 6 threads = 3,3,3,3,3,1 (12.5% imbalance) + 16 (expert,weight) items each with a full DMA fill/drain. DMA-only 101 us vs compute-only 70 us vs 111 us real for a 9.46 MB op (ideal at 141 GB/s: 67 us + 6 floor).
Note: ffn_down_exps of this GGUF is Q4_1 (640 B tiles, Q8_1 activations), heavier dot (compute-only ~61 us/op).

## 2026-09-30 22:12 MV_XPIPE (htp/mv-id-xpipe.h, flag HTP_OPFLAGS_MV_XPIPE, host env GGML_HEXAGON_MV_XPIPE=1, default OFF until gated)
v1 (cross-item DMA pipeline only; pkg s3): ID_NX 111 -> 106.2 us, down 63.6 -> 59.5 us (profile mode).
v2 (+ flat unit distribution: all tile rows of all (expert, weight) items as one list, thread ith gets a contiguous range: <=1 unit imbalance; same dot kernel per tile row => bit-identical by construction; pkg s4): ID_NX 111.1 -> 98.5 us (-11%), down 63.6 -> 53.5 us (-16%), batch 26.40 -> 25.60 ms (-3%). Gates: see below.
v3 (pkg s5): + MUL_MAT_ID_NX for 2..4 act rows (MTP verify), cne1 columns per expert.
Remaining gap for the experts: 98.5 us for 9.46 MB = 96 GB/s vs 142 GB/s streaming. DMA-shape experiment (spoof 17, 1D contiguous tile rows instead of 64 x 576 B rows with 640 B dst stride; pkg s6) queued as q4-dma-m14 vs q4-dma-m17.

## 2026-09-30 22:25 Corrections + results
- CORRECTION: the HTP session has 8 threads (hwinfo: threads 8, hvx 8, hmx 1, vtcm 8 MB), not 6. Expert gate/up = 16 tile rows / 8 threads = 2 each (balanced), down 64/8, dense 256/8: the original kernels were NOT imbalanced; the xp gain (ID_NX 111 -> 98.5 us, down 63.6 -> 53.5 us) comes from removing the 16 / 8 per-item DMA fill/drain bubbles and giving each thread whole contiguous items.
- Gates (pkg s3 v1 and s4 v2, MV_XPIPE=1, Hello 128 rows full-vocab plain logits vs pkg-final2-f1, compare_logits_traces --require-identical): v1 bit-identical, v2 bit-identical (qwen4-gates.out).
- DMA-only experiment on the flat kernel (q4-dma-m14, spoof 14 + XPIPE): ID_NX 94.6 us, down 49.6 us (vs 98.5 / 53.5 real): even with zero compute the experts stream at ~100 GB/s (9.46 MB / 94.6 us), vs 142 GB/s on the 418 MB head.
- GDN row-block decode kernel (htp/gdn-rb.h, GGML_HEXAGON_GDN_RB=1, pkg s7): GATED_DELTA_NET+CPY 58.9 -> 56.8 us/op (-3.6%, 0.06 ms/token): the kernel is HVX-compute bound (about 10 us per head per thread), not DMA-latency bound. Not worth defaulting.
- Per-op floor structure (m16, nothing but framework): ~5 us per op; the a-intercept of 13-15 us in the matmul fit therefore contains ~8-10 us of ramp (act DMA + quant + fork/join + first-tile latency + tail compute) per matmul op.

## 2026-09-30 22:30 KEY FINDING: the weight DMA SHAPE is the Q4 decode bandwidth limiter (spoof q4-dma-m17: NODOT + 1D contiguous tile-row DMA, pkg s6)
DMA-only (no dots, garbage outputs; xp kernels on), profile mode, ms per token (ops/token):
| class | 2D 64x576 B rows (m14) | 1D contiguous 36 KB (m17) |
| dense MUL_MAT (141) | 8.86 | 7.70 (136 GB/s eff) |
| MUL_MAT+ADD (70) | 2.56 | 2.03 |
| MUL_MAT_ID_NX gate/up (40) | 3.79 (94.6 us/op) | 2.83 (70.7 us/op, 133 GB/s) |
| MUL_MAT_ID down Q4_1 (40) | 1.99 | 1.59 |
| MUL_MAT_NX (50) | 1.11 | 0.96 |
| whole batch | 24.62 | 21.41 (-3.2 ms) |
The fit intercept falls from 13.0 us to 3.7 us per op. => each 2D descriptor row costs ~12 ns (28% of a 576 B row at 17.6 GB/s/thread); the real fix is to move weight tile rows with ONE contiguous descriptor.
Q4_1 (640 B tiles, already contiguous in VTCM at stride 640) and Q6_K (896) need only the descriptor change; Q4_0 (576 B tiles, padded to 640 in VTCM) needs a dot kernel that reads tiles at stride 576 (odd tiles are 64 B-misaligned: unaligned vector loads). Plan: GGML_HEXAGON_Q4C (op flag CONTIG): contiguous DMA for aligned types + q4_0c dot kernels for Q4_0 (dense mv / mm / nx and the xp expert kernels). Expected DSP gain up to ~3 ms/token when the dots keep up (compute-only is already ~85% of the DMA time at 136 GB/s).
Verify-step profile with s5 (xp0 baseline): batch 43.5 ms; GATED_DELTA_NET+CPY at M=2 is 141 us/op = 4.2 ms (30 ops), experts 8.1 + 4.5 ms.

## 2026-09-30 22:40 Implementation state (branch qwen4 in llama.cpp-hexagon-qwen4; all opt-in, default OFF, via op flags set by host env)
- GGML_HEXAGON_MV_XPIPE=1: flat unit scheduling + cross-item DMA pipeline for MUL_MAT_ID(_NX) at decode and verify (2..4 rows). Gates so far: s3 (v1), s4 (v2), s7 (+GDN_RB): Hello plain 128 rows bit-identical vs pkg-final2-f1.
- GGML_HEXAGON_Q4C=1 (bit 0: CONTIG) / 3 (+ bit 1: CONTIG8): contiguous weight DMA (one descriptor per tile row): Q4_0 via new dot kernels q4_0c (tiles at stride 576, odd tiles with unaligned loads), Q8_0 via q8_0c (stride 1088); Q4_1 / Q6_K only the descriptor changes (their tiles were already unpadded). Dense mv/mm/nx kernels and the xp expert kernels. pkg s8 = Q4C bit0, s9 = +Q8.
- GGML_HEXAGON_GDN_RB=1: decode GDN row-block units (-3.6% GDN op time, 0.06 ms/token): kept but not recommended.
- GGML_HEXAGON_GDN_SNAPDMA=1 (pkg s10): verify/prompt GDN kernel (T>1, K>1) writes the recurrent-state snapshots with async DMA from a VTCM copy instead of a synchronous HVX copy to DDR (12 us per head; GDN verify op is 141 us vs 58 us at M=1). Bit-exact by construction; gate queued (q14).
- Queues: q12 (s8 gate+prof), q13 (s9 gate+prof), q14 (s10 mtp gate+prof), q7 (s5 MTP profile xp0/xp1 + 4-arm gate).

## 2026-09-30 22:50 Gates + INCIDENT (npu-lock.ps1 race)
- Gate results (Hello, 128 full-vocab plain-logit rows vs pkg-final2-f1, --require-identical): s3 (xp v1) identical, s4 (xp v2) identical, s7 (xp + GDN_RB) identical, s8 (xp + Q4C=1) identical, s9 (xp + Q4C=3, i.e. Q4_0 + Q8_0 contiguous kernels) identical. See qwen4-gates.out.
- INCIDENT: several of my own queue scripts waited for the NPU lock at the same time via npu-lock.ps1 (Enter-NpuLock uses New-Item -ItemType Directory, which is not an atomic test-and-set: two waiters polling in lockstep both "got" the lock; owner.txt then had two lines with the same timestamp) and two llama-cli processes ran concurrently for ~3 min (pkg-final2-f1 ref run and pkg-qwen4-s8 gate run; both were logits-trace gate runs, so correctness data is unaffected, but any profile/timing run that overlapped is suspect). The profile numbers taken before this (q4-anat-*, q4-xp*, q4-dma-m14/m17, q4-gdn-prof, q4-mtpprof-*) came from the same multi-queue period: they are internally consistent but will be re-confirmed by lockrun-protected interleaved runs. From now on ONE serial queue (qwen4-all.sh) and every run through lockrun.py (atomic os.mkdir) - qwen4-lib.sh / qwen4-gate.sh. Not touching the shared npu-lock.ps1; recommend the coordinator replace New-Item by an atomic mkdir (cmd /c mkdir, or [System.IO.Directory]::CreateDirectory + owner check is NOT atomic either).

## 2026-09-30 23:05 First real Q4C measurements: the BRANCHY contiguous dot kernels are SLOWER (s10), pair-unrolled rewrite (s12)
Profile mode, same session, lockrun-protected (q4-m1-*; the plain run q4-m1-prof-all may have overlapped with an orphaned run for its first seconds, treat as indicative):
| | baseline pkg-s10 flags off | s10 xp+Q4C=3+SNAPDMA |
| plain decode batch (1113 ops) | 26.4 (q4-anat-m0) | 27.2: MUL_MAT 9.05->9.77, ID_NX 111->114, ID 63.6->66.4 (no gain, slower) |
| MTP verify M=2 batch (1353 ops) | 43.28 (mprof-off) | 48.03: ID_NX 203->290, ID 114->159 us (worse), GDN+CPY 139->120 us (SNAPDMA works: -0.56 ms), MUL_MAT 12.49 -> 12.46 |
Diagnosis: with the DMA now fast the dot becomes the limiter, and my first c kernels (q4c_load_tile with a per-tile parity branch and an array through a pointer) compile badly (no pipelining across tiles). Rewrite (s12): process tiles in PAIRS, even tile = 4 aligned vectors + scale, odd tile rebuilt with valign from the same 9 aligned vectors (1152 B), no branch, no unaligned loads, same float operation order. Q8_0 same with 17 vectors per pair.
Correctness already established for the branchy version: Q4C=3 + xp plain logits bit-identical (s8, s9); s10 (xp + Q4C=3 + GDN_SNAPDMA) MTP target rows (verify M=2) bit-identical 128/128. The pair version is gated again (queue m3: plain gate, profiles, MTP gate, ABBA).

## 2026-09-30 23:15 s12 (pair-unrolled Q4C) profile: plain batch 24.83 ms (xp-only 25.50, base 26.40); the dot is now the limiter
q4-m3-prof-all (xp + Q4C=3 + SNAPDMA, pkg s12, profile mode): MUL_MAT 8.77 (base 9.05), ID_NX 94.4 us (xp 98.5, base 111, DMA-only potential 70.7), ID 57.6 (xp 53.4), MUL_MAT+ADD 2.42 (2.73), NX 1.23 (1.35). q4-m3-prof-xp reproduces the earlier xp-only profile (98.5 / 53.4 / 25.50 ms): the profile numbers are repeatable.
Disassembly of the linked skel (hexagon-llvm-objdump on libggml-htp-v81.so, tiled_vec_dot_q4_0c_32x1): the pair loop is ~90 VLIW packets for 2 tiles (45/tile), most packets hold a single permute (vshuff pair / valign: 5 per 128 B of nibbles = 20 per tile, +2.5 for the odd-tile valign, + a few stack spills); the old padded kernel needs ~33 packets/tile. Measured: old kernel compute-only 65 cycles/tile, c-pair ~90 cycles/tile => ~2 cycles per packet, i.e. the Q4_0 dot is bound by the single HVX permute slot (the 4-bit unpack of the shared HMX tile layout needs vshuff-1 + 2 valign + 2 vshuff-2 per 128 B), NOT by DMA any more. Consequently the DMA fix only pays what the dot can absorb: ID_NX 111 -> 94 us instead of 71.
What would remove the limit (not done, needs a new weight/activation layout + HMX dequant adaptation + PPL gates): store Q4_0 tile bytes as [row*4+cpl] (lane = row, 4 k-pairs) so that and/lsr + one vshuff(-1) + one vdeal(-4) produce the rmpy operands (2 permutes per 128 B instead of 5), at the price of 2-3 extra permutes per 128 B in the prefill HMX dequant (HVX-bound there already); or an activation-permuted vmpy-accumulate formulation (per-lane products, reduction over the 4 cpl blocks once per tile).

## 2026-09-30 23:25 FOR THE `layout` AGENT: exactly what limits the Q4_0 decode dot, and the layouts that need fewer permutes
Measured facts (profile mode, 8 HVX threads, DSP clock 2112 MHz; branch qwen4 `llama.cpp-hexagon-qwen4`, tools: `GGML_HEXAGON_OPS650_SPOOF=14/15/16` = matmul kernels skip dot / skip DMA / skip both, `npu_campaign/ops650_prof.py <detail.log>`, runs via `npu_campaign/qwen4-lib.sh` q4run, qwen4-m*.sh queues):
- Decode Q4 matmuls are NOT DMA bound by the data volume: DMA-only (no dots) at ONE contiguous descriptor per tile row streams 133-136 GB/s (ID_NX 9.46 MB: 70.7 us); with the 2D descriptor (64 rows x 576 B, 640 B VTCM stride) only ~100 GB/s (12 ns per descriptor row). Compute-only (no DMA) for the same op: 63 us net. So with a fast DMA the dot is ~equal to the DMA time and decides the result.
- Dot cost = 65 cycles per 32x32 tile (x1 kernel `tiled_vec_dot_q4_0_32x1`, 8 threads). Disassembly (hexagon-llvm-objdump -d on build-qwen4/.../libggml-htp-v81.so): ~33-45 VLIW packets per tile, most packets hold ONE permute (HVX has one permute slot): `unpack_and_interleave_4bit_x2` = vshuff(-1) + 2 valign + 2 vshuff(-2) = 5 permutes per 128 B of packed nibbles = 20 permutes per tile (+ the f16 scale convert: vshuff(-4)). Everything else (and, lsr, 8 vsub, 8 vrmpy, scale math) pairs up.
Current Q4_0 tile (repack_q4_0_tiled, ggml-hexagon.cpp): byte[cp*32+row] = q[row][2cp] | q[row][2cp+1]<<4, cp=0..15 (vector i = cp 4i..4i+3, i.e. lane = (cp%4)*32 + row), then 64 B fp16 scales (row order) at +512. Good for HMX dequant (and/lsr, vshuff(hi,lo,-1) gives the [cp][row][parity] f16 element order directly), bad for vrmpy, which needs lane = row with 4 CONSECUTIVE k in the 4 bytes of the lane.
Layout 1 (zero permutes in the HVX dot; = the repo's own Q6_K tile, see matmul-ops.h comment above HTP_MM_WEIGHT_TILE_SIZE_Q6_K): byte[4*row + b] of vector i (i=0..3) = q[row][4*(2i)+b] | q[row][4*(2i+1)+b] << 4, b=0..3 (low nibble = k-group 2i, high nibble = group 2i+1, group g = k 4g..4g+3). HVX: W(2i) = and(v,0x0F), W(2i+1) = lsr(v,4) -> vsub 8 -> vrmpy with act group g (act tile layout unchanged). 0 permutes (only the scale convert). HMX dequant per group (Q6_K style): vunpack(bytes->halfwords) + vdeal(-4) = 2 permutes per group = 4 per 128 B vs 3 today (prefill dequant is HVX bound, so expect a few % on prefill; check with PPL gates and pp1024).
Layout 2 (2 permutes in the HVX dot, HMX side costs more): byte[row*4 + cpl] of vector i = q[row][2cp] | q[row][2cp+1]<<4 with cp = 4i+cpl. HVX: L = and, H = lsr, vshuff(H,L,-1) (byte interleave: 16 rows x 8 bytes per output vector) then vdeal(-4) (32-bit deal) gives W0 = per-row word [L0,H0,L1,H1] = k 8i+0..3 and W1 = [L2,H2,L3,H3] = k 8i+4..7 -> 2 permutes per 128 B (8 per tile) instead of 5 (20 per tile).
Tile-size/alignment note: a 576 B tile makes every odd tile 64 B misaligned in a contiguous stream (valign x5 per odd tile = the cost that made my contiguous-576 kernels slower). Cheap fix without padding: store tile PAIRS as [q_even 4 vectors][q_odd 4 vectors][one scale vector: even scales in the low 64 B, odd scales in the high 64 B] = 1152 B, all quants 128 B aligned, one vror per pair for the odd scales (needs n_k_tiles even: K=2048/512/4096 here). The 640 B padded tile (+11% bytes, 1D DMA, no kernel change) is what pkg-qwen4-s13 tests.
Everything is integer-exact: any of these layouts must give bit-identical logits (gate: npu_campaign/qwen4-gate.sh TAG PKG "ENV" plain|mtp hello vs pkg-final2-f1; PPL 12.1307 / 13.3331).
Coordination: I will not touch the Q4_0 dot/layout further except tile padding (s13) until I read LAYOUT_LOG.md; I own xp (MV_XPIPE), GDN snapshot DMA, GDN state-gather fusion, the OPPOLL work and the temperature-0.6 matrix.

## 2026-10-01 00:10 s12 plain decode ABBA (lockrun-protected, AC, Hello 256 tokens, interleaved x3; xp + Q4C=3 + GDN_SNAPDMA)
base pkg-final2-f1: 35.77, 36.37, 36.45 (mean 36.20) | s12: 38.47, 38.41, 38.87 (mean 38.58) => +6.6% plain decode. Gates: plain logits bit-identical (s12). MTP verify (M=2) with s12 flags is SLOWER on the experts (c-dot is permute bound at 2 act columns) -> do not use Q4C Q4_0 kernels for verify; s13 (Q4_0 tiles padded to 640 B in DDR, original dot + 1D DMA) is the candidate that should fix that.
Incident 23:17-23:40: a stale hold shell of mine (the m3 ABBA hold, not matched by my kill pattern) started a second llama-cli while a new chain ran; both hung in exit and RAM filled. Fix: one chain only (qwen4-chain.sh atomic mkdir), qwen4-guard.py pre-flight (RAM>20000 MB, no live llama-*) inside every lock hold, exact-PID kills only.

## 2026-10-01 01:15 s13/s14 results
s13 = Q4_0/IQ4_NL tiles padded to 640 B in DDR (original dot + one contiguous 1D DMA per tile row) + MV_XPIPE + Q8_0 contiguous kernels + GDN snapshot DMA. Gates: plain logits 128/128 bit-identical; MTP target rows 128/128 and draft rows 72/72 bit-identical; with GDN_STATE_MAXT=4 also bit-identical (target 128/128, draft 72/72).
Profile (plain, profile mode): batch 26.40 (base) -> 23.61 ms; ID_NX 111 -> 82.8 us, MUL_MAT 9.05 -> 8.25 ms, MUL_MAT+ADD 2.74 -> 2.30.
Plain decode ABBA x3 (Hello 256 tok, AC, lockrun): base 35.48/35.06/35.64 (35.39) vs s13 40.11/40.73/39.90 (40.25) = +13.7%.
MTP verify M=2 profile: base 43.28 ms; s13 43.27 (experts unchanged: ID_NX 200 us, still limited at M=2); s13 + GDN_STATE_MAXT=4 (state gather folded into GDN: GET_ROWS 101 -> 71 ops) + SNAPDMA: 41.35 ms (GDN 139 -> 108 us).
s14 = s13 with all opts default ON (env GGML_HEXAGON_MV_XPIPE/Q4C/GDN_SNAPDMA/GDN_STATE_MAXT=0/1 override), branch qwen4 commit a21dc7f. Final battery queue qwen4-m8.sh running on pkg-qwen4-s14 (PPL x2, gates hello+code plain+mtp, edge check, ABBA x3 for Hello/CODE x plain/MTP/MTP+pmin0.5).

## 2026-10-01 03:55 FINAL: pkg-qwen4-s14 (branch qwen4 a21dc7f, worktree llama.cpp-hexagon-qwen4; all opts default ON)
Gates on s14 (no env): PPL 4x512 12.1307, 16 chunks 13.3331 (exact); plain + MTP target logits bit-identical on Hello and code (128/128), draft rows 72/72 and 66/66; fusion edge check 5/5 identical (6/40/130/260/520 words). 
Interleaved ABBA x3 (AC, 256 tok, lockrun) vs pkg-final2-f1, tok/s mean: Hello plain 35.26 -> 40.23 (+14.1%) | Hello MTP d1 33.08 -> 35.22 (+6.5%) | Hello MTP d1 pmin0.5 33.28 -> 35.56 (+6.9%) | code plain 35.02 -> 40.16 (+14.7%) | code MTP d1 37.47 -> 40.15 (+7.1%) | code MTP pmin0.5 37.86 -> 39.80 (+5.1%).
=> with s14 plain decode (40.2) is >= MTP everywhere (code MTP 40.15 ~ plain): MTP no longer pays; use plain.
Not done (time/NPU sharing): OPPOLL stress/ABBA, temp-0.6 matrix, item 5 A/Bs (scripts qwen4-q6/q11/q9/m4 ready); spin fix itself is built (bounded spin 400 ms then blocking fallback, 120 s watchdog; not stress tested).
