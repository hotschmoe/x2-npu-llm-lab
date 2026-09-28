# Campaign status

## Independent-prompt correctness and untraced A-B-B-A confirm fast Q8_0 lead

Sep27 local continuation: matched 64-token native runs with prompt `Explain why a solar eclipse cannot occur at every new moon.` used the archived fast Q8_0 and scalar-hoist binaries. Both terminal0, all64 logged logit rows and selected tokens bit-identical. Candidate decode13.80tok/s (72.45ms/token); scalar12.71tok/s (78.70ms/token). This is a second prompt, not a broad workload suite.

No-logit-trace/no-device-profile 128-token `Hello` A-B-B-A, one fresh process/model load per arm, all terminal0: scalar A1 76.37ms/token/13.09tok/s; fast B1 70.26/14.23; fast B2 70.32/14.22; scalar A2 76.81/13.02. Mean displayed decode latency76.59 ->70.29ms/token (~8.23% reduction), corresponding throughput13.055 ->14.225tok/s (~8.96% increase). All four generated-text transcripts match SHA6ac2572fdb5ea93f39431f3c009a60bf902c87bf65bd56dfa6daeedf4f1ff8e7; no `.logits.bin` was produced in these rate runs. The separate traced full128 and independent-prompt runs establish exact logged logits for their respective settings; transcript identity here is weaker than logit identity but confirms the untraced outputs. Each wrapper restored baseline installed hostC9E4/DSPD2D/catalog5CB7. No inference/build process remains. Fast candidate remains archived, flag OFF in nested build, not installed. User requested pause after this investigation; do not start another experiment until resumed. Full goal remains unfinished.

## Exact Q8_0 HVX conversion yields measured decode speedup

Sep27 local: OFF-default `GGML_HEXAGON_Q8_0_FAST_REFERENCE` replaces the F32 Q8_0 scalar `roundf` code loop with HVX multiply, exact half-away-from-zero rounding via integral/fraction comparison, and saturating vector pack. It requires `GGML_HEXAGON_Q8_F32_REFERENCE=ON`. A naive direct HVX F32-to-int conversion differed in1009/2048 codes (truncation); add-signed-half then convert matched real rows but failed11/2048 deliberately constructed near-half codes. The final integral/fraction path matches scalar tiled output and installed CPU packed bytes exactly on five frozen CPU/NPU model rows (T1, T9 rows0/8, T4 row3) plus a constructed exact/adjacent-half fixture, including every replicated tile lane. On one frozen K2048 row v81 simulator cost changes from314,504 instructions/442,986 pcycles to~95,200 instructions/~109,200 pcycles; this is only a simulator signal, not device rate.

Signed candidate `backup-hmx-q8_0-fast-controlled-candidate` host390778A550001CAA24B52F599BF6424CEA0660614D2F81437E511D2FDA381961, DSP7A714E862FD8F1058FBA836CF68BF50670E3B0CC9951458681E95282151DBE67, catalog9BA0CC6A27529D22B0B343D52F8F6C6E2DA407C1E2527EFCF4E2F347F2CBF897; archive cache and ninja verify flag ON and scalar-hoist control flags. Matched native 2-token capture terminal0: 25 layer-0 retained frames bit-identical to scalar control; both logit rows identical. Basic-profile replay0 four-token complete one-token batches (3x1384ops) mean70,461us (min70,150/max70,793), versus scalar-hoist75,896us (75,296..76,250), ~7.16% less DSP batch time. Gate expert ops sum6,458us vs7,030us; expert down5,126us vs6,378us, but summed op deltas are attribution, not additive wall time. Four logits/tokens exact. Full128 paired native runs terminal0 and all128 logits/tokens bit-identical A==B==prior scalar control; traced rates13.84/13.78tok/s (72.26/72.56ms per token) versus scalar12.74/12.86tok/s (78.48/77.77ms). Direct comparison to matched CPU128 confirms first selected-token difference remains step93 (CPU60239/candidate3812), so this preserves the control's CPU divergence, not CPU identity or MTP correctness. Wrapper restored baseline installed hostC9E4/DSPD2D/catalog5CB7. Scalar rebuild restored nested build DSP hashD986 and fast CMake flagOFF/compiled define absent. Candidate is best measured correct-output decode so far, pending broader independent workloads/untraced benchmark. Full goal active.

## Current one-token MoE trace shifts speed target to F32 Q8_0 packing and DMA/dot overlap

Sep27 local: scalar-hoist signed control mode3/replay0 profile4 terminal0, wrapper restored baseline. New wrapper `-ProfileMode 3 -TraceCapacity 65536` supports bounded event capture. Seven batches captured; final one-token batch6 has1384ops/76,901us DSP time and full event integrity: reported/observed [44494,38964,38764,38764,38684,38684,38684,38684,0,0,0], no saturation/missing/unmatched/open/out-of-batch events. Mode3 application timings (~191seconds for4tokens) are invalid throughput due hundredsMB text logging; basic-profile controls remain latency reference.

Clipped final-batch layer0 `MUL_MAT_ID` expert-down op108us: each worker activationQ8_1 quant~4.12..4.17us, HVX compute~41.46..42.77us, DMA residence union~66.66..69.58us with~36.47..37.84us overlap with traced HVX. DMA outside traced HVX~29.82..32.02us is not proven DMA stall. Layer0 `MUL_MAT_ID_NX` fused gate/up op187us: worker0 activation Q8_0 quant42.248us, HVX compute65.491us, DMA union86.016us, overlap33.829us/outside52.187us; worker1 quant40.402us/compute63.614us/DMAunion88.287us. Source/profile op shows a single shared F32 activation row [2048,1] with K2048 and eight selected IDs; `hvx_mm_matmul_id` parallelizes its16 x128-element quant blocks over8workers, so this is not eight redundant activation rows. This trace supports optimizing Q8_0 F32 code generation and/or weight DMA/compute overlap, not pursuing slower CPU-style Q8_1 down packing. Counts are clipped per-op and overlap categories are not additive wall time. Full correctness/coherence/MTP goal active.

## Q8_1 CPU-contract runtime intervention closes locally, fails full-model gate

Sep27 local: OFF-default `GGML_HEXAGON_Q8_1_F32_REFERENCE` uses F32 max/divide and nearest-even codes plus HF stored scale/sum before the existing Q8_1 tile replication. Simulator ELF SHA4F9B4AFBB41C16F2987D02047A110D8887E4F892A379812946187F96739B98D9; on NPU and CPU frozen T1 layer-0 inputs it produces exactly installed CPU Q8_1 bytes (0/4096 code and0/512 scale+sum-byte differences; contiguous Q8_1 SHA DA1A8BD13162EA056E6665DDE504CB1A77AF1191FF80B3BE98D1199E8DEB5BE5). Signed archive `backup-q8_1-f32-decode-controlled-candidate` hashes: host390778A550001CAA24B52F599BF6424CEA0660614D2F81437E511D2FDA381961, DSPC4A9C5AB7284318BFEF99DD498C79C3BD4E09D8D92F35B6F487E7B405F82C364, catalogDCF1DAB797DC4F7F57BA1143CC4E5A1AA5DA0DF015F5096BDE7DCBDCB7622362. Archive manifest/cache/ninja confirm flag ON; candidate source included. No publication/commit.

Matched native capture terminal0: 30 complete layer-0 frames; compared to scalar-hoist control, only four `ffn_moe_down-0` small-token frames change (T2 twice, T4, T1). All upstream frames and T9 expert-down are bit-identical. CPU-vs-NPU T1 down NMSE falls1.4026357e-5 ->4.64301155e-14, maxabs2.60770321e-8. Native installed CPU Q8_1 quantizer+Q4_1dot1 on candidate's own frozen T1 inputs matches same value; rotated expert IDs fail unchanged1e-7 threshold with NMSE1.434296. This is a strong local runtime correction. It does NOT prove whole-model correctness, and the candidate's first two logits differ from both CPU and prior NPU control while first two selected tokens match.

Matched replay0/basic profile4 terminal0: three complete1384-op one-token batches mean77,133.333us (min76,958/max77,245), versus scalar-hoist75,896us (75,296..76,250); 40 `MUL_MAT_ID` expert-down ops sum6,826.667us vs6,378us, 40 gate `MUL_MAT_ID_NX` ~7,029us vs7,030us. Four selected tokens match control, logits differ from row0. Paired full128 NPU repeat both native0 and all128 logits/tokens A==B bit-exact. Against independent CPU128, both first selected-token difference atstep60 (CPU1061/NPU353), vs prior control step93; first logit-bit difference0. Candidate versus prior NPU control also first differs atstep60 (prior NPU1061/candidate353), so the intervention introduces this earlier selected-token split on the same generated prefix. Traced decode12.64/12.72tok/s vs prior12.74/12.86, no gain. Therefore REJECT candidate for promotion despite local oracle success: CPU-style quantization is slower here and worsens full-model token agreement. Exact source of step60 divergence remains unlocalized; do not claim Q8_1 correction itself is mathematically wrong. Wrapper restored baseline installed hostC9E4/DSPD2D/catalog5CB7; subsequent scalar builder restored nested CMake flagOFF/compileddefine absent and buildDSP hashD986 scalar control. No inference/build live. Full performance, coherence, and MTP goal active.

## v81 simulator closes sampled decode expert-down quantizer hypothesis

Sep27 local: offline `hexagon_hvx_q8_1_sim.c` includes existing `matmul-ops.c` and calls `quantize_row_f32_q8_1_tiled` on exact F32 rows exported from layer-0 one-token SwiGLU captures. Current `hvx-mm-kernels-tiled.h` SHA D3A7CD9782BF8791D7E6F647489BDF68609AE7A30203BBD53152C44A9644D344 and `matmul-ops.c` SHA C0B748074A63863DC1A901D6C0FEBB1BB5340542A41F2C7216B2ECC5E8AB62E7 match signed scalar-hoist archive byte-for-byte. v81dgb_1 simulator terminal0; final candidate-flags LTO/vector/xqf/O2 ELF SHA AE56E1BE9A665F09FC5E9FB637D578772046ADB7FA6A752979D7A0BA2212F6C0. Earlier non-LTO build and repeated sim runs produced the same exact NPU 163840-byte tile SHA64ED1FFE1DB925FF499F351C3B4624E565F34230247DFB6DCF58BDC403682F10. Frozen NPU input SHA70CB241353B3D14B07D958A31F2FA88DAC23575FDE90676AEBB10EC25AA84CDA; unpacked contiguous Q8_1 SHAEE9D8FBC984FBDB02733315A9E2A3518F147478981D8FE0A45CE9AF286308ACE. Fixture bytes are checked against trace and every tile's replicated code/scale/sum lanes are checked.

Simulator Q8_1 packed bytes + installed CPU Q4_1 native dot1 on NPU's own T1 frozen rows match captured NPU expert-down at NMSE4.70744871e-14/maxabs2.70083547e-8, fixed gate1e-7 pass. NPU F32 rows quantified by installed CPU Q8_1 + same dot miss NMSE1.40309456e-5; simulator packing differs in42/4096 codes and80/512 scale+sum bytes. Complementary CPU capture: CPU's own Q8_1 dot matches its down output exactly, but simulator Q8_1 + same dot misses NMSE1.39024114e-5. Rotating expert IDs unchanged threshold rejects NPU simulator packed oracle (NMSE1.43423959, maxabs0.0976954652). The previous approximate half-stage emulation only reached4.26721667e-6; exact ISA source simulation resolves that gap. This strongly attributes the sampled decode expert-down mismatch to Q8_1 activation quantization contract; it does not establish physical runtime activation bytes, all layers/tokens, complete model agreement, or cause of first selected-token divergence93. No NPU binary modified or installed for this test. Full goal active.

## Paired layer-0 MoE capture isolates decode-relevant expert-down arithmetic

Sep27 local: matched `Hello`/seed1234/temp0 two-token CPU and signed scalar-hoist NPU captures both exited0. Each has30 complete frames (six named tensors x five occurrences). `ffn_moe_topk-0` expert IDs are bit-identical across all five occurrences. At prompt T9/occ2, CPU-versus-NPU NMSE is 1.15300781e-10 for `attn_post_norm-0`, 5.30688832e-8 gate, 9.08101273e-8 up, 1.19344943e-7 SwiGLU, but 4.53075492e-5 expert-down. At decode T1/occ4, corresponding input/gate/up/SwiGLU differences are 8.23054e-15/9.43828574e-15/1.06128773e-14/4.33887726e-13, while expert-down is 1.4026357e-5. This is a selected-expert output before router-weighted reduction, not the final model output. CPU and NPU captured their own input values; do not substitute one side's inputs into the other's reference.

Original GGUF layer-0 expert-down is Q4_1 K512/N2048/E256. Installed CPU Q8_1 quantizer + Q4_1 vec_dot nrc1 reproduces CPU's own captured down tensor exactly (NMSE0/maxabs0) for T9 and T1; same primitive on NPU's own captured inputs differs (T9 NMSE4.53111565e-5/maxabs0.00117273629; T1 1.40309456e-5/maxabs0.000576511025). Dense original-GGUF F64 projection on own inputs differs from CPU T9 by NMSE4.49995103e-5, but from NPU T9 by only2.03786697e-7. For T1 both are similarly far from dense (CPU5.75551029e-5, NPU5.82089385e-5), yet differ from each other. The NPU's failure against the CPU primitive is not by itself an NPU error: arithmetic contracts differ.

The source dispatch threshold is `HTP_MM_HMX_MIN_NROWS=4`: MUL_MAT_ID T9 is HMX-eligible, while T1 is HVX fallback. HMX gathered Q4_1 path converts F32 activation to F16, dequantizes tiled weights to F16, writes F16 HMX output then scatters to F32; Q8_0 activation-reference switch excludes Q4_1. HVX Q4_1 fallback declares Q8_1 tiled activation rows. Its tiled quantizer explicitly narrows activation and maxima to HF, computes HF inverse/scaled codes, and stores HF scale and affine sum, unlike the installed CPU generic Q8_1 quantizer. This is a concrete candidate arithmetic explanation for T1, not yet a numerical reproduction of captured NPU output. This source-derived path distinction is consistent with the T9 dense-oracle pattern, but runtime kernel selection and precise numerical mechanism should be independently traced before causal claims. Especially T1/HVX difference to CPU is the decode-correctness lead.

Original GGUF Q4_0 expert up/gate own-input native CPU dot1 controls: CPU T9/T1 NMSE2.799e-14/2.914e-14 up and2.066e-14/2.567e-14 gate; NPU T9/T1 NMSE8.205e-8/2.814e-14 up and5.062e-8/2.618e-14 gate. All pass fixed1e-7 gate. Dense up/gate controls are ~1e-5 to5e-5, as expected for a Q8_0 activation contract. Rotated expert IDs fail unchanged gate on NPU T9 down (NMSE1.3852) and up (1.8354). Thus selected IDs and slot mapping are not the obvious first fault. No causal attribution to first selected-token divergence93 yet; scalar hoist remains diagnostic/prefill-only, baseline restored, full goal active.

Exploratory `probe_hvx_q8_1_approx.py` applies the HVX source's half-stage shape to the NPU frozen T1 SwiGLU rows, then uses the installed CPU Q4_1 dot1 as a controlled common projection. It changes29/4096 Q8_1 codes and71/512 scale bytes from CPU quantization and reduces NPU down NMSE from1.40309456e-5 (native CPU Q8_1 oracle) to4.26721667e-6, maxabs0.000230901642. On CPU's own frozen rows it instead *increases* error from0 to1.28010922e-5. This supports a quantizer-path explanation directionally, but does not reproduce Hexagon's inverse/conversion instructions, tiled dot reduction, or actual packed activation bytes; it is explicitly NOT a correctness oracle. Next require either HTP-side packed Q8_1 capture or tighter ISA-exact frozen reproduction before calling cause. Decode-latency optimization still unproven; full goal active.

Decode priority from existing matched replay0/basic device profiles: scalar-hoist mean complete T1 batch75,896us vs FC80 control74,707us (three batches each, same1384ops); 40 `MUL_MAT_ID_NX` expert gate projections sum7,030us vs7,027us, 40 `MUL_MAT_ID` expert-down sum6,378us vs6,234us, and121 `RMS_NORM+MUL` sum9,659us vs9,538us. MoE gate+down are~13.4ms of75.9ms instrumented batch; scalar-hoist does not improve them. Individual op sums are attribution, not necessarily additive wall time. Next decode optimization should target these repeated paths or normalization with frozen correctness gates, not expect the prefill scalar-hoist alone to raise token rate.

## Scalar-hoist returned profile localizes prefill gain to preparation

Sep27 15:40 local: scalar-hoist replay0/basic profile4 native0, exact four logits/tokens bit-identical to FC80 stage-control profile4; CPU bounded first4 selected tokens match but logits differ. Baseline three hashes independently restored/no inference live. Strict tagged stage parser requires complete1982/T9batch and256validstage records: scalar prep127543876us, product_issue1230us, export1768883us, worker129722380us, jobs16390, products52280. FC80 same shape/counts: prep214160734us, issue1232us, export1761807us, worker216332279us, samejobs/products. Prepdelta=86616858us (40.45% of FC80prep); workerdelta86609899us. Prompt eval131774.61ms vs FC80control~218517.41ms, delta86742.8ms in ordered instrumented comparison, closely tracking prep. Product-issue is command issue, not pure HMX; export may include HMX completion. Three complete1384/T1 decode batches average75896us (75296..76250), compared with FC80~74707us/C202~74977us; no proven decode improvement, maybe slight diagnostic regression. `parse_stage_profile.py --ops1982` rejected because it expects exactly one one-token head; no T9 category claim from that parser. Full128 equality above remains exact. Candidate is compelling diagnostic prefill efficiency, not production promotion or full goal completion.

## Scalar-hoist full128 exact; CPU divergence unchanged at93

Sep27 15:36 local: matched scalar-hoist repeat session29689 terminal0, nativeA/B0, allthree baseline hashes restored independently/no inference live. Full128 logits and selectedtokens A==B bit-exact, and each A/B==archived C202 activation-low-reuse control bit-exact. CPU128 independent reference first selected-token difference93 in both (CPU60239/NPU3812), first logit-bit difference0; no CPUcorrectness improvement. Prompt eval131388.68/131355.29ms vs C202 repeats217261.47/217255.66ms, ~39.5% lower in ordered instrumented matched-family comparison. Traced decode12.74/12.86tok/s vs C20212.94/12.83tok/s, no convincing decode gain. Exact full-output equality is stronger than shortcapture but does not prove all inputs/productioncoherence or MTP. Next basic-profile4 stage decomposition vs FC80, then independent uninstrumented ABBA if measuring practical speed. Full goal active.

## Scalar-hoist short capture passes exact and own-input gates

Sep27 15:26 local: scalar-hoist capture session73865/nativeRUN_EXIT0, prompt eval131329.23ms vs matchedFC80stagecontrol~218517ms, two-token reported12.41tok/s not benchmark. All70 raw frames14namesx5occ bit-identical to stage-timing control; both logits rows bits/tokens identical to control. CPU128 boundedfirst2 selectedtokens match but logits differ (NMSE.005190/.001314), noCPUbitagreement claim. Independent baseline3 hash restoration verified, no inference live.

Eight Q8_0 own-input GGUF frozen gates threshold1e-7 passed on T9occ2 andT1occ4: ssm1.74528379e-12/2.37376283e-14, up1.31141976e-14/1.72705649e-14, gate1.07053158e-14/1.73451528e-14, down5.10350439e-13/4.7011909e-15. Rotated-weight negatives on ssm/downT9 andT1 fail unchanged threshold with NMSE2.01364224/1.99739199/2.02216418/1.92381248. Prior all2080Q8_0blocks for same frozen inputs match installedCPU quantizer bytes, but not CPUgraph activationbytes.

This is strong short-capture numerical evidence and ordered instrumented prefill-efficiency signal; not untraced ABBA, full128repeat, CPU128 agreement, production cache coherence or MTP. Next matched full128A/B strict bits/control/CPU firstdivergence, baseline restore, then stage profile for returned prep attribution if performance remains relevant. Full goal active.

## Scalar-hoist signed control archived and same capture live

Sep27 15:23 local: candidate archive backup-hmx-scalar-hoist-controlled-candidate completed signed with host390778A550001CAA24B52F599BF6424CEA0660614D2F81437E511D2FDA381961, DSPD9862347AC10DDDC1B95FC79FC21D06B346B532E76277FE14FC535CACE475F82, catalog6A2E535C8CD93AD22DF29D15D1C6FC0AF33FF1426C764C94AF1BD7F249B338EF. Cache+ninja show bothweight-scale-hoist and scalar-hoistON; host matches stage decoder. Signed DSP disassembly has separate hmx_prepare_low_operands helper and worker call, zero matched HVX register instructions in each; helper may call standard libraries. Archive hashes.json independently verified; baseline DSP/catalog restored before runtime. OFF-default flag guarded by hoist, dedicated builder `scalar` mode and 19otherbuilders/29configurelines resetOFF. Buildterminal0, archive terminal0, source diffcheck0, PowerShellparse0, twelve offlineMoE tests pass0. No speed/correctness claim from assembly.

Current scalar-hoist capture session73865/PID117596 started15:23:09, matched three candidate hashes installed, same Hello/n2/seed1234/temp0/Replay1/14tensor names. Initial filter string looked overescaped in printed command, but trace frame enumeration proves all14 intended names each appear twice in28completeT2warmupframes; no filter gap. At15:25:48 native still live, trace519920bytes and detail22143bytes, no prompt/decode or logits yet. Preserve matched installed files until terminal/wrapperrestore; no overlapping build/inference. This is not yet evidence that scalar hoist avoids previous over-budget behavior or passes output gates. Goal active.

## Scale-hoist capture stopped as over-budget; baseline restored

Sep27 15:17 local: controlled capture had no prompt-batch output since14:09:15, while matched C202 T9 prefill was~217s; after~69min and explicit process/hash preflight, stopped only campaign llama-cli PID328756. Stop-Process returned while Windows briefly still listed process; same wrapper session61732 then terminal/nativeRUN_EXIT=-1, threw expected nonzero stage result and ran finally restoration. Independently verified no inference processes remain and all three baseline hashes restored: hostC9E4D8ADE6884DA62DF7A597DCEAC6582AC036236C06134231A405BAF8E6399E, DSPD2D98EDEA1BDFD8652EE0766C4B7707AE1209E335164E6115478091C304B68CD, catalog5CB7F7EF8F6D6E88C5247D4EB5AC43CA527D7DF15BA66C547CBC25A3507CE09F. Candidate's28completeT2warmupframes matchcontrolbytes; no T9 prompt or logits/nativecompletion evidence. Do NOT call capture numerically passed, timed benchmark, proven DSP deadlock, or locate stalled instruction. It fails practical capture latency budget and is unqualified. No data deleted. Next independently archived no-HVX scalar-hoist controlled candidate, explicitly validate final signed worker/helper disassembly and normal exact/negative/model gates before any promotion; then return to gathered expert/CPU93/productioncoherence/MTP/full goal.

## Live native host wait-chain snapshot, no visible cycle

Sep27 15:06 local: session61732/PID328756 confirmed live and samehandle polled; detail log22144bytes/tensor519920bytes both unchanged since14:09:15. Read-only existing inspect-wait-chain.ps1 queried all46 nativehost threads via Windows Wait Chain Traversal. Every thread reported one node, no WCT errors, no DETECTED_CYCLE=1 and no multi-node chain. This excludes a WCT-visible host wait-chain cycle at this instant only; WCT does not report DSP execution state or prove forward progress/nondeadlock. No restart/kill/build/install. Goal active.

## Native Q4_0/Q8_0 expert up-gate oracle lane added

Sep27 15:04 local: same61732/PID328756 remains live, no NPU binary/source/install/inference overlap. Extended offline selected-expert checker with q4_0_q8_0_native_dot1: originalGGUFQ4_0only, raw18byteblocks/finiteHFscales, installedCPUQ8_0from_float+Q4_0vec_dot traits, nrc1 per output row, fixed1e-7 gate/selectedIDmapping/packedQ8_0rowSHA. ExistingQ4_1/Q8_1native path refactored through type-validated shared primitive; prior eleven tests still pass. This provides a nativeCPUprimitive lane for Q4_0 up/gate expert weights (K2048,N512,E256) when future actual expert captures exist. It is not graph activation-byte proof, not full CPU graph equivalence, and no causal claim about CPU93.

New analytic K64/N3/twoexpert/twoslot/twotoken nonzero nibble fixture with signed integer Q8_0activations/scale1 passes full nativeoutputs and CLIpositive; rotatedIDsgate fails unchanged threshold, packedactivationSHA remains same; malformed rawshape/duplicateIDs reject. Initial generated expert codes repeated mod16, making rotatedIDs spuriously pass; explicitly offset expert1 codes by3 and re-ran. Twelve unittest methods pass0/CLIhelp0. Negative-fixture distinctness is a required invariant, not evidence of actual model behavior. Full goal active; current capture still pending.

## Active CPU Q8_0 quantizer matches Python on six actual frozen inputs

Sep27 15:02 local: session61732/PID328756 confirmed live, samehandle polled, no competing inference or candidate change. Read-only local CPU trait8 from_float applied to existing complete residual-stage-timing capture inputs (not live output/model inference). Compared full packedQ8_0 bytes with gguf Python quantize: attn_post_norm-0/final_output-0/ffn_swiglu-0 occurrence2 T9 shapes[9,2048]/[9,4096]/[9,512], and occurrence4 T1 equivalents. All2080blocks match: zero HFscale-byte changes and zero code-byte changes. InputF32SHA T9:3d4c5463.../ce365073.../0805c4ad...; T1:1eb557f7.../ca12786c.../d77edb0b.... Probeexit0.

This rules out Python-vs-installedCPUQ8_0 quantizer mismatch for these exact frozen NPU-side inputs and supports existing plainprojection own-input gates. It does NOT establish original CPU graph inputs/packedbytes, arbitrary activation equality, expert-downQ8_1 behavior or CPU93 divergence cause. Prior syntheticQ8_1 tie difference remains separate and real, not contradicted. Full goal active; native capture result pending.

## Complete preparation-function probe qualifies scalar-control mechanism

Sep27 14:58 local: same61732/PID328756 confirmed live. Extracted archived struct and complete hmx_prepare_low_operands into in-memory standalone compiler probes, samev81/HVX/HMX/O2/xqfflags, noLTO linking/no file writes. Default function assembly789lines/frame640bytes includes scaleconversionHVX and localzeroingHVX. Loop vectorization/interleave/unroll restriction removes conversionHVX but leaves vxor/vectorzero stores; therefore loop pragma is NOT sufficient to promise a wholly scalar preparationfunction. Standalone/inlining context differs from finalworker, so frame/instruction numbers are not runtime performance evidence.

Function target("no-hvx") supported by HexagonClang19.0.07 with-Werror, exit0:786assemblylines/frame264bytes, no matched vector-register instructions. Scalarhalfconversion/rounding retained; memset/memcpy/halfconversion callees remain outside function-level guarantee. This provides stronger futurecontrol: explicitly isolated no-hvx preparationhelper (consider noinline to keep target boundary), then disassemble complete LTO-linked signed worker/helper/callees as applicable before capture. No candidate source/build/install changes made. Full graph/lifetime/arithmetic gates remain required. Full goal active; capture lacks native terminal result.

## Standalone scalar-hoist compiler control verified, not installed candidate

Sep27 14:56 local: session61732/PID328756 still live, unchanged candidate. Read actual archived compile flags: v81/HVXv81/HMX/fvectorize/enable-xqf-gen=true/O2/LTO. Standalone stdin-to-stdout HexagonClang19.0.07 assembly probe with corresponding architecture/vector/O2 flags (noLTO linking) compares32HF->F32 conversions. Default emits vmemu/vsplat(0x3c00)/vmpy(hf,hf)->qf32/qf32->sf/vshuff/store, matching candidate's new sequence. Same loop with clang loop vectorize(disable) interleave(disable) unroll(disable) emits scalar memuh/__extendhfsf2/memw loop, no HVX register instructions. Compilerexit0; no source/binary files written and no candidate build/install.

This confirms a concrete future scalar-hoist control mechanism, not full-function/LTO guarantee or numerical/performance proof. After current native closure, separately archive candidate with narrowly scoped pragma and inspect final signed worker for remaining compiler-generated HVX before any claim. Keep original output/provenance/negative/capture/full128 gates. Full goal active.

## Native nrc2 stride fixture passes; expert-down source uses nrc1 fallback

Sep27 14:53 local: same61732capture live. Offline suite eleven methods pass0 (no skips observed). InstalledQ4_1 trait advertisesnrows2; new native primitive fixture K512/twoweightrows/twoactivationrows passes independently decoded integer dot including m-1 offset. bx/by are byte rowstrides, bs is FLOAT output stride (source s+bs), testedbs4 with sentinel gaps untouched. nrc2 and four corresponding nrc1 results match exact analytic integer outputs. This verifies primitive stride/layout under exact arithmetic, not arbitrary reduction equivalence.

More relevant source evidence: ggml-cpu.c1477..1531 generic MUL_MAT_ID one_chunk explicitly calls vec_dot(...nrc1); ordinary MUL_MAT may selectnrc2. Expert graph first attempts tiled path, but tiled.cpp598..640 rejects ARM without AVX and its type gate excludesQ4_1 regardless. Source therefore supports nrc1 oracle relevance to generic Q4_1 expert-down. Actual CPU graph runtime-path/activationbytes and source-to-DLL provenance remain unverified; do not replace capture gates with this inference. Next real expertID/input/output capture after native closure remains required. No live-candidate change. Full goal active.

## Windows accelerator device state checked during pending capture

Sep27 14:50 local: session61732/PID328756 confirmed live, no output/native terminal result. Read-only CIM shows Snapdragon X2E94100 Hexagon NPU and Qualcomm FastRPC Device both StatusOK/ConfigManagerErrorCode0. NPU driver30.0.228.10000 (Qualcomm, driverdateJul20local2026), FastRPC685.13804.20.0 (Qualcomm, Jan11local2026). This is device-enumeration/driver provenance only, not DSP execution progress, no-fault proof or deadlock exclusion. Initial broad regex also matched Microsoft Input Configuration Device; it is not accelerator evidence. No device reset, driver change, inference overlap or live-candidate mutation. Full goal active; continue native-result monitoring.

## Native MoE CLI positive and routing-negative gates verified for both layouts

Sep27 14:49 local: session61732/PID328756 live, samehandle polled, tensor snapshot still519920bytes/14:09:15mtime. No native result or candidate change. Offline suite ten methods pass0. New end-to-end main() test uses synthetic88-byte frames, mocked GGUFReader metadata/originalQ4_1 bytes, and real installed CPU quantizer/nrc1dot; no model load or NPU call. Independently calculated affine expected outputs, twoexperts/twoslots/twotokens. Shared and per-slot input layouts each produce NMSE0/nativeexit0. Rotated expert IDs each produce nativeexit1 with unchanged threshold1e-7 and negative-control label; nrc1 scope disclaimer verified. This closes a CLI wiring/gate gap, not actual model or CPU graph correctness. Full goal active; runtime capture still pending.

## Live warmup snapshot matches control bytes; changed HMX worker not qualified

Sep27 14:46 local: session61732/PID328756 live and samehandle polled. Windows log discovery matching Qualcomm/Hexagon/FastRPC/DSP/NPU exposed no matching log; not proof absence of DSP faults. Read-only strict88-byte-frame snapshot comparison against complete residual-stage-timing control: candidate28 complete frames versus control70; all candidate header/name/payload bytes match corresponding first28 control frames, all F32 finite. First model.input_embed[2048,2,1,1] has3374nonzero values; last ffn_shexp-0[2048,2,1,1] has4096. No new trace write since14:09:15 and no logits/native terminal result.

Snapshot is warmup-only evidence, not complete capture/output qualification. Host ggml-hexagon.cpp6004 excludes plain HMX for m<=HTP_MM_HMX_MIN_NROWS(4), so T2 warmup output equality does not establish execution/correctness of changed HMX residual worker during pending T9 prompt. Do not infer first-QKV location, deadlock or performance from callback stop point. Candidate remains installed unchanged; full goal active.

## Hoisted vector load footprint checked; F32 branch also encounters HVX setup

Sep27 14:44 local: same session61732 and nativePID328756 confirmed live. Source/layout review rules out the hypothesized tile-boundary overread from the new128-byte scale load: Q4_0 packed576/aligned640 with scaleoffset512 gives load[512,640); Q8_0 packed1088/aligned1152 with scaleoffset1024 gives[1024,1152). DMA copies only packedbytes into padded destination; upper64bytes are padding, not meaningful scales. No claim about initialized padding or arbitrary lane arithmetic. Selected32 HF scales correspond to lower64bytes; shuffle/store supplies32F32values.

Disassembly343A also places vsplat(half1.0) and stack vector store at273f8/27400 before the low-operand weight-type path, so the new HVX footprint can affect F32 activation-only jobs too, despite source-level F32 early return. Source arithmetic unchanged does not imply compiled resource use unchanged. Queue job stays in scope through synchronouspop, raw-weight nextDMA remains deferred, and local worker frame640bytes alone does not prove overflow of16384-byte queue stack. No established runtime failure cause; scalar-disassembly control remains next experiment after native closure, not a live-candidate edit.

## Scale-hoist compiler introduced HVX; missing explicit lock is not established bug

Sep27 14:43 local: session61732/PID328756 remains live with no native result. Read-only disassembly of signed archived workers: FC80 has no matched vector-register instructions; 343A introduces v3.h=vsplat(0x3c00), stack vector store, vmemu load, vmpy(hf,hf)->qf32, qf32->sf conversion and shuffle/store around the hoisted scale conversion. This is a genuine compiled execution-path change, not merely fewer scalar conversions. HMX queue source drops HVX context when idle and has no explicit lock before work, initially suggesting a context-acquisition risk. However installed computev81 qurt_hvx.h documents automatic exception-driven HVX resource acquisition and deprecated explicit locks, so absent explicit lock is NOT proof of failure. No DSP fault/terminal result observed, and no runtime instruction position known.

After native closure, consider a separately archived scalar-hoist control with loop vectorization disabled and verify disassembly contains no new HVX before testing; alternatively measure intentional HVX conversion as its own resource-aware candidate. Preserve arithmetic/output gates and matched host/DSP/catalog. Do not change live candidate or attribute the pending run to HVX/stack/driver based only on this static evidence. Full goal remains active.

## Nonzero Q4_1 multiblock native primitive fixture passes

Sep27 14:40 local: session61732/PID328756 revalidated live, no new native output. Offline suite now nine methods pass0. Added seeded K512/N5/two-expert/two-slot/two-token nonzero nibble-code fixture: per-block activation extrema+127/-127 give exact Q8_1 scale1, integer signed activations avoid quantizer ties, weight d1 and affine m-1/0. Independent integer decoded-weight dot products match native nrc1 and scalar reference exactly across16 blocks. Wrong expert map fails fixed1e-7 gate. This covers low/high nibble order, signed activation code dot, multiblock addressing and affine contribution under exactly representable arithmetic; it does not establish arbitrary floating-point reduction equivalence or actual model correctness. Installed candidate and live capture preserved; full goal remains active.

## Multiblock native expert mapping fixture passes; capture still live

Sep27 14:38:53 local: session61732/PID328756 confirmed live and same handle polled with no output or native terminal result. Tensor file remains519920 bytes, last write14:09:15. No candidate mutation, restart, or overlapping model inference. System/Application warning/error check since capture start found HAL event20 for an ACPI realtime-clock query, not an observed DSP failure; causality is not established.

Offline test adds K512/N3/three-expert/two-slot/three-token Q4_1 affine-only fixture with distinct per-block offsets and distinct activation magnitudes. Native nrc1 and scalar reference match analytic outputs exactly; repeated native output and provenance match. Rotated expert IDs preserve packed activation SHA but fail fixed1e-7 NMSE gate. Duplicate/out-of-range/noninteger IDs reject. Eight unittest methods pass. This validates multiblock addressing and selected-expert/slot-token mapping for a synthetic affine fixture, not general code-dot reduction, CPU graph equivalence, actual model correctness, or the CPU93 divergence cause. Full goal remains active.

## Native expert-down nrc1 oracle lane added; controlled tie fixture separates lanes

61732/PID328756 reconfirmedLIVE14:33:50localSep27 andsamehandlepolled; no nativecompletion/restart/kernelchanges/othermodelinference. Offlineprimitives only.

Added check_frozen_moe_math.py --reference q4_1_q8_1_native_dot1: originalQ4_1required, fullselectedexpertIDs/sharedorper-slotlayout, validatesraw20-byteblockshapes/finiteHFscales andCPUtraitABI/functionpointers/type/nrows. InstalledactiveCPUQ8_1from_float plusQ4_1vec_dot atnrc1 for everyoutputrow; returnsF32primitiveoutputs, prints librarySHAandpackedQ8_1activationSHA. Fixed1e-7gate/wrongexpertnegative unchanged. Explicitnrc1lane, NOTCPUgraphnrc2/GEMMequivalence/NPUpacking/actualmodelcorrectness.

Oracleunittestimports singleCpuTraits definition fromchecker (noABIduplication). Nativeaffineoffsetcase remains64exact. Newcontrolledtiecaseweightonlyselects+.5activationelement, d1/m0/amax127: scalarreferenceprojection1 versusactiveCPUprimitive0, exactlyexpected isolatedquantizerdifference. Invalidrawshape rejectsbeforecall. Sevenunittestmethodspass0/CLIhelp0. NotproofthisfixtureoccursinmodelorcauseofCPU93; realfrozenexpertcaptureawaitscurrentrunclosure. Fullgoalactive.

## Active CPU Q8_1 tie behavior and affine primitive verified via traits

61732/PID328756 remainsconfirmedLIVE, samehandlepolled; no nativeexit/restart/competinginference/kernelbinarychanges. OfflineCPUprimitives only, notmodelinference.

Read actual ggml_type_traits_cpu ABI (twofunctionpointers,intenum,paddedint64nrows;32bytes), from_float signature and vec_dot signature. Installed ggml-cpu.dll exports ggml_get_type_traits_cpu, so directsymbolnonexport isnotendofsafeprimitiveaccess. Native test initializesCPU, queriesQ8_1type9 andQ4_1type3, validatesvec_dot_type9 andnonnullfunctionpointers. Activefrom_float at ties127/+.5/-.5/+1.5/-1.5 returns127/0/0/2/-2 (ties-even) versus scalarref127/1/-1/2/-2. Difference isconfirmedfixture-level compiled behavior, consistentwith ARMvcvtn source. Do NOTpatchNPUrounding orattributeCPU93 basedonlyonthisfixture.

NativeQ4_1dot pointer nrc1 withsingleblockd0/m2/activationones returns64, matchingaffinePythonoracle exactly; testsrealoffsetpathbutNOTgeneralARMdotreduction/graphnrc2/GEMM equivalence. Sevenunittestmethodspass0. CPU DLLSHA70C3D396D0127A675F92560533E1CC71E399C9305B679F4FC96625DD2831515D, baseDLLAC592F1... referencequantizer304blocks evidenceunchanged. This identifies separate scalarreference versus activequantizerlanes; realfrozenexpertcaptures stillneeded. Nextnativecaptureclosure/exactgates thenproperIDs/input/output captureandnativeprimitivecounterfactuals. FullgoalCPUagreement/coherence/MTPremainsopen.

## Q8_1 reference bytes match installed ARM64 export on304 blocks

61732/PID328756 reconfirmed LIVE14:28:39localSep27; samehandlepolled/no restart/otherinference/build/kernelchanges. Hostthreadsnapshot shows6UserRequest/36Unknown/4EventPairLow wait states; notproofDSPterminal/deadlock. Preserve matchedfiles and awaitnativecompletion.

Python isARM64 and installed ggml-base.dll exports quantize_row_q8_1_ref. Addednative-export comparison to campaignoracle unittest:19rowsx512=304blocks, seededmixedsign rows at2^-24..2^8 pluszeroandsignedhalf-ties. Allreferenceintegercodes and storedHFd/HFs bit-identical to nativecompiledexport, sixunittestmethods pass0. Installed baseDLLSHAAC592F1C55B02F44D0AC43ECB021D35E3A8929E69484C3370FF0A11CD85C47A4. No NPU inference or model load involved, tinyofflineCPUquantizer only. This closesreferencequantizerfixture bytecoverage, NOT actualoptimizedCPUactivationbytes/ARMdotreduction/NPUpacking/fullmodel correctness. ggml-cpu.dll doesnotexport requestedQ4_1dotfunctions directly; no compiled-dotclaim. Nativefixturetest skips explicitly ifDLLabsent. Realfrozenexpertcapture remainspendingcurrentrunclosure.

## Q4_1/Q8_1 expert-down reference algebra implemented and tested offline

61732/PID328756 reconfirmed LIVE14:26:08localSep27; nativecompletionabsent, no restart/sourcekernel/binarychanges/competinginference. Pythonoraclework only. Preserve matchedfiles and pollsamehandle.

Read quantize_row_q8_1_ref, block_q8_1, genericQ4_1/Q8_1dot andARMimplementation. New check_frozen_moe_math.py --reference q4_1_q8_1_ref requiresoriginalGGUFQ4_1. F32amax/127/inverse/scaled values, ties-away reference codes; d storedHF, s storedHF from integercodesum times ORIGINALunroundedF32d. Parsesraw20-byteQ4_1blocks d/m/low16+high16nibbles, integerdot, F32blockscale products/offsetproducts andF64sum. ExplicitlynotcompiledARMbit equivalence oractualruntimeactivationbytes. No looseningfixed1e-7gate, wrongexpertIDnegativeunchanged. NativeCPUoptimizedQ8_1quantizer/reduction rounding maydiffer; frozen owninput andreference-code comparison neededbeforecauseclaims.

Fiveinmemoryunittestmethodspass: priorframe/mapping tests plus zeroactivation and signedhalf-ties, exactaffineoffset/code-dot syntheticcase, malformedrawshape and separatelyroundedstoredsum. Concrete mixedfixture: sumcodes1168, stored_d0.0259857177734375, stored_s30.34375 versus stored_d*sum30.351318359375; testguardsagainstincorrectlyreconstructingsumfromstored_d. CLIhelp0. Realmodelcapture/compiledCPUcomparison/rawruntimebytes remainunproven. Nextcurrentcaptureterminal/exactgates, thengatheredexpertcapture withactualIDs andproperup/gateQ8_0/downQ8_1referencecontracts; fullgoalCPU93/coherence/MTPstillopen.

## Frozen MoE oracle frame tests and actual expert metadata verified

Current61732/PID328756 stillconfirmed live14:22:21localSep27 and polledaftermetadata; no native terminal/restart/otherinference. Offlinework only. Added test_frozen_moe_oracle.py undercampaign(notllama tests): threeunittestmethods pass, covering completeF-orderframe/sharedbroadcast/occurrenceselection, seven frame negatives, correct/wrongexpertprojection, duplicate/missing/nonintegerIDs/nonfiniteweight. Initialbadmagicfixture replaced wrong bytepattern and didnotmodifyframe, so testfailed; corrected actualfirst4bytes and suitepasses. This tests Pythonoracle, notrealcapture/runtime arithmetic.

ActualGGUF metadata read: layer0 up/gate Q4_0[K2048,N512,E256], rawdata[256,512,1152]; down Q4_1[K512,N2048,E256], rawdata[256,2048,320]. Thus down is NOT Q4_0 or Q8_0. CPU Q4_1 usesQ8_1 activation scale/sum contract; checker nowexplicitly rejects q8_0reference for nonQ4_0/nonQ8_0weighttypes instead of falselycallingdown CPUmath. Dense down comparison remains onlydensemathematical oracle; separateQ8_1sumcontract neededforCPUruntimecompare. Graphcallback ffn_moe_topk and ffn_moe_down verifiedinsource (down before scaled/biased/reduced branches). No modelcapture orcauseofCPU93claim yet. Nextsame61732terminal/exactcapturegates, thengatheredIDs/input/perexperts captures andproperquantizationcontract.

## Frozen MoE projection oracle prepared while hoist capture remains live

Previous turn classified verifiedwait; SAME61732/PID328756 re-confirmed live at14:18:03localSep27 and polled again after offlinework. Native completion absent; preserve matched installed files, no restart/competinginference/build/kernelchanges. Hoist performance/output unqualified.

Added npu_campaign/check_frozen_moe_math.py, offline only, no captured modeltest yet. Reads complete88-byte frames with matching per-name occurrence counts; F32shared or per-slot activation layout explicit, I32topkIDs, per-expertF32output before routing reduction. Validates GGUF[K,N,E] dimensions and IDrange, dequantizes only selected experts, original dense/Q8_0 reference activation math, fixed1e-7local threshold. Wrong-expert mapping rotatesIDs1modE while holding inputs/outputs fixed; no loosenedgate. project function rejects duplicated per-tokenexpertIDs, missingweights/nonintegerIDs/nonfiniteweights orinputs/shape mismatch. Synthetic tiny per-slot projection arithmetic, swapped-expert difference and three invalidID controls pass; CLIhelp0. FrameI/O, shared-layout CLI, realGGUF/capturedexpert/runtime packing equivalence remain untested; no modelaccuracy claim.

Next after currentnative closure/exact gates: capture attn_post_norm-0,ffn_moe_topk-0,ffn_moe_up-0,ffn_moe_gate-0,ffn_moe_swiglu-0,ffn_moe_down-0 on matched CPU/NPU sameinputs. Verify actualcallback names/shapes and gather expertweights first. Original mathematical projection gates and wrongIDs must run on exactselected captured IDs; don't compare postroutingreducedoutput as per-expert matrixoutput. This is the next CPU93/gatheredprecision investigation, not automaticproofcause ornewruntimeprecisionimplementation.

## Weight-scale hoist archived; exact capture live

Latest continuation61732 stillconfirmed LIVE/PID328756 at14:16:52localSep27. No native terminal/restart. Read-only stack check: FC80worker allocframe0x188(392bytes),343Aworker0x280(640bytes), configured HMX_QUEUE_STACK_SIZE16384; no claimthis explains slowdown. Artifact snapshot contains28 complete warmup tensorframes, lastffn_shexp shape2048:2, no logits yet; in-progress snapshot is NOT qualification or prooffirstQKV hung (callbacks may await batchcompletion). Initial inlinePython quoting syntax rejected, correctedreadonly snapshot0. Preserve installedmatchedfiles, pollsamehandle; fullcapturegates await actualterminal. Moreelapsedthanprecedingcontrol doesnotjustifyrestart/kill orspeedclaim.

Continuation verified SAME61732/PID328756 remains LIVE at14:14:25localSep27; longer than prior control, observation timeout is not terminal and no restart. No speed claim; could regress. Read-only archived hmx_residual_worker disassembly: FC80control666lines/7 static __extendhfsf2 sites/6 __truncsfhf2/1roundf;343Ahoist693lines/6extend/6trunc/1roundf. Static call-site counts and listing size are not dynamic invocation counts, attribution or performance proof. Initial PowerShell foreach pipeline syntax rejected, corrected read-only command completed0. Preserve matched installed files and poll61732 until native terminal/wrapperrestore, then exact gates regardless timing direction.

8266 terminal0. Guarded signed archive backup-hmx-weight-scale-hoist-controlled-candidate completed0 with all C202/timing/hoist controls verifiedcache+ninja, source/sharedheaders/host+Dspcache/ninja saved. Matched host390778A550001CAA24B52F599BF6424CEA0660614D2F81437E511D2FDA381961 unchanged, DSP343A68C87C5CB2677DE7D58571BC1EBF4BEBD372F75097CE0E26332D3505B136, catalogDB4095FCF9D4D92EC2F873D6913119F193DCB9AD3D663185B07A8FBA2A016CC6, hashesmanifestadded; three archive hashes independently read.

Capture qwen-layer0-weight-scale-hoist-npu LIVE confirmed session61732, same14names/Hello/n2/seed1234/temp0. Preserve matched installed threefiles until native terminal/wrapperrestore; no competingbuild/inference/sourcecandidatechanges. Next SAME61732 ->baseline3hashes ->strictall70tensorbits versus qwen-layer0-residual-stage-timing-npu (or C202) ->strict complete2logitbits ->own-input8Q8T9/T1gates andssm/downwrong-weight negatives ->full128repeat/control/independentCPU ->separatebasicprofile4 returnedpreptiming versus FC80 (thenuninstrumentedbenchmark beforepromotion). Capturetime alone notperformance qualification; no measuredhoist gain yet. Gatheredexpert precision/firstdivergence/productioncoherence andMTP stillnext fullscope priorities after bounded prep experiment.

## Residual weight scale hoist implemented; build live

New OFF-default GGML_HEXAGON_HMX_WEIGHT_SCALE_HOIST_REFERENCE requires Q4 or Q8 operand correction. In hmx_prepare_low_operands non-F32 branch, each rawtile's32 HF column scales loaded/converted once into128-byte local F32array, using same Q8offset1024/Q4offset512. Originalkk/cc signedcode decode/highHFsubtract/lowHFrounding expression retained, scalar product uses cached exactF32scale. OFF original loads/conversions untouched. F32activation-onlyearlyreturn unchanged. No ownedplane/hostformat/GGUF/queue/DMA change. Existing19builders/29configurelines resetflagOFF; dedicated build-hmx-weight-scale-hoist-reference.cmd retainsC202plusobservable timingON and matched3907host. Archive RequireWeightScaleHoist validatesactualcache+ninja. Source diffcheck0/PowerShellparse0.

Build SAME8266 confirmed LIVE, hostninja no work, DSPobjects throughmatmul compiled/final link pending. No candidate install/inference/source changes afterlaunch. Next poll8266 terminal -> guarded signed archive backup-hmx-weight-scale-hoist-controlled-candidate (allC202/timing/hoistguards, expectedhost freshlyverify3907) -> manifest ->14name2token capture and exact70tensor/2logitcontrolbits, own-inputQ8T9/T1/negative gates ->full128strictcontrol/repeat/CPU/restoration ->paired returnedprep timings versusFC80, plusuninstrumentedcontrol beforeperformancepromotion. No speedclaimyet, noCPUcorrectpromotion. This is bounded diagnosticprefillefficiencywork; explicitlyreturn togatheredexpertprecision/firstdivergence/productioncoherence andMTP afterward.

## Tagged stage profile closed: counters observable; preparation dominates diagnostic prefill

40585 terminal/native0. Strict complete4 logits and selected tokens bit-identical to C202 activation-low-reuse profile0. Independent CPU bounded4 token gate0, logits differ/not fullqualification. Baseline three hashes independently restored C9E4/D2D/5CB7, no inference/build live. New decoder/profile-return path is observably working, unlike FARF. Stage parser passes complete1982 T9 batch,256 operator records; identity/validity/count/timing/tile checks pass. Five additional parser negatives (wrongtoken, producttilebound, jobtilebound, incompletebatch, workeroverop) pass; eleven negatives total plus positive multicolumn fixture. This is parser coverage, not runtime PMU/overflow edge coverage.

Complete1982 prefill wall217659742us. Returned stage sums: prep214160734us, product_issue1232us, export1761807us, worker216332279us, jobs16390/products52280. Prep98.9962% of measured worker and98.3924% of complete batch wall. Largest QKV layer0 prep3395605us/worker3427601us, products768/jobs256; layer1prep3394453us. Product issue excludes outstanding completion that can be charged to export; never call1232us pure HMX execution or hardware peak proof. Stages sum attribution not additive host latency; instrumentation/qtime quantization/withinrun timing must remain explicit.

Threecomplete1384 decode batches74707.333us (73975..75384) versus C20274977.333us essentially unchanged. No residual-stage records in1384 selected batches; parser correctlyrejectsabsence, notzero. This is consistent with T1 taking HVX rather than residual HMX and means prep optimization addresses diagnostic prefill, NOT a demonstrated single-stream decode gain. Profileprefill218517.41ms/reported4.26tok/s loggingaffected, notthroughputbenchmark. Full128instrumentation equivalence, PMUruntimepreservation/overflow edges and independentbenchmark remainunproven.

Next bounded efficiency candidate: decode32 raw column scales once per raw weight tile rather than1024 repeated scale loads/HF->F32 conversions inside kk/cc; OFF-default separate control, same scalar products/rounding/high operand/ownedplanes/lifetime. Requires own-input exact/math/negative gates and full128control equality, then returned prep timing and uninstrumented control. This speeds numerical research, not goalcompletion. Do not remain solely in diagnosticprefill: after bounded experiment, investigate unqualified gatheredexpert prefills/first-divergence and productioncoherence; CPU93 and MTP remain explicit unresolved single-stream requirements.

## Stage-response matched candidate archived; runtime profile live

90354 provenance rebuild terminal0. Guarded signed archive backup-hmx-residual-stage-response-controlled-candidate completed0, manifest added and three hashes independently verified: host390778A550001CAA24B52F599BF6424CEA0660614D2F81437E511D2FDA381961, DSPFC805363A1B0393F9E66201C9864A422046861BB1FE890A7DF876D4A85CDB605, catalogABF0F851FC4018CC0EBA35B59551D0B4EADDDBC3713EF5A1C2056E82FCC2F2C7. Source/sharedheaders/host+Dspcache/ninja archived. Newhost Q5format/compileddecoder and all C202/DSPtiming controls guarded.

Matched residual-stage-response-basic-profile4-replay0 LIVE in confirmed session40585, DeviceProfile1/Replay0/fourtokens/Hello/seed1234/temp0. Preserve installed matched newhost/DSP/catalog until native terminal/wrapper restore, no overlapping build/inference/candidate edits. Next poll SAME40585 -> independent baseline3hash restoration -> strict four-row logits/tokens versus activation-low-reuse-basic-profile4-replay0 -> independent CPU boundedfourtoken gate -> complete1384 decode attribution plus complete1982/T9 residual counters with new parser. Runtime observability, exact preservation and timing still unproven until closure.

Added parse_residual_stages.py: pairs stage record to next exact operator name/tensor/shape, complete selected batch count, valid/positivejob/product/tile bounds, stage sums<=worker<=operator, required activationtoken axis. Reject absent stage records (not zero), malformed/duplicate/orphan/unpaired and gathered operators. Tested positive multi-column job (sixproducts/job) plus six invalid/identity/timing/duplicate/missing negative cases; archived uninstrumented log rejects absentcounters. Initial overly tight threeproducts/job bound corrected before runtime to matrix tile bound. Independently verify nativeexit/exactoutputs before relying on parser; logs alone not qualification. PMU runtime preservation/overflow edge/full128 still untested. Full goal active.

## Tagged residual-stage response implemented; matched build live

22038 is now terminal0. Provenance rebuild launched and confirmed LIVE as SAME90354, host compiled/linked then DSP objects through matmul compiled, final link pending. Poll90354, not22038. No source changes after rebuild start. Final host hash must be re-read after rebuild before archive.

FARF-only route replaced by operator-local six uint64 accumulators in DSP context: prep/product-issue/export/worker ticks plus jobs/products. Residual jobs receive pointer to context totals, synchronous pop guarantees lifetime/read ordering; totals reset before each proc_op_req. Existing descriptor size unchanged: BASIC only replaces normally unused pmu words with shared version tag0x48525331, four aggregate qtimer-to-us values, jobs/products and validity. Conversion/count values exceedingUINT32_MAX mark all returned values invalid/zero; PMU mode keeps actual PMU counters, TRACE unchanged. profile_data now zero initialized so BASIC unused slots deterministic. OFF-default instrumentation remains compile gated. No arithmetic/plane/queue/DMA changes. Product issue interval may leave HMX completion in export, not pure execution time.

Matching host recognizes tag only opt_profile1 and prints residual-stage-v1 with operator names/shapes, validity and explicit units. FARF logging removed. Dedicated timing builder now builds host target before explicit C202 DSP configure; all other18builders reset timingOFF. Archive adds explicit ExpectedHostHash (old2226default unchanged), compiled newdecoder-string and Q5format-cache guards when timing/newhost requested; archives host cache/ninja plus htp-ops.h/htp-ctx.h. Archive parse0/source diffcheck0. Host build20E5E7718A5C10FC62E815EB0C1A1E3A05AD96DA23A69E35C90FDC5E1CE79094 completed, ordinary preexisting host warnings only. SAME22038 confirmed LIVE DSP link pending after matmul compiled; no candidate installed/inference.

A shared-header comment was clarified after build started (no semantic/code change), so rerun dedicated builder after22038terminal to align archived source/cache/host/DSP provenance before recording final hashes. Do not archive while build live. Next terminal/rebuild -> new guarded archive backup-hmx-residual-stage-response-controlled-candidate using freshly verified ExpectedHostHash and all C202/timing guards -> hashes manifest -> guarded DeviceProfile1 four-token run and exact C202 profile bits/CPU bounded gate/baseline3 restore -> require observable stage records and strict parser. Runtime returned timing, edge/overflow coverage, PMU preservation and full128 equivalence remain unproven; no speed/correctness promotion. Full campaign active.

## FARF stage capture closed: exact preservation, no observable timing

33818 terminal/native0. Strict all70 selected internal tensor records identical to C202 activation-low reuse, finite gate0; both complete2 logits and selected tokens bit-identical0. Own-input ssm_out T9 q8_0 NMSE1.74528379e-12, native0. Rotated-weight-row ssm_out/down negatives reject unchanged threshold1e-7 at2.01364224/1.99739199, native1; sessions4313/94464/9054 all terminal. Baseline host/DSP/catalog independently hash-restored C9E4/D2D/5CB7. No inference/build live.

Prefill218263.16ms versus C202 capture217466.16ms (instrumentation/log perturbation possible; not independent overhead benchmark), reported2token12.39tok/s not benchmark. Count of HMX_RESIDUAL_STAGE in completed detail and stderr logs is ZERO. FARF summary route is therefore a dead end for observable host timing in this campaign; no stage data claim. Full128 timing-candidate equivalence not run, and not needed to pursue unobservable counters further before transport repair.

Next verified transport design: aggregate worker counters per operator into existing profile response, explicitly tagged diagnostic payload only in BASIC profiling (never overwrite actual PMU mode); matching host decoder must recognize tag and print named stage units and provenance, bounds/overflow checks required. Keep normal descriptor size/PMU mode unchanged if feasible; do not silently relabel PMU values. Worker queue synchronously pops before main reads counters. Extend exact source archive to relevant shared/context headers, validate new matched host hash rather than assuming2226. Arithmetic remains unchanged. Read actual files and review descriptor initialization before implementation; no implementation or claimed returned timing yet. CPUagreement/productioncoherence/gatheredprecision/MTP remain open; goalfullscopeactive.

## Stage-timing candidate archived; capture live; FARF return path unproven

70760 terminal0. Guarded build/sign/archive completed0, exact source/cache/ninja saved in backup-hmx-residual-stage-timing-controlled-candidate. Host2226402B762C8E2233059CEEF1D686BE51731FBAD70B24BFA23A75F2C08A2CDD, DSP1865EEC335BBF708B91BB580BB13C2661917BAD22EFB186295657CC43C09BF38, catalogE0DCB1B32E48D899EADA7EBB0094BC5897DD3EBA7158C1B3C3A00B2F1878B67A; hashes.json added. Capture SAME33818 confirmed live PID1576688/start13:49:51localSep27, qwen-layer0-residual-stage-timing-npu, 14 names/Hello/n2/seed1234/temp0. Preserve matched three files until native exit and wrapper restoration; no competing build/inference/source changes.

Important measurement correction: historical retain-probe notes explicitly say FARF metrics not visible in host stderr; current capture has no HMX_RESIDUAL_STAGE records yet. Timer source compiles but observable return path is UNPROVEN. Absence is not zero timing; do not build claims/parser around unavailable counters. Finish current exact tensor/logit/own-input negative gates, then design verified bounded/aggregate profile return path or capture DSP logs. Existing profile PMU slots are real PMU data, not permission to silently relabel/overwrite them. HMX command interval versus completion/export ambiguity remains. No performance claim from this instrumented candidate; full goal active.

## Residual stage timing implemented; build live

New OFF-default GGML_HEXAGON_HMX_RESIDUAL_STAGE_TIMING uses existing HAP qtimer with compiler memory barriers. Worker-local uint64 counters, one FARF summary per synchronous residual job: shape/type/operand flag/product count/prep_ticks/product_ticks/export_ticks/worker_ticks. No fixed-capacity event buffer, no arithmetic/buffer/queue/DMA changes. Prep brackets low-operand helper; product brackets selection/clear/load-multiply loop; export includes paired HF extraction/identity multiply/F32 combine/bias/store. HMX completion can be charged to export: product interval is command-loop timing, NOT proven pure completed HMX execution. Worker includes setup/loop/timer overhead, log occurs after worker endpoint; logging still changes surrounding scheduling. Timer perturbation requires exact gates and independent uninstrumented comparison.

Existing18 builders/28configure lines explicitly reset newflagOFF; dedicated build-hmx-residual-stage-timing.cmd enablesON with exact C202 controls. Archive RequireResidualStageTiming validates cache+ninja. Initial generated reset patch rejected on PowerShell serialized string wrapper, no changes; corrected explicitstring patch passes. Source diffcheck0 and archive PowerShell parse0. Build70760 confirmed LIVE, objects compiled through matmul-ops, link pending. No inference or install. Next SAME70760 terminal -> guarded signed archive backup-hmx-residual-stage-timing-controlled-candidate with all C202 required flags plus RequireResidualStageTiming -> manifest hashes -> short capture/exact C202 tensor+logit gates and own-input negatives -> full128 equivalence if short gates pass -> parse stage records with completeness/shape/product/nonnegative checks. Do not interpret unexecuted timing as evidence. Full campaign scope/CPU/coherence/MTP remain open.

## Activation-low reuse profile closed: decode unchanged; prefill matrix costs dominate

19443 terminal/native0. All four logits/tokens bit-identical to B090 profile; independent CPU bounded four-token gate0 with different logits (not full qualification). Baseline three hashes independently restored C9E4/D2D/5CB7; no inference live. Three complete1384-op one-token batches mean74977.333us (74793..75227) versus B09074856.667us, essentially unchanged/no decode speed claim. RMS2048 fused mean8539.667us versus8568; head3019us versus3066.333. Profile reported4.59tok/s is logging affected, not throughput qualification.

Complete1982-op prefill batch independently counted: wall217076716us, plain MUL_MAT178222170us, MUL_MAT+ADD24502221us, MUL_MAT_NX13308123us, gathered ID_NX827643us/ID136549us. Largest individual QKV plain operators around3.41seconds each, shapes2048:8192 x2048:9 ->8192:9. Attribution covers complete operators, not pure HMX/preparation/export and not additive host latency. Remaining scalar weight-low preparation and paired exports are concrete next instrumentation/optimization targets; current per-op diagnostic controls cause enormous prefill overhead and are not production-promoted. Full CPUagreement/coherence/gatheredprecision/MTPremainopen.

Corresponding B090 complete1982 batch wall337060257us, plain MUL_MAT277814883us, MUL_MAT+ADD37866117us, MUL_MAT_NX20333615us, gathered ID_NX827993us/ID137026us. Plain category reductions99.593/13.364/7.025seconds account for the observed~119.984second DSP wall reduction; gathered categories essentially unchanged, consistent with implementation's plain-only scope. This is ordered profile attribution, not an independent benchmark or proof of internal stage isolation. Next safe task with no inference live: add separately gated residual-worker stage timing for operand preparation, HMX products and paired export, capacity-safe and OFF by default; archive/build/test exact own-input negatives and full control equivalence before relying on timings. Do not optimize guessed pure-HMX cost from whole-worker spans.

## Activation-low reuse fullpair closed: all128 bits preserved, prefill savings repeated

Following closure, matched activation-low-reuse-basic-profile4-replay0 launched in confirmed live session19443 (DeviceProfile1/Replay0/four tokens). Preserve C202 matched files until native terminal and wrapper restoration; no overlapping builds/inference/candidate edits. Poll SAME19443, then baseline3 hashes, strict four-row bits versus rms-native-output-basic-profile4-replay0, independent CPU bounded four-token gate and complete1384 batches via parse_stage_profile.py --by-weight. Profiled rate is not a throughput benchmark.

33033 terminal0/both native0. Strict complete128 A/B logits and selected tokens bit-identical0; each independently versus corresponding B090 control bit-identical0. Independent CPU both first diverge at unchanged93 CPU60239/NPU3812, same margins/logits; not full CPU qualification. Baseline three hashes independently verified restored C9E4/D2D/5CB7; no inference live at closure.

Prefill A/B217261.47/217255.66ms versus B090337350.42/337151.04ms, approximately35.60% lower with highly consistent ordered repetitions. Traced128 decode12.94/12.83tok/s versus B09012.83/12.89: no convincing decode gain. This reduces repeated diagnostic preparation work while preserving entire tested output sequence; not independent ABBA/untraced production qualification. C202 remains OFF-default and unpromoted. Next four-token replay-off stage profile and exact four-row control gate, complete1384 batch attribution; pure prep/HMX/export isolation still separate. Full CPU agreement/coherence/gathered experts/MTP remain open.

## Activation-low reuse arm A preserves all128 control bits; arm B live

Session33033 armA native0. Strict complete128 logits/selected tokens versus B090 wholecache-rms-native-output-repeat128-a exits0, all bits identical. Independent CPU token gate exits1 at unchanged step93 CPU60239/NPU3812, margins0.0516681671/0.104560852; later rows not interpreted as same-prefix. Prefill217261.47ms and traced128 decode12.94tok/s (9816.96ms), versus preceding B090 pair337350.42/337151.04ms and12.83/12.89tok/s. One ordered arm supports preserved output and prefill savings, not independent ABBA benchmarking, decode gain qualification or CPU-correct promotion.

SAME33033 armB confirmed LIVE PID1598760/start13:35:16localSep27. Preserve matched C202 host/DSP/catalog until native terminal and wrapper restoration; no concurrent inference/build/candidate edits. Next poll33033 -> strict128 A/B, independentB versus B090, independentCPU_B, independently verify baseline3 restoration. Then separate four-token stage profile using existing guarded profile mode. Full scope remains open including CPU agreement, production coherence, gathered experts and MTP.

## Activation-low reuse capture closed; full128 gates next

Full128 matched C202 repeat launched in confirmed live session33033, label wholecache-activation-low-reuse-repeat128-a/b. Preserve installed matched three files until both native processes finish and wrapper restores baseline; no concurrent build/inference. Poll SAME33033. Required closure: strict A/B all128 bits, each versus B090, independent CPU each stopping at first changed prefix, independently verified baseline three hashes. Do not restart from elapsed time or poll timeout.

92078 terminal/native0. All 70 selected internal tensor records (14 names, five occurrences) are bit-identical to the archived 8DC1 Q8-operand capture. Bounded two-row logits and selected tokens are bit-identical to B090; independent CPU two-token gate passes, but logits differ and this is not full-run qualification. All eight original Q8_0 projection gates pass: T9 ssm_out/up/gate/down NMSE 1.74528379e-12/1.31141976e-14/1.07053158e-14/5.10350439e-13; T1 2.37376283e-14/1.72705649e-14/1.73451528e-14/4.7011909e-15. Wrong-weight-row controls reject at the unchanged 1e-7 threshold: ssm_out 2.01364224 and down 1.99739199, both native1. Independent baseline host/DSP/catalog hashes restored C9E4/D2D/5CB7; no inference live at capture closure.

Prefill 217466.16ms versus preceding Q8-operand capture 337671.14ms: 120204.98ms/35.60% lower in this ordered capture comparison. Reported two-token decode 12.38tok/s is too short for a benchmark. Exact sampled tensor preservation removes the anticipated B090-versus-8DC1 attribution ambiguity for these frames. Savings remain capture-level evidence, not isolated preparation timing or independent benchmark. C202 remains OFF-default and unpromoted. Next matched full128 A/B repeat, each versus B090 exact logits/tokens, independent CPU gates, baseline restoration; then stage profile. Full CPU agreement, production coherence, gathered expert precision and MTP remain open.

## Owned activation-low reuse implemented, built and capture live

New OFF-default GGML_HEXAGON_HMX_ACTIVATION_LOW_REUSE_REFERENCE requires an operand-reference control. Ready marker is local to each synchronous mr activation-transfer chunk, outside nc and nx p loops. Residual job carries marker pointer only when operand reference active; reuse guarded rows<=32 and r==0. Other generic rows reconstruct normally. F32 low-A returns early when ready; Q4/Q8 low-A reconstruction skipped only after first complete prep. Weight-low reconstruction still runs per raw chunk; existing deferred DMA/owned tail capacities/HH-HL-LH exports/bias/lifetime unchanged. No new plane, overlay, GGUF or host format. Jobs synchronously queue/pop before marker scope ends. Existing18 configurebuilders resetflagOFF; dedicatedbuilder retainsB090RMScontrols and enables reuse. Archive checks actualcache+ninja flag.

Build92198terminal0/diffcheck0. Guarded signed archive backup-hmx-activation-low-reuse-controlled-candidate: host2226402B762C8E2233059CEEF1D686BE51731FBAD70B24BFA23A75F2C08A2CDD/DSPC202172943E848492E66DBC112302EB2575D507D35CBFE5CDF74F2863AF3CD7E/catalog7234B2FCE5BD398F31E1243A05750FA0A168AD2553B581BEA0E5C4E168510991; source/cache/ninja/hashmanifestarchived. Baseline DSP/catalog restored byarchive beforecapture. Initial patchcontext mismatch rejected without change; correctedpatch applied.

Capture92078/PID1645796/start13:20:47localSep27 LIVE qwen-layer0-activation-low-reuse-npu, same14names/Hello/n2/seed1234/temp0 as earlierQ8operandcapture. PreservematchedC202untilnativecompletion/wrapperrestore; no competingbuild/inference/edits. Next SAME92078 ->native0 ->originalallfourQ8projectionT9/T1gates andwrongrowcontrols ->all14selectedtensorbits versusqwen-layer0-q8-operands capture ->bounded2logitbitsversusB090fullcontrol ->baseline3hashes. Earliercapture lacksB090RMSvectorflag: if internaltensorbits differ, separateB090capture isneeded toattribute anydelta; do not blamecachefromconfoundedcomparison. Localmathonowninputs remainsindependent. Full128repeat/controlbits andstageprofilecosts onlyafterlocalgates. Expectedcostmaydrop butnomeasuredimprovementclaimyet.

## RMS vector fullpair closed: repeated bit preservation and traced gain

5778terminal0/bothnative0. Strictall128A/Blogitbitsandselectedtokensidentical0. IndependentAandBversus69F2controlstrictall128bitgates0. BothCPU --require-tokens failatunchanged93 CPU60239/NPU3812, same logits/margins. BaselineC9E4host/D2DDSP/5CB7catalog independentlyhash-restored; noinference live.

PrefillA/B337350.42/337151.04ms; traced128decode12.83/12.89tok/s versus69F2 10.95/10.98. Reproduceddirectionand~17% rate gain inorderedpairs; independentlycontrolledABBA/untracedbenchmarkstillnotrun. Profilewidth2048RMS21.997->8.568ms andbatch88.434->74.857ms supportsattribution; outputprecision preservedforthistestedsequence. B090remainsOFFdefault diagnosticoption, notCPUcorrectbestdecode orproductionpromotion.32fused/unfusedchanginginputreusecasespassedearlier; near-tie/subnormal/scaleedgecoverage/broadcorpusremainunproven.

Nexthigh-impactefficiencywork: separateownedactivation-lowreuse acrossplainHMXcolumn/nxjobs toreduce~337secondprefill; scalarreconstruction currentlyrepeats per32-outputchunk. KeepB090RMScontrolandmathcontracts, preserveDMAweightlifetime/tailcapacity andtestactualown-inputprojectiongates, wrongmaps, exacttensor/logitcontrolbits, thenprofilecosts. CPUqualification/coherence/gatheredexpertprecision/MTPremainseparateunfinishedrequirements; goalactive/fullscope.

## RMS vector arm A preserves complete128bits;12.83traceddecode; armB live

5778armA native0. Strict --require-identical versus archived69F2wholecache-rms-register-barrier-repeat128-a exits0/all128logitsandselectedtokensbitidentical. IndependentCPU --require-tokens exits1 firstselecteddifference93 CPU60239/NPU3812/samemarginsandlogits ascontrol. Thusvectoroutput efficiency preserves entiretestedsequence butdoesnotfixCPUqualification. Prefill337350.42ms/traced128decode9896.17ms/12.83tok/s versus69F2pair10.95/10.98. This agreesdirectionallywithprofilecategorygain butoneorderedarm isnotindependentABBA/untracedbenchmark orqualifiedbestdecode.

Same5778armB LIVE PID1647460/start13:10:26localSep27. PreserveB090matchedhost/DSP/cataloguntilbothnativeCLIterminal/wrapperrestore; no competingbuild/inference/sourcecandidatechanges. Next SAME5778 ->strictall128A/Bbits/tokens, independentBversus69F2bits/tokens, independentCPU_Bfirstprefix gate, independentbaseline3hashrestore. Edge/scale arithmetictests, broadcorpus and repeatedbenchmark stillpending. Goal remainsactive/fullscope includingCPUagreement/coherence/MTP.

## RMS vector-output profile closes: exactfourrows and material cost signal

Live fullpair5778/armAPID1647360/start13:03:53localSep27. PollSAME5778, no concurrentbuild/inference/candidateedits, preserveB090matchedfiles.

18150terminal/native0. Strict equal-four-row --require-identical versus69F2profile exits0: allbitsandselectedtokens identical. IndependentCPUboundedfourtoken gate0, logitsstilldifferentasbefore/no fullCPUqualification. Baseline3hashes independently restoredC9E4/D2D/5CB7.

Threecomplete1384-op one-token DSPbatches wallmean74856.667us (74786..74932), preceding69F2profile88434.333us (87743..89689):13.578ms/~15.35% lower inthisorderedpair. RMS_NORM+MUL width2048 mean8568us/81ops (8559..8580), previous21996.667us:13.429ms/~61.05% lower category. AllfusedRMS9558.667us versus22978us; unfusedRMS1275us. Outputhead3066.333us versus3064us essentiallyunchanged. Profileprefill337687.89ms remainsunacceptablyhuge; reported4.51tok/sisprofile/logging-affected, notqualifiedthroughput. Thesearethreewithin-run samples inoneorderedcomparison, notindependentABBA/productionbenchmark.

Actualoutputprecision andcostsignaljustify full128control checks, notpromotion. Fullrepeat launchedwholecache-rms-native-output-repeat128-a/b withmatchedB090. PreserveinstalleduntilbothnativeCLIterminal/wrapperrestoration. NextcandidateA/Bstrict128bitgate; eachversusarchived69F2(or8DC1)strict128bitgate; independentCPUtokenchecks stopatfirstchangedprefix; baseline3hashrestore. Then originalown-inputedge/scale arithmeticcoverage andindependentrepeatedperformance remainnecessary. CPUcorrectness/coherence/gatheredexperts/MTPstillopen; goalactive.

## Native-output archived target and compiler lowering flags verified

18150/PID125728 live. Read-only archivedbuild.ninja confirms -mcpu=v81 -mv81 -mhvx=v81 -mhmx -O2 -flto and -mllvm -enable-xqf-gen=true. Thus this is not a silentlyolderHVXtarget despitecallerdisassemblyqf32labels. Extendedqfloatgeneration is an explicitcompilercontrol, not provenreasonfortheobservedlabels or numericalbehavior. KeepcurrentB090experiment intact. If exactmodelbits fail, freezeown-inputproducts and inspect targetlowering, potentiallyseparatexqfgenerationcontrol; do not silentlychangeflags, blamehardware orweakenbitgate. Currentprojectiondiagnostics and unweighted/fusedCPUreuse aretolerancechecks, notIEEEbitproof.

## Native-output RMS control built;32reusecasespass; exactprofile pending

Live profile18150/PID125728/start12:56:41localSep27. PollSAME18150; matchedB090installed, no concurrentbuild/inference orcandidateedits.

New OFF-default GGML_HEXAGON_RMS_NATIVE_OUTPUT_REFERENCE requiresRMS_F32_REFERENCE. Scopeexisting2048-widehelperonarch>=79 andn divisiblebyVLEN_FP32; sequentialscalarF32squares/F64sum/mean/sqrt/div unchanged. Outputusesexistinghvx_vec_mul_f32_f32 twice with anexplicitHVX_Vector intermediate, unalignedvectorloads/stores, no restrictaliaspromise; scalarfallback retained. Weighted/unweighted/scale callersreusehelper. Existing17configurebuilders explicitlynewflagOFF; dedicatedbuild-rms-native-output-reference enablesitwith69F2controls inoneconfigure. Archivecache+ninjaandregressionactualflagguardsadded. Diffcheck0.

Build7255terminal0. Helperinlined, oldsymbol-specificobjdumpmissingwarningnotproof; caller rms_norm_mul_f32 showsorderedscalarF64addandvectoroutputloop. Disassembler labelsoutputmultiply qf32 plusconversion despiteSFhelperAPI; do NOT infer universalIEEEequivalence fromintrinsicname orassembly. Runtimeexactgates are required. Signedarchive backup-rms-native-output-controlled-candidate: host2226402B762C8E2233059CEEF1D686BE51731FBAD70B24BFA23A75F2C08A2CDD/DSPB090854B60427057E7AA75276944469B7495AD65B5D499BDCFEB0265020AB83D/catalogA16504D4BB66076BDD1F8BE33B45F935AD4530FF9F168590EED6D357C98ECA3C; source/cache/ninja/hashmanifestarchived. RMSreuseFusion0andFusion1bothpass8/8replayonand8/8replayoff:32cases total, changinginputs/T1T9/2048and256/inplaceandout. TheseCPUtolerancegatesnotexactproductedgeproof.

Matched rms-native-output-basic-profile4-replay0 launchedDeviceProfile1/Replay0/logits/no tensorretention. PreserveB090matchedfilesuntilnativeCLIterminal/wrapperrestore, no competingbuild/inference. Next sameprofilehandle ->baseline3hashrestore ->strictfourrowbitsversusrms-register-barrier-basic-profile4-replay0 ->independentCPUboundedtokens ->complete1384opcost/2048normwidthcomparison. If bitgate fails, diagnosefrozenown-inputproducts withoutweakeninggate; ifpass proceedfull128equivalence/repeat andedgecoverage beforepromotion. No speedorcorrectnessclaim yet.

## RMS register fullpair and fused reuse gates closed

98724 terminal 0, both native exits 0. All128 logits/selected tokens bit-identical A/B and independently B versus archived8DC1; A versus8DC1 previously also strict pass. Both independent CPU token gates still first diverge93 (CPU60239/NPU3812), same logits/margins; no fullCPUqualification. PrefillA/B337519.79/337605.43ms; traceddecode10.95/10.98tok/s versus preceding8DC1 10.85/10.86. Ordered repetitions, not ABBA/untraced independent benchmark; small speed signal only/no throughputpromotion. BaselineC9E4/D2D/5CB7 independently hash-restored, no inference live.

Then matched rms-register-barrier-fused-reuse Fusion1 oracle exits0 replayon/off, 8/8each (16 total), T1/T9/2048and256/in/outofplace/32changinginputs. Logs explicitly show fused RMS_NORM+MUL, closing prior weighted-branch coverage gap for these cases. Baseline3hashes independently restored again. Combined with earlierFusion0 gates, 32cases pass; tolerance-based CPUtests are separate from modelcontrol bitidentical evidence. Scale-specific edge/rounding tests and broadcorpus remain outside this coverage.

Registerbarriercontrol stays OFFdefault; archived69F2 is an arithmetic-equivalent diagnostic option for this tested sequence, not a CPU-correct bestdecode. Next meaningful efficiency experiment: reuse existingnative SF multiply helper to vectorize two RMSoutputproducts underaseparateOFFcontrol, preserving roundedintermediate, sequentialF64sum/scale, in-placevalidity/architectureguard. Require own-inputedge/fused/unfusedtests, exactmodelbits, and isolated/repeatedcost evidence. Separate lowoperandreuse addresseshugeprefill. Productioncoherence, fullCPUagreement, gatheredexpertprecision andMTPremainopen; goalactive.

## RMS register arm A preserves all128 control bits; arm B live

98724 arm A native 0/128 rows. Strict --require-identical versus archived wholecache-q8-operands-repeat128-a exits 0: all128 logits and selected tokens bit-identical. This extends the four-row equivalence evidence to the complete tested generated sequence. Independent CPU --require-tokens exits 1 at unchanged step93 CPU60239/NPU3812, same margins/logits as control; optimization does not solve correctness. Prefill337519.79ms/traceddecode10.95tok/s versus earlier control10.85/10.86 is one ordered timing, not repeated ABBA speedup qualification.

Same98724 arm B live PID842844/start12:46:23 local Sep27. Keep matched69F2 installed until both native terminal/wrapper restoration, no competing build/inference. Next SAME98724 -> full128A/Bstrictbitgate, independentBversusarchived8DC1strictbitgate, independentCPU_Bgate, baseline3hashrestore, then Fusion1 RMS reuse CPU oracle using prepared Fusion parameter. No promotion/fullgoal remains active.

## Fused reuse regression route prepared without touching live binaries

Added validated optional Fusion parameter to existing run-matched-rms-regression.ps1 (default 0 unchanged), forwarding it to the existing oracle. PowerShell parse passes. This allows the same changing-input/in-place RMS chains to test Fusion 1 with matched archived binaries after the current pair closes. Needed because previous 16 passing cases exercised unweighted RMS plus separate multiply only. Do not count this script edit as a executed fused runtime gate. No binaries/source/config/build changes during 98724; armA PID1699524 remains live.

## Native SF vector multiply is an existing backend path for later RMS output optimization

98724 armA/PID1699524 remains confirmed live, matched69F2 untouched. Read-only SDK/source audit: SDK19 hvx_hexagon_protos.h exposes Q6_Vsf_vmpy_VsfVsf (Vd32.sf=vmpy(Vu32.sf,Vv32.sf)), distinct from qf32 intermediate multiplication used by fast RMS. Existing hvx-base/hvx-arith/hvx-div/float-matmul wrappers already select this native SF path on supported architectures. Thus a later separate RMS control can reuse native SF vector products for output normalization and weighting, preserving an explicit rounded SF intermediate and the sequential F64 reduction/scale calculation. No need to invent a qfloat equivalence claim or change sum order.

Required controls: scalar-versus-vector exact products for normal/negative/zero/subnormal/near-rounding cases on this DSP, weighted/unweighted/scale and in-place reuse, archived full-tensor equality on frozen inputs, model full128 equality/repeat, repeated cost measurements. Padding/alignment/aliasing and target-architecture guards must follow existing helpers. Current live candidate not edited or rebuilt. This is a concrete existing-intrinsic avenue, not proof of numerical equivalence or speed.

## RMS register profile closed: exact four rows, small timing signal; fullpair next

Live fullpair session 98724, arm A PID1699524/start12:39:47 local Sep 27. Poll SAME98724, preserve matched69F2 installed, no concurrent builds/inference.

14621 terminal/native 0. All four profile logit rows and selected tokens bit-identical to prior q8-operands-basic-profile4-replay0 (strict equal-length gate 0). Independent bounded CPU four-row token gate 0; CPU logits remain different, no full CPU qualification. Baseline three hashes independently restored C9E4/D2D/5CB7.

Three complete 1384-op samples: batch mean 88434.333 us (87743..89689) versus prior 88723.667 us; width-2048 RMS_NORM+MUL 21996.667 us (21974..22027), 81 ops, versus prior 22725.333 us. Difference 728.666 us (~3.21% category), but total batch mean only 289.334 us (~0.33%) lower, with overlapping ranges/other costs varying. This is one ordered pair with within-run samples, NOT independent ABBA evidence or qualified speedup. Profile prefill 338120.16 ms and reported 4.18 tok/s are not production benchmarks. Volatile-stack removal is insufficient for a large decode gain on this evidence.

Full128 repeat launched wholecache-rms-register-barrier-repeat128-a/b, matched 69F2. Keep installed until both native terminal/wrapper restore. Next strict candidate A/B all128 bits/tokens, strict candidate versus archived wholecache-q8-operands-repeat128-a all128 bits/tokens, independent CPU gates stopping at first changed prefix, independent baseline three-file hashes. Only then judge arithmetic equivalence; existing failed CPU step93 is expected to remain if all bits preserved. Further efficiency and correctness work remains; no promotion.

## RMS reuse coverage qualification while candidate profile waits

14621/PID1708044 remains confirmed live. Existing matched reuse wrapper sets Fusion 0: three-layer RMS -> separate multiply -> add chains exercise unweighted reference normalization, T1/T9 and in-place changing-input reuse, not the fused weighted helper branch. Do not describe 16 passes as direct fused RMS_NORM+MUL proof. Current four-token profile uses Fusion 1, so its exact comparison against prior Fusion 1 profile will test model-level fused output over that bounded prefix only. Further fused/scale-specific tests and full128 controls remain necessary before promotion. No candidate changes during live profile.

## RMS register-barrier candidate built; reuse gates pass; profile live

Live profile handle 14621, CLI PID 1708044, start 12:32:34 local Sep 27. Poll SAME14621; do not restart on elapsed time. Installed DSP 69F2, keep matched files intact until native terminal.

New OFF-default GGML_HEXAGON_RMS_REGISTER_BARRIER_REFERENCE requires RMS_F32_REFERENCE. Bit-preserving memcpy to uint32, tied empty volatile register asm, memcpy back replaces only volatile square/normalized-product variables in existing 2048-wide reference; OFF path unchanged. Sequential F64 sum, mean/sqrt/divide, separate F32 products, weighted/unweighted/scale and in-place behavior retained. All 16 existing configure builders explicitly reset flag OFF; dedicated build-rms-register-barrier-reference.cmd enables it with exact 8DC1 controls in one configure. Archive and RMS regression guards verify actual compiler flag; archive also cache. Initial generated patch was rejected for duplicate file operations before changes; regenerated one operation per file successfully.

Build 63815 terminal 0. Pre-link disassembly read old binary and is not candidate evidence; post-terminal linked disassembly shows square and normalized-product stack stores/reloads removed, separate sfmpy products and ordered convert_sf2df/dfadd retained. No runtime speed/equality claim from assembly. Signed guarded archive backup-rms-register-barrier-controlled-candidate: host 2226402B762C8E2233059CEEF1D686BE51731FBAD70B24BFA23A75F2C08A2CDD, DSP 69F26C20616FE6D601B531C95118AAF73DC92C2D592BBC47D2D9719F6C0C9E2F, catalog 0FB72527A55B33A173C47146EE841129719CD56479E15D48D82AF82EFBE9F179. Source/cache/ninja and hashes.json archived. Existing RMS reuse tests pass 8/8 replay-on and 8/8 replay-off, 32 changing inputs, T1/T9, width 2048/256, in/out-of-place. These tolerance-based CPU tests are not candidate-versus-control bit proof.

Matched profile rms-register-barrier-basic-profile4-replay0 launched with DeviceProfile 1, Replay 0, logits retained, no tensor-retention capture. Preserve matched 69F2 files until CLI terminal/wrapper restoration. Next same returned profile handle -> terminal and independent baseline hashes -> strict equal-four-row --require-identical versus q8-operands-basic-profile4-replay0 -> independent CPU bounded four-row token gate -> complete 1384-op costs/width-2048 comparison. Only if equality holds consider capture/full128 controls and repeated performance measurements. No promotion; full objective remains active.

## Decode normalization cost isolated to 2048-wide diagnostic arithmetic

Extended existing --by-weight profile output with normalization input-width groups; py_compile and current complete 1384-op profile parse pass. Three steady decode samples: RMS_NORM+MUL width 2048 sums 22725.333 us across 81 ops (~280.56 us/op), width 128 sums 735.333 us/30, width 256 sums 263 us/10. Unfused RMS width 128 sums 1061.667 us/60 and width 256 237.667 us/10. Thus 2048-wide fused normalization accounts for ~95.8% of measured fused normalization cost and ~25.6% of DSP batch wall in this diagnostic control, not a production-mode attribution.

Exact archived 8DC1 rms_norm_reference_f32 disassembly shows native convert_sf2df/dfadd reduction, not a software double-add helper loop. Volatile F32 square and normalized-product barriers produce stack stores/reloads. sqrtf remains a call; reciprocal expansion is visible. Do not claim software double emulation explains this routine, and do not remove rounding barriers without independent exact-output proof. An alternative barrier/vector arithmetic implementation must preserve separate F32 square and normalized-product rounding, sequential F64 reduction or a proven exact equivalent, in-place safety and weighted/unweighted/scale branches. This evidence prioritizes 2048 arithmetic separately from tiny 128 head-normalization drift. No source/build/install changes; baseline remains restored, no inference/build live.

## Q8 basic profile closed: numerical prefix preserved, operator costs measured

96188 terminal/native 0. Four-row bounded candidate bit gate passes exactly against unprofiled 8DC1; independent CPU four-row token gate passes (logit bits differ, as before). This does not supersede failed full128 CPU qualification at step 93. Baseline host C9E4/DSP D2D/catalog 5CB7 independently verified restored; no inference live. Profile replay explicitly 0, DeviceProfile 1, tensor retention disabled. Prefill 337926.78 ms; reported four-token eval 755.90 ms/3.97 tok/s is logging/profile-affected, not a throughput benchmark.

Three complete 1384-op one-token decode batches pass strengthened parser: wall mean 88723.667 us (88338..89228). Summed operator attribution: RMS_NORM+MUL 23723.667 us/121 ops; ordinary MUL_MAT excluding head 14611 us/220; MUL_MAT+ADD 7113.333 us/70; gathered MUL_MAT_ID_NX 6957.333 us/40 and MUL_MAT_ID 6305 us/40; ordinary MUL_MAT_NX 4156.667 us/50; output head 2984.333 us/1. Combined normalization is the largest measured decode category (~26.7% of batch wall), not output head. Costs include whole-cache controls and diagnostic scalar RMS; no comparison to historical timings with different controls.

Prefill 1982-op batch wall 337313304 us: MUL_MAT 278029014 us, MUL_MAT+ADD 37882816 us, MUL_MAT_NX 20345111 us; gathered expert up/gate 816439 us and down 132280 us, RMS_NORM+MUL 48052 us. These plain matrix categories dominate the diagnostic prefill, but existing queue worker includes preparation/HMX/export, so internal stage attribution remains unmeasured. Two warmup 2054-op batches and initial 1994-op decode are separately visible; not mixed into steady 1384-op samples. Legacy parse_profile gap values cross 32-bit cycle wraps (negative artifacts); do not interpret its gap totals for these long batches. Operator usec sums and OPBATCH wall used above, not invalid cycle gaps.

Next performance work: independently time and optimize repeated low-operand preparation in plain HMX prefill, and scalar 2048 RMS arithmetic in decode, preserving exact-output/local/negative/full128 gates. Full CPU agreement, gathered expert precision, production coherence and MTP remain open; no candidate promoted.

## Decode profile parser strengthened and exact tensor attribution tested

Existing parse_stage_profile.py gains optional --by-weight attribution by exact matrix operator and first tensor. NX/fused costs remain combined and explicitly labeled, not falsely split into per-weight costs. Decode selection now also checks ordinary matrix activation token axes: second input for ordinary/ADD, last input for MUL_MAT_NX; gathered expert axes remain excluded because top-k is not token count. This catches multi-token prompt batches that nevertheless have a one-token output head.

Historical alignment-profile16 control: 15 complete 1381-op decode batches pass; 2054-op two-token batch formerly accepted on head shape alone now correctly rejects. Initial broad check mistakenly treated NX's second weight as activation; corrected using logged NX input layout and reran. py_compile passes. Historical costs are parser validation only, not current candidate measurements. Profile 96188/PID1739532 still live; same handle next, no competing build or inference.

## Explicit bounded-prefix comparator added for short profile controls

compare_logits_traces.py now accepts --prefix-rows only with a positive count and a strict bit/token gate. It requires single contexts, complete selected-token logs for both full traces, and enough rows before cropping. Output explicitly states bounded scope and NOT full-run qualification. Default strict equal-length behavior is unchanged. Tests: py_compile passes; 128-arm four-row bit gate exits 0; capture-two versus repeat-128 bounded two-row bit gate exits 0; same unequal traces without bounded option reject; CPU versus candidate bounded 94 rows still rejects at known step 93; unavailable 129-row prefix rejects. No CPU/full128 gate is weakened.

Profile 96188/PID 1739532 remains confirmed live after these tests. Next completion comparisons: wholecache-q8-operands-repeat128-a.logits.bin versus q8-operands-basic-profile4-replay0.logits.bin --require-identical --prefix-rows 4; independent wholecache-chat-cpu-reference128.logits.bin versus profile --require-tokens --prefix-rows 4. These are four-row instrumentation-change diagnostics only. Preserve matched 8DC1 installed and no competing build/inference until same handle is terminal and wrapper restores baseline.

## Matched Q8 operator-cost profile live, replay explicitly disabled

Added profile mode to existing run-q5-weight-control.ps1: explicit fresh label, 2..16 profile tokens, DeviceProfile 1, Replay 0, logits retained, no tensor-retention capture, unchanged matched installation/manifest/whole-cache guards and finally restoration. PowerShell syntax parse passes. Profiling disables replay in host code, so this is not a replay-on throughput benchmark. Existing HMX_COMP queue interval wraps the entire residual worker, including scalar operand preparation and F32 export; it is not pure HMX instruction time. Current trace capacity 256 also limits fine-grained whole-model tracing. No internal stage attribution claim yet.

Session 96188 LIVE q8-operands-basic-profile4-replay0, PID 1739532, start 12:20:39 local Sep 27, matched archived 8DC1 host/DSP/catalog installed. No concurrent build/inference. Next SAME96188 -> native terminal/wrapper restore -> independent baseline three-file hashes -> verify four selected tokens against CPU and numerical prefix versus unprofiled candidate with explicitly bounded comparison -> parse complete decode operator batches, separating output head, ordinary projections, gathered experts, and normalization. Profiled four-token data is attribution only, not independent performance/correctness qualification. Preserve installed candidate until CLI terminal.

## Q8 operand full128 pair closed: exact repeat, CPU prefix unchanged

94826 terminal 0; both native CLI exits 0. Strict --require-identical exits 0: all 128 logit rows and selected tokens are bit-identical. Independent CPU --require-tokens gates exit 1 in BOTH arms at first selected-token difference 93 (CPU 60239, NPU 3812). CPU margin 0.0516681671; candidate opposite margin 0.104560852. Step 0 NMSE 0.0051901287/max absolute 0.854408741. Stop CPU comparison at changed prefix. Thus sampled original-Q8 operand corrections reproduce locally, but do not extend the preceding stable 93-token prefix. No qualified best decode or promotion.

Prefill A/B 337502.44/337481.14 ms; traced decode 10.85/10.86 tok/s. Baseline host C9E4, DSP D2D, and catalog 5CB7 restored and independently hash-verified; no llama-cli live. Initial catalog check used nonexistent libggml-htp-v81.cat; resolved wrapper's actual libggml-htp.cat and checked that file. Full goal remains active: CPU agreement, production coherence, efficient prefill/decode, profiling, and MTP are unfinished.

Next efficiency experiment can reuse activation-low across synchronous column/nx jobs with isolated stage timing and bit-identical 8DC1 controls, reducing diagnostic iteration cost without claiming to solve CPU disagreement. Separate numerical interventions must continue addressing uncovered arithmetic/gathered-expert/output-head scopes with frozen own-input gates, rather than changing quantization thresholds or weakening token gates.

## Activation-low reuse design bounded by current runtime layout

Read-only source audit confirms both operand-reference entry points clamp m_chunk_n_rows and n_chunk_n_cols to 32, so each owned activation-low plane represents exactly one current activation row chunk. A later separate OFF-default efficiency control can split hmx_prepare_low_operands into activation and weight preparation, reset an activation-ready marker after each transfer_activation_chunk_threaded, and preserve the low-A plane across column jobs and, for nx, weight outputs. The ready marker must live outside both nc and p loops, but inside mr; it must never survive a new activation transfer or operation. Low-B remains rebuilt for each raw weight chunk, with existing DMA deferral and owned-tail capacity checks unchanged. No extra plane or Group-C overlay is needed in this scoped synchronous path. F32 activation-only jobs also benefit. Future pipeline support is a separate problem.

Required evidence before efficiency promotion: isolated preparation/HMX/export timing, bit-identical selected tensor captures versus archived 8DC1, original local gates and wrong-map controls, full-128 identical candidate-versus-control logits/tokens, and repeated timing with profiling/tracing costs disclosed. This optimizes diagnostic cost, not the unresolved CPU agreement or production coherence by itself. Current 94826 candidate/source/build remain untouched while arm B is live.

## Current-capture gated normalization cross-check closed

8DC1 layer-0 T9 gated normalization independently passes original frozen math: local NMSE 8.84576307e-13, paired CPU difference 9.7530781e-13, input-effect prediction 1.34524466e-13, prediction residual 9.15673843e-13. Values reproduce the prior upstream-identical capture. Wrong-head mapping rejects with NMSE 86.936239/native exit 1. Correct paired source is qwen-layer0-recurrent-inputs-controlled-cpu.tensors.bin. Initial use of projections-controlled-cpu was rejected for missing selected gated-normalization tensors; that failed selection is not a numerical result. No threshold changes or candidate edits. Remaining local arithmetic coverage is a diagnostic target, not a proven explanation for token 93.

## Q8 operand arm A closes: local gains do not extend CPU prefix

94826 arm A native exit 0, 128 rows. Independent CPU --require-tokens exits 1 at first selected-token difference 93 (CPU 60239, NPU 3812), unchanged from F32 activation-only and Q8 output-only controls. CPU margin 0.0516681671; candidate opposite margin 0.104560852. Step 0 NMSE 0.0051901287, max absolute error 0.854408741. Stop at the changed prefix; later rows are not comparable CPU continuations. Prefill 337502.44 ms, traced decode 10.85 tok/s. Local original-Q8 projection gains are real, but do not qualify whole-model correctness or performance.

Same session 94826 arm B is live, PID 1757348, start 12:12:24 local Sep 27. Preserve matched 8DC1 host/DSP/catalog until native completion and wrapper restoration; no competing builds or inference. Next: strict full-128 same-backend bit/token comparison, independent CPU arm B gate, then independent baseline three-file hashes. No promotion.

## Read-only gated normalization coverage audit

qwen35moe build_norm_gated is ordinary build_norm(input,ssm_norm,RMS) then ggml_silu(gate) then ggml_mul; not a distinctgatednormruntimeop. ExistingRMS_F32_REFERENCE onlyselectsne0==2048 inRMS/RMS_MUL/RMS_SCALE. GDNheadnormne0==128 therefore stillusesnormalHVX arithmetic; SiLUfastmath separatelyunchanged. This explainsuncoveredlocalarithmeticcoverage, notprovesthatitcausesremainingfullmodeldivergence orsinglecodeboundary. Future128RMS/SiLU interventionsmustbedistinctcontrolledflags withown-inputoriginalCPUarithmetic/negativecontrols, notchangingcurrent94826 matched8DC1. Holdcurrentpairintact untilbothterminal/repeatCPUgates/restoration.

## Corrected Q8 output CPUcross now dominated by one reference quantization boundary

94826 armA/PID1760568/start12:05:50 local confirmedlive, no competingbuildinference. Frozen8DC1 ssm_out T9matchingCPU pair measuredoutputNMSE2.0430645e-11/max1.50967389e-6 (earlieruncorrected~6.468e-8), predictedinputcodeeffect1.85321465e-11/residual1.78504004e-12/fraction0.0873707137, CPUownlocalmath4.55918744e-14. Exactly1/36864referenceQ8codeschanged,1/1152codeblocks,0scaleblocks. Tokenrow5/column3143 codesNPU-37/CPU-36: scaledNPUhex-0x1.240010p+5/CPU-0x1.23fffap+5, straddling-36.5; notnewtierule. InputdifferencefewF32ULPs, storedHFscalesidentical. Referencecalculationsnotactualruntimebytecapture/fullmodelcauseproof. Do notpatchquantizerthresholds toforceagree.

This localcrossgain comesfromactualprojectionoperandintervention, notchangedupstreaminput (SHAce365073...). Remaininggatednorm arithmetic/inputprecisioncantriggeroneboundary evenwhenlocalgatepasses; assessonlyaftercurrentfullpairterminal. Full128CPU/repeatstillpendingSAME94826; no promotion.

## Q8 operand full128 pair live

Session94826 LIVE wholecache-q8-operands-repeat128-a/b matched8DC1host/DSP/catalog. PreserveinstalleduntilbothnativeCLIterminal/wrapperrestoresall3; no concurrentbuild/inference. Next SAME94826 ->armAterminal/CPUfirstselecteddifference ->armBterminal/full128samebackend --require-identical andindependentCPUarmB --require-tokens ->independentbaseline3hashrestore. Usewholecache-chat-cpu-reference128.logits.bin andstopCPUcomparisonatfirstchangedprefix. Expectedscalarprefill~338secondsperarm; local8gatespassedbelow, no wholemodelimprovementassumed. PreviousCE15/0027stable93prefixstillnotqualified.

## Actual Q8 operand correction passes local gates; fullpair next

11793 terminal0/native0, wrapperbaseline restored. Prefill337671.14ms/13tokens/traced2token decode10.21tok/s (priorCE15prefill229111.41ms); scalarQ8prepadds~108.56seconds inthese captures, notisolatedkernelcycle attribution orqualifiedbenchmark. All8 originalQ8_0fullprojection gates native0: T9ssm_out1.74528379e-12/up1.31141976e-14/gate1.07053158e-14/down5.10350439e-13; T1unchanged2.37376283e-14/1.72705649e-14/1.73451528e-14/4.7011909e-15. PriorCE15T9~4.22e-8/3.24e-8/2.08e-8/5.80e-8 nowimprovedonowninputs. Ssm_out exactsameinputSHAce365073..., HFoperandidealcomparisonnow~4.225e-8, confirmingmeasuredcorrectiontowardoriginalmath ratherthanunchangedHFcontract.

Required8upstreamT9embedding/norm/QKV/z/alpha/beta/attn_output/final_output allbitidentical versusCE15, defaultstrictframecomparison/native0. Wrongweightrow controlsrejectssm_out2.01364224/down1.99739199/native1. LatersharedFFN inputschangelegitimatelydownstream; owninputgates remainindependent. No fullmodeltoken/performancepromotion. Nextsamecontrolledfull128pair matched8DC1 withCPUreference andstrictrepeat, fullgoal active; standaloneQ8correction remains diagnosticOFFdefault/pipelined/gatheredunsupported.

## Operand capture confirmed live; output-head scope checked

11793/PID1778072/start11:57:18 local remainsconfirmedlive atUTC19:03:18, no terminal/error. Do notrestartonelapsedtime; newscalarQ8lowprep addsworkbeyondprior~229secprefill. Installedmatched8DC1 untouched/no competingbuildinference. ActualGGUFinspection: output.weight Q6_K [2048,248320], token_embd.weight Q4_0 [2048,248320]. ThusnewQ8operandflag doesNOTcorrect theoutputhead; do notattributeaddedworktoQ8head orclaimitsprecisionfixed. Embedding gather isalsonotplainQ4matmulscope. Next SAME11793 ->nativeCLIterminal/wrapperrestore ->all4originalQ8projectionT9/T1gates/negativecontrols/required8upstreamregression ->full128onlyafterlocalpasses. This turnverifiedwait plusnewscopeevidence, notfailure/blocker.

## Q8 operand capture live; compiled scalar preparation costs audited

11793/PID1778072/start11:57:18 local confirmedlive, replayuid18/1985ops/no terminal/error. Preserve8DC1 matchedcandidate/no concurrentbuildinference. Read-only exactarchived8DC1 DSP hmx_residual_worker disassembly confirms __truncsfhf2/__extendhfsf2/roundf/memset/memcpy helpercalls in scalarlowoperand preparation, not a vectorized path. Source calls preparation peroutputcolumnchunk/job; chunk32 means QKV8192columns repeatsactivationlow256times, z4096repeats128times, ssm_out2048repeats64times perrowchunk. Actual time attribution NOT measured; these are staticoptimizationtargets, not cycleprofiling or proofsinglecause ofslowprefill.

Later efficiency step afterlocal/fullnumerical controls: cacheactivationlow onceperactualactivationrowchunk in its separatelyownedtail acrosscolumnjobs (andnxweightswithsameactivation), preservingvalidrows/stride/lifetime andexactQ8roundaway/HFscale rules. Lowweight remainsperrawweightchunk; deferDMAuntilreadcomplete. Vectorizing halfconversion/quantization requiresnear-tie/negative/zero/subnormal fixtures andunchangedoriginalgates. Do notreuseoverlaidact_f32 or silentlyskipquantboundarycases. No source/build/install changesmade during11793. Next SAME11793 ->terminal/native0 ->originalall4T9/T1projectiongates/negativecontrols/upstreamregression ->full128paironlyafterlocalpass.

## Q8 operand candidate built and signed; actual capture launched

Build15592 terminal0. Guarded archive backup-hmx-q8-operands-controlled-candidate: host2226402B762C8E2233059CEEF1D686BE51731FBAD70B24BFA23A75F2C08A2CDD/DSP8DC1830D48FC3F140861B826CBCDB19590970E46BA0211F9CFF58276A6F6C053/catalog5D2D65BE799E0ED4FD5E705111FB6FCA866BDDEF70320B8F329F11F5DF90AA47; actualallcontrols/compiledQ8operandflag verified/sourcecache+ninja archived/hashes.json saved. Archive restoresbaselineDSP/catalog aftersigning.

Actual capture11793 LIVE qwen-layer0-q8-operands-npu launchedsameHello/chat/n2/seed1234/temp0/all14layer0filter asCE15. Keepmatched8DC1 installeduntilnativeCLIterminal/wrapperrestore; no competingbuild/inference. Next SAME11793 ->terminalnative0 ->all4originalQ8_0T9/T1 fullprojectiongates/negativeweights/upstream8tensorregression ->full128controlledpair onlyafterlocalpasses. K512/2048/4096 ownedtailruntimecapacity checks active; scalarpreparationmayincreaseprefill. No performance or numericalqualificationclaimed beforemeasurement.

## Separate Q8_0 operand correction implemented; direct controlled build live

New OFF-default GGML_HEXAGON_HMX_Q8_OPERAND_REFERENCE requiresQ8outputresidualandQ8_0activation controls. Scope synchronousplainQ8 K512/2048/4096 only, bothsingle/nx. Activationlow usesoriginalDDR Q8_0F32amax/roundf/storedHFscale minusactualHFhigh. Weightlow decodesactualuntouchedraw1152stridetiles: signedbyte(kk/2)*64+2cc+kk%2, scale1024+2cc, explicitcode>=128 subtract256; subtractactualHFweight. ExistingQ4nibblebranch/F32activation-onlybranch preserved. HH/HL/LH pairedexports/F32sum reuseownedtailtwo32xK planes+6144exportbytes, chunk32/fullcapacityguards/deferrednextDMA. Nohostformat/GGUFchange/gatheredexpert/pipeline support.

Allresetbuilders newflagOFF; new build-hmx-q8-operand-reference.cmd establishesALLCE15controls plusQ8operandON inoneexplicitconfigure (no repeatedintermediatebuilds). Archive RequireQ8Operands checkscacheANDactualcompilerflag. Build15592 LIVE afteridleverified, installedbaselineuntouched; diffcheck0. Next SAME15592 ->terminal ->guardedmatchedsignedarchive/hashes ->actualprojectioncapture ->originalQ8_0T9/T1 localgates/negativecontrols/required upstreamregression ->full128repeat/CPUgate. Lowpartpreparation isscalar/expensive diagnostic, not throughputpromotion. PreviousCE15stable93 CPU prefixstillrejected.

## Q8 output-only full128 pair closed: exactrepeat, unchanged93prefix

63487 terminal0/bothnative0/128rows. Strictsamebackend --require-identical native0/all128 logitbits andselectedtokensidentical. IndependentCPU armsA/B --require-tokens native1/BOTHfirstselecteddifference93 CPU60239/NPU3812, CPUmargin0.0516681671/candidateoppositemargin0.157621384. Step0NMSE0.0156941799/max1.44742346. Stopafterchangedprefix. ThuslocalQ8outputcorrectionreproduces butdoesNOTextend preceding0027 stable93prefix. No promotion/qualifiedbestdecode. Prefill229290.37/229129.88ms/traceddecode10.95/10.89tok/s; hugeprefillstillunacceptable.

Wrapperandindependentbaseline3hashchecks restoredC9E4host/D2DDSP/5CB7catalog; no inference live. OfflineproposedQ8decoder nowpassesK512downsample aswell asK2048up/K4096ssm_out: bits exact/15paddedcolumns+tailzero/HFpairsweightreconstructionNMSE0, K512swapnegative2.01882286/unsigned14.4863775. Samplebytes37b7949e162875e130f65f87401407052f6e280068623c4cdef6c4be12dc5dc3; proposedtiles1221b608abac2cb0297eb0a509e846fc68fea65399df81c10649c0138d1c1fdd. Offlineonly/notruntimepackingproof.

Next separate OFF-default plainQ8operandcorrection shouldscopeK512/2048/4096, reuseownedlowplanes/pairedexports/chunk32/capacityguards, reconstructQ8activationoriginalDDR reference andsignedrawQ8weight exactlayout, defernextDMAuntil lowweights done. KeepCE15allcontrols, noGGUFchange/gatheredexpertclaim/pipelinepromotion. LocalfullprojectionoriginalQ8_0 gates andnegativecontrols before full128controls. Correctness, efficientprefill/decode/coherence/MTPgoalremainunfinished.

## Q8 output-only arm A closes: CPU prefix still93; arm B live

63487 arm A native0/128rows. MatchingCPU --require-tokens native1 firstselecteddifference93 CPU60239/NPU3812; CPUmargin0.0516681671/candidateoppositemargin0.157621384. Step0NMSE0.0156941799/max1.44742346. Stopafterfirst changedgeneratedprefix. Prior0027 firstdifference93 ->CE15 still93: measuredlocaloutputprecisiongain is NOT enough to extend testedwholemodel agreement. Some logitdistances improve andothers worsen; no monotonic/fullcorrectness claim. Prefill229290.37ms/traceddecode10.95tok/s, not qualifiedperformance.

Same63487 nowarmB LIVE PID1802284/start11:48:43 local, labelwholecache-q8-output-repeat128-b. KeepmatchedCE15 host/DSP/catalog installed untilbothterminal/wrapperrestore; no competingbuild/inference. Next SAME63487 -> terminalnative0/128rows -> strictfullrepeatbits/tokens ->independentCPUarmB gate ->independentbaseline3hashrestore. Output+operand factors remainseparate; futureQ8operandablation onlyafterpairclosed. Fullgoalactive/incomplete.

## Q8 shared-down remaining operand factorial measured; fullpair live

63487 armA/PID426088/start11:43:58 local confirmedlive, preservematchedCE15 installed/no competingbuildinference. NewactualCE15 shared-down T9inputSHAd0845592... originalQ8_0 gate5.8030682e-8/native0. Frozenfactorials isolateactivationHF2.83154404e-8 andweightHF2.96672254e-8, both5.80330092e-8. Eitheroperand correction aloneleaves~2.8-3.0e-8; bothHFpairs reconstruct originaloperanddotexactlyinidealF64math. ThreeproductsHH/HL/LH predicts1.18590271e-15 CPUcontract, pairedexports4.98521022e-13; fourproductsnoimportantgain(4.98707485e-13). CurrentmeasuredoutputmatchesHFoperandideal1.70585678e-13, confirmingcurrentremainingtestederror isoperandprecision, notfailedexport.

SingleHFexportforcorrectedproducts predictsCPUcontract4.43271263e-8 anddistancefromCURRENTuncorrectedoutput1.00674591e-7; latter isNOT originalCPUgatefailure/proofbecauseitcomparesdifferentintendedcontracts. KeeporiginalQ8_0 gate separate. FutureplainQ8operandintervention needsbothlowplanes+pairedexport, ownedtailK512/2048/4096 capacitychecks/signedrawtiledecoder. Frozenpredictionisnotruntimeproof/fullmodelcause. Wait63487 bothterminal/fullrepeatCPUgates beforechangingcandidate.

## Q8 output local gates closed; controlled full128 pair live

Capture13379 terminal0/native0; baseline restored. Prefill229111.41ms/13tokens/traced2token decode10.79tok/s, no performanceclaim. All8 originalQ8_0 fullprojection gates pass: T9 ssm_out4.22452876e-8/up3.24018631e-8/gate2.07674045e-8/down5.8030682e-8; T1 corresponding2.37376283e-14/1.72705649e-14/1.73451528e-14/4.7011909e-15. T9output-to-HFoperandideal ssm_out5.79628878e-13/up9.26884619e-15/gate8.14513919e-15/down1.70585678e-13. Thus pairedexport correctionmeasured; remainingHFoperand errorstays (ssm_out original6.467e-8 ->4.225e-8). Wrongweightmap negative ssm_out2.01345966/down1.99717131/native1.

Upstreamregression required8names T9embedding/norm/QKV/z/alpha/beta/attn_output/final_output allbitidentical versus0027. Tensorcomparator defaultstrict rawframe mode initiallyrejecteddifferentcapturefilterlayout, not numericalfailure. Addedexplicit --align-selected-tensors requiringname-regex/no rawrecord slicing; independentperfile nameoccurrencefilter BEFORE strictzip/layout/requirednames gate, defaultunchanged. py_compile0; upstreamselected8of8/native0; changedlinearout strictnegative rejects2.73003496e-8/native1; missingselectedffn_up rejectsstrictzip/native1; defaultselfpair70of70/native0. This doesnot align different generatedprefixes or bypassselectedmissing/layout/finitegates.

Fullpair63487 LIVE wholecache-q8-output-repeat128-a/b, matchedCE15host/DSP/catalog, sameHello/chatseed/temp andoriginalwholecachecontrols. Preserveinstalled untilbothCLIterminal/wrapperrestore; no competingbuild/inference. Next SAME63487 -> armAterminal/CPUfirstdifference -> armBterminal/fullsamebackendbits+toks -> independentCPUarmB ->baseline3hashes. Previous0027 hasstable93CPU prefix, newoutput-onlyeffectnotassumedmonotonic. Pipelined/gatheredexpertlimitations remain; no promotion.

## Q8 output capture verified live; original shared projection mappings checked

13379 confirmedlive/PID416364/start11:36:38 local; replayuid18/1985ops, no terminal/error. KeepmatchedCE15 host/DSP/catalog intact. Read-onlysource verifies ffn_shexp is shared-down output BEFORE scalar sharedexpert gate; capturefilter correctly selects it rather than ffn_shexp_gated.

OriginalCPU qwen-layer0-projections-controlled-cpu T9oracle mapping checks native0: inputffn_swiglu/outputffn_shexp/blk.0.ffn_down_shexp Q8_0 K512 error6.46874981e-15; inputattn_post_norm/outputffn_gate/blk.0.ffn_gate_shexp Q8_0 K2048 error1.49114049e-14. These validate original reference/callback scope for upcomingNPU checks, not candidate accuracy. Their HFoperandideal CPUdistances5.84706858e-8/2.02316356e-8 show remainingoperandprecision expected evenafteroutput correction. No relaxed1e-7 gate; next SAME13379 ->terminal/native0 ->fullactualNPU T9/T1 gates andnegativecontrols beforefull128. No competingbuild/inference.

## Q8_0 output-only build closed; actual projection capture live

Build50196 terminal0. Guarded archive backup-hmx-q8-output-controlled-candidate signed: host2226402B762C8E2233059CEEF1D686BE51731FBAD70B24BFA23A75F2C08A2CDD/DSPCE153BC580E1626918D834046119C8DDA7A093382BBF44E5F4034894965F340A/catalog9BBFC31E1AA362693DCD09EF632C6AF6051A6A9D14D61DA966AF68640E213B28. Allprior0027 controls plusactualQ8outputcompilerflag verified; hashes.json/source/cache/ninja archived. Archive temporarilyrestoresbaselineDSP/catalog aftersigning.

Capture13379 LIVE qwen-layer0-q8-output-npu sameHello/chat13tokens/seed1234/temp0/n2/filter includesinitialembedding/norm/QKV/z/alpha/beta/GDNout/gatednorm/linearout/postnorm/sharedup/gate/swiglu/down. RetainmatchedCE15 untilnativeCLIterminal/wrapperrestoresall3; no concurrentbuild/inference. Next SAME13379 -> native0 -> originalQ8_0 T9andT1 ssm_out(4096->2048),sharedup/gate(2048->512),down(512->2048) fullprojectiongates, HFoperandideal error versusprior0027, wrongrownegatives, upstreamregression. Operandroundingexpectedremaining, noCPUexact/performancepromise. Onlyafterlocalgatesfull128samebackend/CPU controls. PipelinedT>32/gatheredexperts remainoutside diagnostic support; defaultOFF unaffected.

## Plain Q8_0 output-only residual diagnostic implemented; build live

New OFF-default GGML_HEXAGON_HMX_Q8_0_RESIDUAL_REFERENCE extends existing paired-residual worker to synchronousplainQ8_0 K>=256 only. Operand activation/weight HF rounding is deliberately unchanged; noQ8operandflag/decode added. Bothsingle/nx residualselection nowtakesk, so Q8K<256 normalpath remainsunchanged where rawweight scratch maylack6144bytes. Existing rawweight ownership/capacitycheck/deferrednextDMA/F32store/bias-once reused. PipelinedselectedQ8operations explicitlyreject, gatheredexpert/defaultOFFunchanged. NoGGUF/hostformat changes. All existing resetbuilders explicitlynewflagOFF; new build-hmx-q8-output-reference.cmd establishes0027 controls then Q8outputflagON. Archive RequireQ8Residual checkscacheANDactualcompilerdefinition.

Build50196 LIVE; no inference live, installed baseline untouched. Diffcheck0. Next SAME50196 -> terminal0 -> guardedmatchedsignedarchive/hashes -> synchronousactual layer0 capture includes final_output/linear_attn_out/sharedFFN inputs/outputs -> originalQ8_0 localgateandHFoperandideal residual/negativecontrols -> full128 controls onlyafter localvalidation. Expect remainingoperanderror~4.22e-8 evenifoutputexportfixed; do notrelax original1e-7 or assumefullmodelprefiximprovement.

## F32 activation-low full128 pair closed: stable93prefix, no promotion

44754 terminal0/bothnative0/128rows. Fullsamebackend --require-identical native0: ALL128 logitrows/selectedtokens bit-identical. IndependentmatchingCPU armsA/B --require-tokens native1: BOTH firstselecteddifference93 CPU60239/NPU3812, CPUmargin0.0516681671/candidateoppositemargin0.089805603. Step0 NMSE0.00497397253/max0.957984209. Stop afterchangedprefix. Previouscombined5C39 firstdifference60 ->new0027 firstdifference93 is reproduced, but full128 gate stillfails. HistoricalQ6-only also93; noqualifiedbestdecode orcompletion.

Prefill230549.68/230488.92ms; traceddecode10.89/10.90tok/s. This diagnostic path is not performant/shippable. Wrapperandindependent3hashchecks verify restoredhostC9E4/DSPD2D/catalog5CB7; no inference live. Next distinctfactorcontrol should first extend pairedoutput-residual export to synchronousplainQ8_0 only (preserveall0027 controls), locallyverifyssm_out originalQ8_0 andsmallKsharedFFN regression/negativecontrols, then separateQ8operand precision after measuredoutputfactor. OFF-defaultflags, bothsingle/nx, explicitpipeline rejection/ownedrawscratch/deferredDMA/compiledarchiveguard, noGGUFchange or gatheredexpertclaim. Avoid mixingoutput+operandchanges inone unexplainedpromotion.

## Proposed Q8_0 tile decoder validated offline; full arm B live

New check_frozen_q8_tiles.py reads original Q8_0 GGUF packed bytes and reconstructs proposed1152-byte aligned tiles for17spreadweightrows/15paddedcolumns. Independent ggufdequantizer comparison proves decodedF32bits exact, paddedcolumns/alignedtail zero. ssm_out K4096 samplebytesSHA5886ce336cf47073f043eb2bff7f2cf6253e7aef6409c0a4e9ccb28745647e3d; proposedtilesSHA0b647942775ef7aeb007cb68dc64fcad13ba8d1fbf22a0be4df59dc56e0cc3ba. SwappedKpair negativeNMSE2.00876057 andunsignedcode13.6064796 reject original1e-7 threshold; high+lowHFweights reconstructionNMSE0 on thissample. py_compile0/oracle0. This validates proposed offline mapping only, NOT actual runtimehost/DSP packing or fullmatrix/model qualification. No installedcandidate/build changes.

44754 armB/PID1828256/start11:26:34 local remains confirmedlive. Next SAME44754 -> terminal/native0 -> strictfullrepeatbits/tokens andCPUarmB firstdifference -> independentbaseline3hashrestore. No modelpromotion fromarmA93prefix. Q8runtimeablation waits for currentpairterminal.

## F32 activation-low arm A extends CPU prefix to93; arm B live

44754 arm A native0/128rows. Matching CPU --require-tokens native1: first selecteddifference93 CPU60239/NPU3812, CPUmargin0.0516681671/candidateoppositemargin0.089805603. Step0 NMSE0.00497397253/max0.957984209. Comparison stopped at first changedprefix. This extends priorcombined5C39 firstdifference60 to93 and matches historical Q6-only longestintendedplacement prefix, but full128 gate still fails. No promotion or CPU bit-exact claim. Prefill230549.68ms/traceddecode10.89tok/s, unacceptableprefill/notqualifiedbenchmark.

Same44754 arm B now launched wholecache-f32-activation-repeat128-b. Preserve matched0027 installed until bothterminal/wrapperrestore; no concurrentbuild/inference. Next SAME44754 -> native0/128rows -> fullsamebackend --require-identical and independentCPUarmB gate. NumericallocalF32/gatednorm gain plus longerarmAprefix is evidence, not fullmodelqualification; Q8_0projection precision factor stillpending future controlledablation.

## Q8_0 raw tile/ownership audit complete; current pair unchanged

Read-only host repack_q8_0_tiled and DSP Q8 dequant agree: raw signed code for Kindex kk/outputcolumn cc is tile[(kk/2)*64+2*cc+kk%2], NOT Q4 nibble mapping. HFscale at1024+2*cc; DDRtile1088bytes, VTCMrawalignedstride1152bytes. Existing Q4 low decoder cannot be reused unchanged. Original reconstructed weight is signedcode*storedHFscale in F32; subtract actual HMXweightHF high for low plane. K4096 has128tiles and ownedtwo32xK lowplanes524288bytes plus6144bytes export scratch, reserved AFTER full L.total_bytes/alignment with release capacity check. Do not reuse GroupC act_f32 overlay, do not prefetch next rawweight until lowweight/export complete. Need independent immutable rawtile decoder/negative-layout fixture before runtime Q8 operand extension. No code/build/install changes during44754 fullpair; matched0027 remains live.

## Remaining plain Q8_0 projection factor measured while fullpair runs

44754 live/PID1845604/start local11:21:46; no competing inference/build. Offline frozen capture0027 T9 final_output ->linear_attn_out using original blk.0.ssm_out.weight Q8_0(4096->2048) passes unchanged Q8_0 gate NMSE6.46736826e-8/max0.000184077114. This path is NOT covered by plainQ4/F32 residual controls.

Matching CPU projection pair now only1/36864 referenceQ8code differs,1/1152blocks,0scaleblocks. Measuredoutput6.46822887e-8; predictedquantizedinputeffect1.85321465e-11, denseinputeffect3.49250584e-13; residual6.46804812e-8/fraction0.999972055. ActualCPU owninputmath4.55918744e-14. Earlier upstreamdrift dominated this projection; after local fixes, current tested residual is overwhelmingly projection-side, not input-code effect. No runtime activation-byte or fullmodel cause proof.

Offline ssm_out operand factorials: activationHF1.65842006e-8, weightHF2.09033006e-8, both4.22498809e-8. HH+HL+LH ideal8.45479038e-16/paired-HFexport1.7402274e-12 predicted CPUcontract; currentoutput versusHFoperandideal2.73028205e-8. Thus BOTH operand and output precision factors remain for this sampledQ8_0 projection. Future separate OFF-default plainQ8_0 residual+operand ablation needs K4096 coverage/ownedtailcapacity/rawQ8tilemapping and localregression, not blindly Q4 flag reuse. Original1e-7 already passes and must remain unchanged; no promotion or cause claim before intervention/full128. Keep current44754 candidate unchanged until botharms terminal.

## F32 activation-low full128 pair live

Session44754 LIVE wholecache-f32-activation-repeat128-a/b with matched00270743 host/DSP/catalog candidate. Keep installed until both nativeCLI terminal/wrapperrestore; no concurrent build/inference. Next SAME44754 -> arm A CPU firstselectedgate when terminal -> arm B terminal -> strictfullsamebackend --require-identical and independent CPU arm B --require-tokens. Use wholecache-chat-cpu-reference128.logits.bin; stop CPU comparisons at first changed generated prefix. Local F32 and downstream gains documented below, no wholemodel promotion assumed.

## Actual F32 activation-low capture closes with local improvement

Session1441 terminal0/native0; baseline restored by wrapper. Prefill230399.88ms/13tokens, traced2token decode10.46tok/s, not performance promotion. Same T9 norminput SHA07e1bfb4...: alpha original dense NMSE1.92626926e-14/max2.74814851e-6 (prior3.56793767e-9), beta9.91406128e-15/max8.69710874e-7 (prior4.67761247e-9), both original1e-7 gates/native0. New wrong-weight-map controls reject alpha1.44622746/beta1.56588645/native1. Frozen HH+LH paired-export measured residual1.27843929e-14/1.50642994e-14; offline partial arithmetic is not HMX bit-exact proof. This measured runtime intervention corrects sampled F32 activation rounding.

FullQKV/z originalQ8_0 owninput gates unchanged9.22032972e-15/9.21797816e-15/native0. Gatednormlocal8.84576307e-13/native0; CPUcross9.7530781e-13 (priorcombined1.09316153e-10). Frozenpredicted1.34524466e-13/recurrentonly9.6327333e-14/zonly2.68519282e-14/interaction2.78050057e-27/residual9.15673843e-13. Strong downstream local improvement, not wholemodel token proof. T33 pipeline rejection persists; broad regression failed, scope synchronous only. Next controlled full128 pair with matched00270743 candidate, then strictrepeat and bothCPU gates. No model promotion until original gates pass.

## Actual F32 capture continues; projection negative control strengthened

Session1441 confirmed live/PID1844348/start local11:15:30 Sep27; replay captured uid18/1985ops. No terminal/error yet; preserve matched0027 candidate during expected scalarQ4prefill. Previous turn was progress: identified reproducible T33 prior-control limit and launched independent synchronous model capture.

Projection checker now offers --wrong-weight-map: rotates full output weight rows while preserving captured output/shape/unchanged1e-7 gate. Rejects paired/subselected/fixture-export combinations so a negative cannot contaminate fixtures or input-effect diagnostics. py_compile0. Existing combined alpha positive dense3.56793767e-9/pass versus negative1.44619036/native1. New candidate still untested on actual alpha/beta numerical accuracy. Next SAME1441 -> terminal -> original local positives and wrong-map negatives -> HH+LH paired-export diagnostics; not a promotion until fullmodel gates pass.

## Synchronous F32 activation-low model capture live

Confirmed matmul-ops.h pipeline threshold exactly m>32; both plain residual entry points explicitly reject pipeline. This explains the known unsupported T33 regime selected by host; no production/pipeline support change made and broad activation regression stays failed. New targeted model capture session1441 LIVE qwen-layer0-f32-activation-npu, matched00270743 candidate; sameHello/chat/seed1234/temp0/n2/initial embedding+norm+QKV+z+alpha/beta+GDNout+gatednorm+linearout filter. Selected T9 synchronous prefill is distinct from failed T33 regime, not a pipeline qualification workaround. Keep all3matched installed until nativeCLIterminal/wrapperrestore; no concurrent inference/build. Next SAME1441 -> terminal/native0 -> alpha/beta originaldense1e-7/fullrows plus explicit HH+LH paired-export measured residual diagnostics -> unchanged fullQKV/z and gatednorm gates -> only then full128 repeat/CPU gates. Expected ~229sec scalarQ4prefill remains unacceptable performance.

## Prior control reproduces T33 failure; pipeline coverage remains unqualified

Prior5C39 activation suite terminal1/native-1073740791 at same first F32 K2048 N2048 T33 op after all T<=32 F32 cases pass. Thus this error is not isolated to new F32 activation-low code. Source has explicit rejection of pipelined residual-reference operations in both plain single/nx paths; host selects pipeline via htp_mm_hmx_pipeline(ne11). Investigate exact threshold before changing anything; production pipeline/default behavior must stay unchanged. Existing full regression remains failed (not relabeled pass). T9 model-capture scope is synchronous and separately testable, but broad row/pipeline qualification is missing. Baseline restored by control wrapper; no inference live.

## F32 activation regression finds T33 internal error; prior control running

Session40252 terminal1; native test-backend-ops exits-1073740791 after DSP INTERNAL-ERROR on F32 K2048 N2048 T33, first failing operation before repetitions. T9/13/24/32 F32 cases and eight small single-row controls pass. Replay0 not reached. Preserve failed logs hmx-f32-activation-precision-reuse-replay-1. Wrapper baseline host/DSP/catalog hashes independently verified restored C9E4/D2D/5CB7. No local numerical qualification or model capture permitted yet.

Launched identical existing activation_precision suite against previous combined5C39 (F32 operand OFF), label hmx-f32-activation-prior-control, to isolate regression versus preexisting diagnostic path limitation. Keep matched control installed until terminal/restore. Investigate T33 path selection/layout from actual evidence, do not assume new operand code caused failure. Offline sweep now explicitly reports two products HH+LH and measured-output NMSE; old capture alpha predicted paired-export CPU error1.053e-14 but measured-oldoutput3.568e-9, consistent with correction not yet present. Script py_compile/localoriginalgate exits0; offline only.

## F32 activation-low candidate built; matched bias regression passes

Build1248 terminal0. Signed archive backup-hmx-f32-activation-controlled-candidate: matched host2226402B762C8E2233059CEEF1D686BE51731FBAD70B24BFA23A75F2C08A2CDD, DSP00270743AB2CA9BA2ED023C5995FB39A3A56C6C2BF7EF9B2184B223C90801F85, catalog2E9F79E973DDD332805296ECC5E5A595486121BFCA6A276260944095C62FD19D. RequireF32Operands archive guard verifies cache AND actual compiler flag, along with all combined controls. Source/config/ninja archived, hashes.json saved. Archive restores baseline DSP/catalog.

Reused matched regression wrapper now supports existing mm_bias and activation_precision cases plus compiledF32operand guard. Bias runner terminal0: replay1 12/12 and replay0 12/12, changing inputs32 repetitions, inplace/outofplace, K32 untouched and K2048, N32/64/256. These are single-row tests and do not validate new HMX lowoperand path by themselves. Activation precision runner newly launched hmx-f32-activation-precision-reuse; retain candidate until terminal/wrapperrestore. Includes F32 and Q4 K2048 T9/13/24/32/33/64 padded/multi-chunk identity cases, but exact dyadic activations do not exercise real rounding residual. Actual frozen alpha/beta capture/local dense gates remains required before full128 promotion test. No numerical/performance promotion.

## F32 activation-low diagnostic implemented; compilation in progress

New OFF-default GGML_HEXAGON_HMX_F32_OPERAND_REFERENCE requires existing F32 residual export. Scope synchronous plain F32-weight K2048 in single/nx paths, not gathered experts or T1 HVX. Own-tail allocation/chunk32/capacity checks reuse Q4 ownership design; F32 low preparation reads original DDR activation minus actual tiled HF high without Q8 reconstruction or raw-Q4 decode. Products HH and LH each use paired-HF export/F32sum, bias only once. Q4 retains its prior three products. Both job entry points carry explicit weight_type. Weight HF rounding deliberately unchanged: offline alpha/beta weight-only error ~1e-14/~1e-16. Existing diagnostic/reset builders explicitly reset newflagOFF; new build-hmx-f32-activation-reference.cmd establishes combined5C39 controls then newflagON.

Build session1248 live; installed baseline binaries untouched. Diffcheck exits0. Re-reading archived18-fixture HMX retained/threeproduct logs passes identical/negative controls, but is historical mechanism evidence ONLY and does not validate new runtime preparation. Next finish SAME1248 -> inspect actual compilerflag -> extend archive compiled guard/manifest -> matched signed candidate -> existing backend F32 cases and actual frozen alpha/beta capture/local gates -> full128 pair if local tests pass. No runtime qualification claimed.

## Combined full128 pair closed: repeatable, CPU token gate rejected

Session99685 terminal0; both native0/128 rows. Full --require-identical exits0: every logit bit and selected token matches across arms. Independent matching CPU --require-tokens exits1 for BOTH arms: first selected difference60 CPU1061/NPU353, CPU margin0.141433716/candidate opposite margin0.0455322266; no comparison beyond changed prefix. Step0 NMSE0.00629679234/max0.697873831. Prefill228653.26/228697.57ms; traced decode10.98/10.91tok/s. Combined local fixes do not extend the CPU prefix and are not a performance/correctness promotion.

Wrapper restored all three baseline files and independent hash check passes: hostC9E4D8AD..., DSPD2D98EDE..., catalog5CB7F7EF8F6D6E88C5247D4EB5AC43CA527D7DF15BA66C547CBC25A3507CE09F. No inference process remains. Next targeted numerical intervention: separate OFF-default F32 activation-low correction, with existing paired-HF output export, frozen alpha/beta local checks, unchanged original gates, then same full128 controls. Gathered expert prefill remains uncovered by plain residual controls; do not claim alpha/beta is the wholemodel cause before intervention.

## Combined full128 arm A closed; arm B remains live

Session99685 continues with arm B, PID1888332/start local11:01:59 Sep27. Arm A native0/128 rows, prefill228653.26ms, traced decode10.98tok/s. Matching CPU --require-tokens exits1: first selected-token difference60, CPU1061/NPU353; CPU margin0.141433716, candidate opposite margin0.0455322266. Step0 logit NMSE0.00629679234/max0.697873831. Comparison stops at the first changed prefix. The large sampled layer0 improvement did NOT extend arm A's CPU token prefix; no promotion. Keep matched5C39 host/DSP/catalog installed until arm B terminal and wrapper restoration, then run full samebackend bit/token equality plus independent CPU arm B gate.

Combined-capture offline alpha/beta operand factorials both pass original dense1e-7 local gates but isolate remaining activation-HF rounding: activation-only errors3.56715716e-9/4.67456e-9 versus weight-only8.67646912e-15/1.49538551e-16. Activation high+low with HF weights predicts8.90866073e-15/3.34771195e-16 using ideal F64 partial arithmetic. Three-product paired-HF export predicts1.05315696e-14/2.37788247e-15; single-HF export loses that gain (~5.52e-8/4.34e-8). These are offline predictions, not runtime/HMX validation or wholemodel causality. A later separate F32 activation-low operand ablation is justified; preserve existing output-residual export and distinct owned scratch, do not alter the running candidate.

Read-only implementation review: hmx_prepare_low_operands currently reconstructs Q8_0 activation codes plus raw Q4 weights; it must NOT be enabled unchanged for F32 weights. A scoped F32 diagnostic should instead form activation low from original F32 DDR minus actual tiled HF high, retain paired exports for HH and LH, and preserve padded-row zeros. Existing F32 output-residual alone does not correct operand rounding. Validate immutable F32-origin fixtures and wrong-layout negatives before actual alpha/beta capture; keep gathered experts and unrelated widths outside this ablation. No source changes made during live pair.

## Combined local factors pass; RMS reuse regression passes; fullpair live

Capture30037 terminal0/native0; baselineC9E4/D2D restoredverified. Actual combinedprefill228609.92ms/13tokens/traced2tokendecode10.47tok/s, no performancepromotion. T9RMS normoutputEXACTofflineCPUreference (0bits/error), sameoriginalembeddinginputSHA cc3f16d5..., compiledCPUcross1.96131189e-15 andreferenceQ8codes0changes.

FullT9QKV/z owninputs originalQ8_0 gates9.22032972e-15/9.21797816e-15/native0. PairedcompiledCPUcross2.21561986e-14/2.11042228e-14, inputquantizationpredicted effect0 because allreferencecodes/scalesmatch. Thus combinednorm+operand factors nearlyeliminate sampledprojectioncrosserror, notjustown-inputerror. Gatednormlocal9.25412387e-13/pass; CPUcross1.09316153e-10 (priorQ4operand-only3.13124837e-7), predicted1.07310971e-10; recurrentonly1.07600949e-10/zonly2.68519282e-14/residual9.59314568e-13. Remaininggatednormdrift nowrecurrentinputside, notzdominance.

ActualF32alpha/beta owninputdense gates3.56793767e-9/4.67761247e-9/pass; idealHFoperanddot8.64861754e-15/7.4869131e-15. These remainingHFoperand errors aremeasured, notproofallremainingrecurrent/modeldrift causality. CouldjustifyseparateF32operandprecision ablation later; do notchangecorrectgatednormkernel.

Preparedandran run-matched-rms-regression.ps1 (matchedmanifest/compiledRMSguard/baselinevalidation/nooverwrites/finallyrestore) usingexistingRMSreusebackendcases. Signed13CA RMS-onlycandidate: replay1native0/8of8 and replay0native0/8of8; ne2048andne256 unchangedwidths,1and9tokens,inplace/outofplace,32changing-input repetitions/3layerRMS+mul+addchain. Baselineall3restoredverified. Newsingle-tokenRMS frozenoraclealso0bits/error andreferenceQ8codes0 changes vscompiledCPU, CPUcross1.82108492e-15 onbitidenticalinputs.

Controlledcombined128pair session99685 LIVE wholecache-rms-q4-operands-repeat128-a/b; keepmatched5C39candidateinstalleduntilbothnativeCLIterminal/wrapperrestore. No concurrentbuildinference. Next SAME99685 -> terminal/native0/128rows -> strictfullsamebackendbits/tokens -> BOTHmatchingCPUfirstselecteddifference. Local3e-7 ->1e-10 normdriftimprovement isnotmodelpromotion,230secprefillstillunacceptable. Alloriginalgates/MTPgoal retained.

## Combined RMS + Q4 operand capture running

New build-rms-q4-operand-reference.cmd establishesexactRMS-onlycontrols then explicitlyenablesbothRMS_F32_REFERENCE andHMX_Q4_OPERAND_REFERENCE. Build10364 terminal0; no productionkernel/sourcechange beyond existingtwoOFFdefaultflags. Signed backup-rms-q4-operands-controlled-candidate matchedhost2226402B..., DSP5C39EE42CFF67FF3B1BB9F9EC020D846C1FB7DCFDD403685F4AF240DCF12E7F2/catalog60B3D6E296EB3B3809FEF19520F0891B98E6554DED15EF1D78E5B65BEA5062DA, actualcompilerbothflags/controlsguarded, exactmatmul/unary/config/sources/hashes.json archived. Archivetemporarilyrestored baselineafter signing.

Actualcapture30037 LIVE qwen-layer0-rms-q4-operands-npu n2/filter initialembedding+norm+QKV+z+alpha/beta+GDNoutput+gatednorm+linearout. Keepmatchedcandidateinstalled untilnativeCLIterminal/wrapperrestore; no concurrentbuild/newinference. Expectexpensivescalaroperandprefill (prior~229sec), nottimeout/deadlockproof. Next poll SAME30037 -> native0 -> RMS pairedidenticalinput/codeboundaryoracle, originalfullQKV/z Q8_0 projectiongates, F32alpha/beta densegates, gatednormcounterfactuals -> thenfull128controls. Combinedeffectnotassumedmonotonic; bothseparatefactors stablebutCPUfirst60.

Read-onlyexpertcoverageaudit: hmx_mm_id_2d_f32 stillusescore_dot_chunk_fp16 intoHFvtcm_output then transfer_output_chunk_scattered_threaded; nextweightDMAstartsafterdequant BEFOREcompute. PlaincorrectionflagdoesNOTcoverthis gatheredpath. Futureexpert residual diagnostic needsdistinctownedexportplanes/deferral/scattermapvalidation/frozenexpertIDs andinputs; do notreuseoverwrittenrawweights or inferbadexpertmath fromdifferentrouter/inputhistories. This is uncoveredcoverage, notisolatedmodelcause.

## RMS-only full128 repeat closed; local fix is not a promotion

61591 terminal0/bothnative0/128rows. Strictsamebackend --require-identical native0/alllogitbitsandselectedtokensidentical. BothmatchingCPU --require-tokens native1/firstselecteddifference60 CPU1061/NPU353; CPUmargin0.141433716/candidateoppositemargin0.0489826202. Step0logitNMSE0.0058026997/max0.943256617. Stopafterfirstchangedprefix. Prefill2070.03/2067.60ms; traceddecode11.01/11.03tok/s versuspreviousEBBA~14.20 diagnostic. ScalarRMS slowsdecode anddoesnotextendCPUgreedyprefix despitefixinginitialboundarycodes. No promotion/gaterelaxation; baselineC9E4/D2D restoredverified, nojobs live.

Separatefactor evidence complete: RMS-only correctsinitialreferenceQ8codesbutleavesprojectionHFoperanderror; Q4operand-only correctslocalprojectionbutretainsupstreamboundarycodes. NeitherextendsfullCPUtokenprefix alone. Next controlledcombinedRMS+Q4operand diagnostic is justified to testinteraction, notassumesuccess; retainslowparttailownership/oldoriginalCPUgates andwillbeexpensiveuntilvectorized. Otheruncoveredpaths (gatheredexpertprefill HFoutputs/F32alpha-beta operandprecision) remainpotentialfactors, notestablishedwholemodelcauses. Keephistorical93prefixincomplete andperformance/MTPgoalsopen.

## Actual RMS arithmetic reference removes the two initial quantization boundaries

Implemented OFFdefault GGML_HEXAGON_RMS_F32_REFERENCE in existing unary worker: 2048-wide F32 RMS/RMS_MUL normalization only (RMS_SCALE uses same reference normalization thenexisting scale/bias). F32 square forcedvia volatile, F64sum, F32mean/sqrtf/divide, separateF32normalize/weightmul. Otherwidths/F16/gatednorm/defaultOFF unchanged. No new buffers/layout/hostformat. Build61857 terminal0/diffcheck0. Otherbuilders/reset explicitlyresetRMSflagOFF. New build-rms-f32-reference.cmd usespriorEBBA controls andQ4operandOFF, RMSreferenceON. Archivehelper RequireRmsF32 checksactualcompilerflag andnowstoresunary-ops.c/hvx-norm.h.

Signed backup-rms-f32-controlled-candidate matchedhost2226402B..., DSP13CA01B80C8875BC8E450A0B190F85D067E0A8B4A34ACAB1F7B56BD96BCA6BB8/catalog505F1D899C37EDB6DAFAF3E989FECE108CC7D27CE2878AEA30BAF8355D6F32EE, exactsources/config/compilerflags/hashes.json. Capture84063 terminal0/native0 qwen-initial-rms-f32-npu1181678bytes, baselineC9E4/D2D restoredverified. Tracedprefill2071.12ms/13tokens/2tokendecode10.87tok/s arediagnostics, NOT performancepromotion.

SamebitidenticalembeddinginputSHA cc3f16d5... CPU/newNPU. FullT9normoutput EXACTofflineCPUarithmeticreconstruction NMSE0/max0/0of18432bitsdiffer; actualcompiledCPUcross1.96131189e-15 remainsbecauseoracleisnotcompiledCPUbitexact. ReferenceQ8_0codes againstCPU now0/18432changes and0/576scaleblocks, previously2codes. Wrongweightnegative newcaptureNMSE0.0165394689/native1; primary1e-7unchanged.

ActualnewT9gate localQ8_0NMSE1.14829988e-8/native0 (operandcorrectionOFF); pairedCPUinputcodechanges0, predictedquantinputeffect0, remaininggatecross1.1483071e-8 nowprojectionoperandrounding. OriginalHFoperandideal matchesnewoutput7.54470003e-15. This demonstrates two distinctprecisionfactors ratherthanclaimnormalonefixesmodel. NewnorminputSHA07e1bfb4... versusCPUc3383c... differbitsbutreferencecodesmatch.

Controlledfull128RMS-onlypair session61591 LIVE wholecache-rms-f32-repeat128-a/b; preserveinstalledmatchedpair/no concurrentbuildinference. Next samehandleterminal -> strictbothfullbits/tokens -> bothCPUfirstselecteddifference. PriorQ4operandprefix60/EBBA65/historical93 stillnotqualified; do notclaimmodel/performance/MTPcompletionfromlocalnormpass.

## Paired initial RMS inputs are bit-identical; arithmetic-path drift isolated

CPU28880 and NPU37152 bothterminal0/native0; NPU archivedEBBA F32residualcontrol (operandOFF), baselineC9E4/D2D restoredverified. Actual model.input_embed name retainedcorrectly. T9inputF32SHA BOTHcc3f16d526e86729e7d34ece0ccf9a0e52ba1e16cc8c820055f99e00da648c52, inputsbitidentical. Pairedactualattn_norm drift1.6755037e-14/max1.90734863e-6 withpredictedinputeffect0. Thus initialnormalizationpath arithmetic, not differentembeddinginputs, accountsforthisspecificcapture's outputdifference. Does notattributeallwholemodeldriftto RMS.

NewCPUarithmeticNumPyoracle original1e-7 pass: CPU1.96131189e-15/3528of18432F32bitsdiffer; NPU1.58895212e-14/10361bitsdiffer. Explicitly notbitexactcompiledCPU sqrt/scale/vector math. Crucially referenceQ8_0 quantization oforacleprediction differs0codes/0scaleblocks fromCPUoutput, but2codes/0scaleblocks fromNPUoutput. WrongweightrotationrejectNMSE0.0165394663/native1. CPUselfpairnative0/inputidentical/alldriftmetrics0; missinginputnegativepreviouslyrejected. This supports OFFdefault F32 RMS arithmetic-reference diagnostic (F32squares/F64sum/F32mean/sqrtf/scale/mul), originallycapturedinputs andunchangedlocal/CPUtoken gates. No suchkernelchangeimplementedyet.

FullQ4correctionrepeat auditclosed: stable128bits/tokens butCPUfirst60both,229secprefillrejected. No livebuild/inference now. Next: RMSdiagnosticintegration usingexistingunary/HVX-worker machinery; pairedinput/originalCPU/localquantboundary andfull128controls beforeanyperformancepromotion. Keepnormal fastnormdefaultunchanged anddoNOTchangeroundingrules.

## Q4 correction repeat audit closed; paired RMS input capture begins

59544 terminal0/bothnative0; baselineC9E4/D2D restoredverified. Strictsamebackend --require-identical native0: all128fulllogitrowsbits/selectedtokensidentical. CPUcomparisons bothnative1/firstselecteddifference60 CPU1061/NPU353, unchangedmargins. Diagnosticprefill228560.59/228541.13ms, traceddecode14.01/14.05tok/s. Not a promotion: previous65/historical93 remainlonger/incomplete; unacceptableprefill. Artifactpairretainedasstabledeadend/precisionmechanism evidence.

New check_frozen_rms_math.py strict original model.input_embed -> attn_norm-0 captureoracle, original F32weights/epsilon, F32square/F64sum/F32mean/sqrt/divide/normalize/weight CPUcontract, unchanged1e-7localgate plusbitdifference report. Optionalpairedcapture requiresidenticalshapes/reportsinputbitidentity andinputeffect/residual; optionalwrongweightrotationnegative separate. Compile0; oldrecurrenttrace correctlyrejects missing original embedding input/native1, so prioroutputdrift cannotbe falselyattributedto RMSkernel. ActualCPUpositive/negativegate not yetexecuted.

CPUcapture session28880 LIVE qwen-initial-rms-inputs-cpu n2/matchingchat/Device none/filter model.input_embed+attn_norm-0+z-0+linear_attn_qkv_mixed-0. Pollsameterminal/native0 -> confirmtensorname/shapes/strictRMSoracle andnegative -> NPU matching capture using archived EBBA F32residual control (no slowoperandpath neededforinitialRMSinputproof), notanotherconcurrentinference. Originalactualinitialattn_norminputSHA wasidenticalold/newQ4candidate, butrevalidateinnewcapture. No RMSkernelreplacement untilpairedoriginalinputevidence.

## Full128 arm A rejects promotion despite local projection improvement

59544 armA native0/128rows completed, prompt228560.59ms/13tokens, traceddecode14.01tok/s (notuntracedbenchmark). MatchingCPU --require-tokens comparisonnative1: firstselecteddifference60, CPU1061/NPU353; CPUmargin0.141433716/NPUoppositemargin0.0657234192. Step0CPUlogitNMSE0.00589067697/max0.920558453. Stopafterchangedgeneratedprefix; laterrowsnotaligned. PreviousF32residualpair65 andhistoricalQ6-only93 remainlonger; neitherqualified. Preciselocalkernelcorrection doesNOTimplymonotonicfullmodel/tokenimprovement. Currentcandidate rejected forperformance/promotion, butstillneedarmB toclosecontrolledrepeat audit.

ArmB nowLIVE PID373104/samewrapper59544; keepcandidateinstalled untilterminal andautomaticrestore. No claim ofcrossrunbitidentityyet. Next poll59544 -> strictboth128bits/tokens -> armBCPUfirstdifference -> ledger. Do not abort/restart solelybecause~229secprefill isexpected.

## Exact reference quantization boundary localized while fullpair runs

New optional --quant-boundary-details on existing paired matmul diagnostic reports up to16changedcode locations with exactF32hex inputs/scaledvalues/originalscales; requirespairedtrace/originalQ8_0 gate, no reference/gate mutation. ActualT9z pairchangesbothatcolumn1842, tokenrows0and6: candidatecode33/CPU32; candidatescaled0x1.0400000000000p+5 =32.5 EXACT, CPUscaled0x1.03fffe0000000p+5 (oneF32stepbelow32.5). Candidateinput0x1.58a01a...p-1/CPU0x1.58a014...p-1; originalscales differby2F32steps although storedHFscaleidentical. Roundawayfromzero onexacthalf explainschangedreferencecode. Selfpairnative0:0changedcodes/scale/codeblocks andallfourdriftmetrics0; Pythoncompile0. Do NOT patchroundf/tiebreaking toforceagreement; quantizers followdeclaredcontract ontheirinputs.

Read-only sourceaudit CPU RMS (ops.cpp3924+, ggml_float double fromvec.h) sums F32 x*x inF64, castsmeanF32, sqrtf/divF32, thenF32normalize/weightmultiply. HTP hvx-norm.h uses qf32vector products/sums, SFvectorreduction andapproxvectorrsqrt beforeSFproducts. These are differentarithmeticcontracts, consistentwith observed1.6755e-14 inputdrift, butactualprenorminput notcapturedhere so not anisolatedRMSkernel attribution. Next usefulprecisiontest requires paired identicalprenorm inputs/weights/epsilon andoriginalCPUoperationoracle; no globalnormreplacement yet.

Full128session59544 stillLIVE/PID1949604 verifiedafterchecks; armalog showscached9prompt/replayuid18/1982ops, no resultyet. Preserveinstalledpair/no concurrentbuildornewinference. Pollsamehandleuntilbothterminal/native0 thenstrictfullrepeatandCPUtokens. Precisionmechanism passdoesnotreplacefullmodelgate orjustify229sec prefillcost.

## Runtime Q4 correction capture passed; controlled full128 pair live

Additional paired-input T9z oracle: only2/18432 referenceQ8_0codes differ in2/576blocks,0/576scaleblocks. Measuredprojectiondrift7.25619691e-9/predicted7.25664291e-9/residual2.11472806e-14; denseinputeffect1.03728946e-14. These referencequantizerchanges explain residualcrossbackendz drift afterlocalkernelcorrection, not runtimebyte snapshots. Initialnormalization differs1.6755037e-14; future upstream exactness investigation is justified beforealteringcorrectedprojection. Strict6requiredT9tensor scope/layout/finite compare reportingexit0 (all6differ, not equalitypass). No concurrentkernelbuildwhile59544live.

Session1510 terminal0/native0; completed2590853byte capture. Wrapperrestored baselineC9E4/D2D verified. Actual prompt eval228709.12ms/13tokens, traceddecode13.30tok/s; this scalar/smallchunk diagnostic is unequivocally NOT shippable prefillperformance. Earlierliveobservation wasnotterminal/stallproof; do notclaimdeadlock. Generatedcodecost hypothesis stillrequires profiling before optimization attribution.

Full frozen T9 projections on SAME canonicalinputSHA887f30af... against unchanged originalCPUQ8_0 gates: QKV9.26978496e-15/max7.94313382e-6 (prior7.46452689e-9), z9.2480069e-15/max4.47401544e-6 (prior1.14785097e-8), bothnative0. Newruntime outputs differfrom HFoperandideal bythe expected operandcorrection. This tests all8192/4096 outputs forlayer0/T9, notwholemodeltokens/runtimequantbytes snapshots.

Paired gatednorm T9local9.28344779e-13/pass, measuredCPUcross3.13124837e-7 (prior5.53117918e-7), predicted3.13094897e-7, recurrentonly8.0595082e-9, zonly3.11388528e-7, residual9.5710084e-13. T1local2.60320304e-12/pass, measured2.33730049e-10, recurrentonly2.31307699e-10/zonly6.78152248e-16. Thus actualprojectionlocalcorrection improves selectedcrossbackendmetrics butdoesnoteliminate differinginputhistory/referencequantizationboundaries; no normkernelaccusation/gaterelaxation.

Launched SAME matched1152136C candidate controlledfull128 wholecache-hmx-q4-operands-repeat128-a/b session59544. Nativecli live; keepcandidateinstalled untilterminal andwrapperbaseline restore. No concurrentbuild/inference allowed. Next pollsame59544 -> bothnative0/128rows -> strictsamebackendfullbits+tokens -> bothmatchingwholecache-chat-cpu-reference128 selectedtoken firstdivergence. Do notreplace best historical93prefix with localprojectionproof or diagnosticthroughput. After modelagreement evidence, prioritize hoisting lowactivationprep/vectorizing in separately ownedHVX-worker context, preserving tailownership/fullcapacityguard.

## Runtime correction capture remains live; generated-code audit identifies scalar prep cost

Revalidated same session1510/PID381368 at17:25:50UTC (~3min45afterstart): processlive/CPU27.71875, tensors562434bytes unchanged since10:22:35local, log remainsgraphreplay1985ops/taskactive. Not terminal, not restarted, installedmatchedcandidatepreserved. Windows waitchain allthreads reports no detectedcycle; does NOT proveDSPprogress orruleoutDSPwait. Capturedpartialframes containonlytwoT2warmupgroups, NOT T9 orT1. Strict firstT2 gateQ8_0 oracle localNMSE1.53873963e-14/native0; doesnotqualifynewT9HMXcorrection.

Read-only objdump exactarchived1152136C DSP hmx_residual_worker shows scalar librarycalls __truncsfhf2/__extendhfsf2/roundf plusmemset/memcpy; no inlinedHVX ops seen in inspectedfunction. Supports substantial scalar-low-prep cost as an investigation target, not proof ofstallcause/throughputnumber. This firstimplementation recomputes32rowlowactivations for every32colchunk and converts paddedrows too; potential next optimization must hoist activationlowprep perrowchunk and use existing HVX-worker context forpacking, whilepreserving separately ownedtail and originalCPUreference. Do not claimthis diagnostic is a latencywin, or launch anothermodelprocess whilethisoneis live. Next poll same1510/terminal -> originalfullQKV/zT9gates ->comparepairedCPU -> decidewhethercost warrants prepoptimization beforefull128.

## Runtime Q4 operand correction implemented; actual model capture live

New OFFdefault GGML_HEXAGON_HMX_Q4_OPERAND_REFERENCE requires existing Q4 residual/Q8_0 activation controls. Synchronous plain single/nx HMX Q4 K2048 only; diagnostic row/column chunks clamped32. Two32xK HF lowoperandplanes plus three exporttiles reserved at aligned tail AFTER L.total_bytes, fullend checkedagainstctx VTCMcapacity. No GroupC scratch reuse/hostformat change. Lowactivation uses originalDDR F32 rows with existing declaredQ8_0 roundf/storedHFscale contract; lowweight uses untouched rawtiledQ4 bytes/alignedtile stride minusactualdequantHFhigh. HH/HL/LH retained exports F32summed; biasonce afterlastproduct. Rawweights preserved untilcorrectionfinished, nextDMAalreadydeferred; selectedpipelines stillreject. Gatheredexperts/T1HVX/defaultOFF unchanged. This scalar/smallchunk path is a diagnostic, not optimized throughput.

Build12922 terminal0: base F32 residualOFFoperand compile then newoperandON compile, bothsuccess. New build-hmx-q4-operand-reference.cmd; baseF32builder/other diagnostic builders/reset explicitly reset newflagOFF toavoidcachecontamination. Archivehelper RequireQ4Operands checksactualCMakecache plusbuild.ninja. Signed backup-hmx-q4-operands-controlled-candidate matchedhost2226402B..., DSP1152136C8D4C6328C52D9742E334EF08CED4A19B1B946D97AD4267E5D53E852D/catalog50177FC6D5AB29B4FB1DDC51E583CDFF68FFF8667471CDEE023600227401A3EC; exactsource/config/compilerdefs/hashes.json stored. Prearchivebaseline restored; Q5T4 archivedprojectionregression unchanged3.57194014e-14/native0 (correct occurrence3; erroneousocc1T4 request correctly rejectedactualT2shape).

Capture session1510/PID381368 currently LIVE qwen-layer0-recurrent-q4-operands-npu n2/same14layer0recurrentfilter/compiledwholecachecontrol. Observed log model loaded/taskstart/replaycaptured, partialtensor562434bytes; no terminal/error yet. KEEP matchedcandidateinstalled untilactualCLIterminal, wrapperthenrestoresbaseline. Next poll SAME handle1510, requireterminal/native0 before numerics. Then originalCPUQ8_0 frozenQKV/z fullprojection gates, pairedCPUrecurrentT9/T1, and controlled128strictrepeat/CPUtoken gates if capturepasses. No local/fullmodel/performance/MTPqualification claimed whilelive.

## Runtime ownership audit and tiled-weight mechanism validation

Critical layout observation: htp_mm_hmx_vtcm_layout_build defines act_f32 as Group C overlapping Group B weight/DST/dequant buffers. This is safe for existing sequential activation prep but NOT persistent low operands while weights/compute are live. Do not reuse act_f32 merely because activation prep finished. Existing raw-weight residual scratch also overwrites rawbytes after dequant; it cannot simultaneously retain raw Q4 data for later lowpart reconstruction. Next runtime design must reserve non-overlapping space or shrink chunks/recompute ownership; no layout/host format silently changed.

Extended fixture exporter packs selected original GGUF Q4 blocks into host repack_q4_0_tiled layout (adjacent K nibbles interleaved across32outputcolumns, scales at512,576logicalbytes/tile), padding15columns zero. Generated hmx-model-tiled-probe.generated.h preserved independently from earlier directF32header. Originalblock decoding checked exact vs GGUF library, independent new check_hmx_model_probe_header.py roundtrips F32hashes/tiledecoder/zero-padding and swapped-nibble negative NMSE2.01356208 (reject); native0. PackedSHA1f8d4372700daaa3f0918d6798e879d7be915b89f6527f5be00a47bf2e1c7c0d/headerSHA1f9d45ff5364806b4b28bdae05256402a4a326d9a31d77ccdd0485a00543876f. Pythoncompile0.

New probe fixture17 reconstructs high/low from packed Q4 bytes using scalar HFscale/F32integer product; every reconstructed value independently checks against original F32fixture16, and numerical reference uses fixture16 rather than the same decoder (avoids self-consistent wrong-packing pass). Build70644 terminal0. Archive backup-hmx-tiled-gate-probe-candidate DSP88990AC8E8828F3E71FED298592B88B6A00A0DA7F710E351AC9F0DCD60472D5A/catalog930C5AF0E21BE552A8BBBE3DEEFDF408F3C9838DA0551D80352C9FFB050463EE, exactsource/generatedheader/config/ninja stored. Runtime61397 terminal0: both HMX-routed arms native0/normal1of1 tests pass. Strict18fixture/576metadataevent decoder exit0/crossrunbitidentical. Packed fixture matches directF32fixture metrics exactly: raw4.92180021e-8/corrected5.67837589e-15/wrongidentity1.81683218. Allpriorchecks unchanged. BaselineDSPD2D restoredverified; nojobs live.

Packing mechanism is now hardware-tested for sampled reference data, not live DMAbytes/fullmodel integration. Next reserve separately owned low tiles (potentially one32row/32col tile at a time), clamp diagnostic chunk sizes if needed and account VTCMtailcapacity; keep rawweights intact until lowpart prep completes and add bias only once. Current defaultOFF production path unchanged; no speed/token/MTPqualification.

## Frozen actual gate operands pass three-product HMX correction twice

Extended existing matmul oracle with --export-hmx-probe: only full original Q8_0 layer0 gate T9, requires unchanged1e-7 projection pass and expected4096x2048 dimensions, exclusive-create ASCII C hex literals. All9 captured inputs are reference-Q8_0 quantized;17 evenly spread original GGUFweight rows [0,255,511,...,3839,4095]. These are declared reference operands, NOT actual runtime quantizer bytes. Generated hmx-model-probe.generated.h includes F32 operand hashes and is compiled ONLY inside OFFdefault retain probe. Header roundtrip independently verifies18432activation/34816weight F32 values exactly. ActivationSHA060b08be2cda459303865addda3be3610617ee5a46aaaa4d2ccd162a4caf8109; weightSHA3921cb4a675792f0d3ecb1c84e8a3321537f4c3b61966ea44cb29792717ed734; headerSHA56dde9ad630398b2d5083f909ae43601361f91358670c259d16463182b0ee15c. Wrongdense export rejected before file reads; scriptscompile0.

Build92162 terminal0. Signed archive backup-hmx-frozen-gate-probe-candidate DSPB8001F89E2E6E1EDD6A1A38858458AEAADC4AFF5DC8EBA12239E8F9E9C7F5FAC/catalog339D57989F84AFB106AB2658A40768EDAC44BB73029E87D846B559E8EAE86D2A; includes generated header/exact source/config/compilerdefinitions. Both HMX-routed test-backend-ops arms native0/1of1 normaltest passes. Strict17fixture decoder exits0/all544metadataevents crossrunidentical. New frozen-model9x17K2048 fixture: raw4.92180021e-8 ->corrected5.67837589e-15, nonzeroHHlow152 and crossproducts exercised; wrongidentity1.81683218 rejects. Previous32correct/negativefixturechecks unchanged/pass. BaselineC9E4/D2D restoredverified/no livejobs.

This upgrades mechanism evidence from synthetic operands to a sampled real-weight/captured-input reference contract. Does not prove whole4096-output tensor, runtime packing/dequantization, full-model tokens, speed, or MTP. Next: diagnostic runtime correction for plain Q4 gate, separate owned lowparts with capacityguard and originalCPU capture/controlled128 gates. Preserve baseline and all original gates; precision work must not be promoted as throughput improvement without untraced paired benchmarking.

## Actual three-product HMX fixture mechanism passes twice

Extended existing OFFdefault retain probe with eight F32-origin operand fixtures: K32/256/1024/2048, padded9x17/full32, zero rows/cols, signed high+low HF operands. Sequential HH/HL/LH HMX products each retained high/low-exported, F32 summed, compared to original F32 operand float64 reference (LL omitted). Reuses existing owned act_f32 scratch before normal activation prep; capacityguard now133tiles+256bytes=272640, no host/layout growth. Explicit nonzero cross-product and main residual requirements. Normal production/defaultOFF unchanged.

Initial archive18546082... first arm native-1073740791 because incorrect fixture condition demanded nonzero final tiny-product residual: dyadic cross-products were HF-exact. Numerical errors already~1e-15..1e-14, but pass0, so not a passing run. Preserved backup-hmx-three-product-probe-candidate and hmx-three-product-probe-a artifacts. Corrected condition checks nonzero HH residual plus cross_used; numerical threshold unchanged1e-12. Decoder now independently checks numeric acceptance rather than trusts DSP pass bit.

Build67359 terminal0; archive backup-hmx-three-product-mainlow-probe-candidate DSP4B1CCA79EEF25CFD2513B94364B9D81A9E92B8D6DF661E8510B6F98EC6ED0394/catalog8E54F252AE3DCAE4494F073774AAF5862C9496BDFE262CC22825A7B14189AE2F. New archive-hmx-probe.ps1 checks actual compiled probe, all reference flagsOFF, baseline hostC9E4, captures source/config/ninja and restores baseline after signing. Both HMX-routed test-backend-ops arms native0/1of1 normal test passes. Strict decoder16fixtures/512metadataevents perarm exits0 and crossrun metric frames identical. All16new checks (eight correct/eight shifted-identity negatives) behave correctly: positive original F32 NMSE1.5748189e-15..1.16217088e-14; wrongidentity1.09..1.175. Old16checks unchanged/pass. BaselineDSP D2D restored/verified; no jobs live.

This validates synthetic operand-correction mechanism on hardware, not original model operands/runtime encoding, CPU token agreement, latency or MTP. Next: frozen model gate operands in HMX fixture before production integration, then owned low-operand workspace and actual originalCPU/token gates. Cross-product raw-HF export remains rejected by offline results.

## Resume: controlled F32 repeat closed; operand corrections screened offline

Controlled EBBA F32 residual repeat128 a/b completed: both native0/128 rows, strict same-backend bits/tokens comparison exit0. Both matching CPU comparisons fail first at token65 (CPU314/NPU364), versus prior Q4 residual token60. Earlier Q6-only token93 remains the longest intended-placement prefix, not a qualified decode. Trace rates14.20/14.20tok/s are diagnostic, not promoted performance. Baseline restored; no inference/build live at this checkpoint.

Paired gated-normalization frozen counterfactuals: T9 measured5.53117918e-7, predicted5.53067801e-7, recurrent-input-only7.46813292e-9, z-input-only5.56143858e-7, prediction residual9.85283507e-13. T1 measured3.03949272e-10, recurrent-only3.00951529e-10, z-only6.78152248e-16. These input swaps are offline diagnostics, not runtime interventions or whole-model causality proof. Refactored oracle self-pair reports exactly zero drift in all six difference metrics, native0; wrong-head rotation rejects NMSE86.9297058/native1. Original local1e-7 gate unchanged.

New FP16 operand sweep on actual EBBA T9 input SHA887f30af... and original GGUF Q4 weights: z weight-only rounding6.62006952e-9, activation-only5.06713033e-9, both1.14780064e-8; QKV3.67948398e-9/3.79364183e-9/7.46455191e-9. Neither single-operand fix removes the error. Ideal two-half operands represent this frozen Q8_0 contract exactly, but are not a runtime encoding/accumulator proof. Three products HH+HL+LH omit LL with ideal NMSE1.48991614e-16(z)/1.07835878e-16(QKV). Ordinary HF partial exports lose the benefit:4.42370024e-8/4.09757375e-8; ideal two-half partial exports recover1.80225584e-15/1.72111833e-15. Four products add negligible benefit in these captures. Both original local Q8_0 projection gates still pass, scripts py_compile0. Initial QKV name attn_qkv-0 correctly rejected missing capture; actual name linear_attn_qkv_mixed-0 then tested successfully.

Next measured hardware hypothesis: three-product operand correction plus retained residual export for plain Q4 prefill, initially z gate only. Requires separate owned high/low operand tiles, no alias with live DMA weights, explicit VTCM capacity/layout accounting, signed frozen fixture/negative controls, original CPU gate, then controlled repeat/token gate. Not implemented or speed-qualified. Naive extra products with ordinary HF exports are rejected as the current precision approach. Do not change already-correct gated normalization or T1 GEMV merely because cross-backend outputs differ.

## Actual F32 residual export improves alpha/beta and recurrent state proxy

Capture26306 terminal0/native0, qwen-layer0-recurrent-f32-residual-npu.tensors.bin2590853bytes, baselineC9E4/D2D restoredverified. OriginalCPUdensefixed1e-7 T9alpha/beta gatesnative0: alpha6.16643664e-8 ->3.5596877e-9 (~17.3x), beta4.78592811e-8 ->4.68340638e-9 (~10.2x). SameactualinputSHA887f30af... asprior89B0 alpha/beta. Bothnewoutputs matchidealHFoperandfloat64dot8.28776982e-15/7.09237548e-15, demonstrates recoveredlow export withoutchangingoperands. FullCPUF32roundingnotclaimed; HFoperandsremain.

PairedoriginalCPU14frames validatedT9/T1 finite/layout/requiredscope0. T9GDNattn_output3.85252446e-9 ->2.3820718e-9; final_output5.59605559e-7 ->5.53117918e-7; linearattnout3.80012807e-6 ->3.54668224e-6. T1GDNoutput1.30008983e-9 ->3.47419374e-10 (~3.74x); final_output3.27631326e-10 ->3.03949272e-10, butlinearoutcrossbackendNMSE EXACTsame1.49593544e-7 duequantizedprojectionboundary behavior. QKV/zunchanged supportsfocusedablation. Crossbackendmetrics notlocalCPUgatefailures; don'tpromotenearerstatealone. GatednormT9oracleexecutednextoriginal1e-7; controlledfull128repeat launched wholecache-hmx-plain-f32-residual-repeat128-a/b withcompiledwholecacheguard. Awaitterminal ->strictbits/tokens ->bothmatchingCPUfirstprefixdifference. EarlierQ4residualtoken60 stablepairremainsrejected; no performance/model/MTPqualification.

## F32 residual alpha/beta ablation capture running

Implemented OFFdefault GGML_HEXAGON_HMX_F32_RESIDUAL_REFERENCE for synchronousplain(single/multiweight)F32weight HMX. Existingworker/ownedraw-buffer6144bytes/nextDMAdeferred/exportF32/bias-once/valid-outputbounds reused; HFoperands remainunchanged. Bothdeferredprefetch branches nowselectnonquantheight=next_cols (quantheighttilecount unchanged), necessaryforF32. Selectedpipelinesreject explicitly, gatheredexpertsunchanged/defaultOFFnormalpathunchanged. Build60773 terminal0/diffcheck0, originalQ5T4upregression unchanged3.57194014e-14/native0. Wrongbetaheadsshape16 previouslycorrectlyrejected1; explicit32viewvalid. BaselineC9E4/D2D verifiedbeforearchive.

Sign/archive0: backup-hmx-plain-f32-residual-controlled-candidate, matchedhost2226402B..., DSPEBBAA8E0B42796BDEAD8524DD7C251C066CDF1542C2DF1D2AD0E34D7A30C3711, catalogC3982EFA94237BDA7C8D47BD4B99496E6ED47CD7AC87C4E00A4DB227533AFD19. Archiveguard verifiesactualcompiledF32/Q4/cache/scale controlsON, exactsource/build.ninja/cache stored. Existingbuildhelpers/resetsetF32residualOFF; newbuild-hmx-plain-f32-residual-reference.cmd setsitON withpriorcontrols. Newcapture qwen-layer0-recurrent-f32-residual-npu n2/same14filter live; preserveinstalledpairuntilterminal. Next originalCPUdensealpha/beta gates plusHFoperandideal, pairedCPUrecurrentT9/T1, gatednormoracle, thencontrolled128repeat/CPUfirstprefixdifference. No outputprecision/performance/fullmodelclaimbeforeactualtest.

## Next residual target identified: F32 alpha/beta HF output

Q4residual full128 strictrepeat exits0: all128 logitrowsbits andselectedtokens identical; armbCPU--require-tokens exits1 token60sameasarma. No promotion: priorQ6-only93 stillbestintended-placementprefix, allincomplete. Installedbaseline restored/no livejobs.

Extendedexistingmatmuloracle with dense_fp16_math/dense_fp16_output_math supportingdiagnostics and explicit --output-heads validatesONLY(1,heads,tokens,1) thenFortrancanonicalmatrix; ordinaryshape/defaultreferencegate unchanged. Pythoncompile0. FrozenactualT9F32alpha/beta fromnew89B0capture on sameinput887f30af...: originalCPUdenseNMSE6.16643664e-8/4.78592811e-8 (bothnative0), but denseHFoperand+HFoutput oracle matches BOTH EXACTLY (NMSE/maxabs0). IdealHFoperandwithoutoutputrounding CPUdistance3.55914582e-9/4.68025689e-9. OutputHFroundingdominates thesetestedprojections too; localdensegate pass is notfullmodelagreement. Gatednormoraclepasses1e-12 so nextfocusedablation canextendexistingplainresidualworker to F32 projections, notnormkernelchange. Requiresguardedraw-bufferownership/capacity/nonquantDMAheightfix (currentlyresidualnextDMAheight assumesquanttiles). F32weight/activationHFroundingwouldremain; no CPUexact/performancepromise. Pipeline/expertcoverageunchanged diagnosticlimits. Next: explicitwrong-output-headsshape negative and priorQ5regression, then F32residualOFFdefaultflag/compile/archive/capture/frozengates/full128repeat. Do notreplaceoriginalCPUdensegate withHFideal.

## Frozen gated normalization passes; Q4 residual fullpair finished

New check_frozen_gated_norm_math.py usesstrict88-bytecapturedframes, originalGGUFF32ssm_normweights and actualmodelRMSmetadataepsilon9.99999997e-7, perhead128 RMS * weights *stableSiLU(z) float64reference, F32secondary. Validatesoccurrence/tokens/shapes/duplicates/dtype/finite; original1e-7localgate unchanged. Pythoncompile0. T9CPU7.32464986e-15, previousNPU5816 8.61214566e-13, Q4residualNPU89B0 9.09842634e-13 allnative0. T1CPU1.60605165e-15/newNPU2.38063176e-12 native0. Wrongheadrotation negative NMSE86.927587/max17.67194 native1 correctlyrejected. Thus largecrossbackendgatednormdrift is input-driven/conditioning inthese captures, notlocalnormprimitivefailure; don't tuneRMSkernelblindly.

Full128Q4residual20496 terminal0/bothnative0; baselineC9E4/D2D restored. ArmaCPU--require-tokens exit1 firstdifference60 (CPU1061/NPU353), previouslyQ5scale-onlytoken15; stillworse thanpriorQ6-only93 prefix. Step0logitNMSE0.00612263828. Tracedrates14.14/14.09tok/s costlycachecontrol, notperformancepromotion. Strictfullpair verifier andarmbCPUcomparison executednext; preservebothnativeexits. No fullmodel/MTPqualification. Nextprecisiontargetdiagnosis: frozenF32alpha/beta onoriginalattn_norm inputs vsCPUfloat64dense andHFoperandmath, notalreadyCPU-matchingGEMV/gatednorm. Betacapture has(1,32,T,1) view requiringexplicitvalidatedcanonicalreshape inoracle.

## Q4_0 residual removes tested export error; downstream effects mixed

Capture42932 terminal0/native0/baseline restoredC9E4/D2D. Frozenlayer0T9Q4QKVoriginalCPUQ8_0 error4.88149871e-8 ->7.46452689e-9 (~6.54x), z5.60138932e-8 ->1.14785097e-8 (~4.88x), bothgate0. Exact inputSHA887f30af... samepreviousQKV/z captures, bothnewoutputmatchHFoperandfloat64ideal~8.04690978e-15/7.49331771e-15. Numericalimprovementverified onactualprojection, notsyntheticfixture. No newhost/model/gatheredexpert change.

PairedCPUrecurrentT9: QKV5.3435466e-8 ->1.2019236e-8; GDNattn_output1.85628452e-8 ->3.85252446e-9; z6.39458475e-8 ->1.91587643e-8, butgatednormfinal_output5.36185455e-7 ->5.59605559e-7 (notimproved), linearout3.60053792e-6 ->3.80012807e-6. T1GDNoutput6.715144e-9 ->1.30008983e-9/final_output1.16536161e-9 ->3.27631326e-10 improved, yetprojectedlinearout7.44765471e-8 ->1.49593544e-7 worsened, consistentquantizationdiscontinuities. These different-inputcrossbackendmetrics are NOT localprojection1e-7gatefailures; do notconflate. Alpha/beta/gates unchanged, supportingtargetedablation. Need originalinput frozen gated-RMS+SiLU oracle (attn_output,z,ssm_normweights->final_output) to distinguishlocalnormerror vsupstream/conditioning beforemorechanges.

Controlledfull128repeat nowrunning wholecache-hmx-plain-q4-residual-repeat128-a/b withexplicitarchivedcompilercontrolguard. Awaitterminal ->strictbits/tokenrepeats ->matchingCPUuntilfirstprefixdivergence. Earliercandidate5816token15 remainsrejected. No performancepromotion orclaimsfromshortretainedtraces.

## Plain Q4_0 residual candidate capture running

Build15616 terminal0, diffcheck0, archive/sign0. backup-hmx-plain-q4-residual-controlled-candidate: matchedhost2226402B..., DSP89B00CC769D16493F14036849F98E574EB47C8915AFFB69BBF0DE8FE653EEB46, catalogBE0A89CFC018ED902D33513F3AEC57F4A5926249584AED0C0DE3FF9632EC4983. Archivehelper RequireWholeCache/RequireScaleF32/RequireQ4Residual checkedbothcacheON andactualbuild.ninjadefines beforearchiving; exactsources/config/compileddefinitions stored. Same previous controls plusQ4residualON; nohost/GGUF/gatheredexperts/layoutgrowth change. Capture42932 live qwen-layer0-recurrent-q4-residual-npu n2/filteridentical toprior5816recurrentcapture. Do not restorepairwhileCLI live. Afterterminal0: frozenQKV/z originalQ8_0 local1e-7 +HFoperandideal, compareagainstpriorCPUrecurrentcaptureT9/T1 beforefull128controlledrepeat. Stillno performanceclaim.

Possible later precision ablation, only ifrecurrentdrift evidencejustifies: small32-output F32alpha/beta projections useHMXHFinputs/weights/output atT9 (crossbackend~6.2e-8/4.8e-8) whileT1HVX~1e-14; matchedhost/DSPprecompute couldforceexistingHVXF32 kernel forsmallplainF32 projections withoutCPUfallback. This is a futurehypothesis, not implemented or benchmarked. Do not changeexistingCPUreference orweaken gate.

## Prefill recurrent drift motivates plain Q4_0 residual ablation

NPUrecurrentcapture15077 andCPU39398 terminal0/native0, matchingretentionfilters/n2; baselineC9E4/D2D restored. Strictrequired14tensorT9/T1framecomparisons0(reportingonly). T9 q/k/vpredelta crossbackendNMSE4.36848709e-8/8.68686329e-8/1.64632751e-8; beta4.78456866e-8/sigmoid3.42048665e-9; alpha6.16539863e-8/gate3.31177242e-10. GDNattn_output1.85628452e-8, z6.39458475e-8, gatednormfinal_output5.36185455e-7. T1q/kpredelta~4.9e-13/1.5e-13, currentgate/beta/alpha/z~1e-14, butGDNattn_output6.715144e-9 fromdifferentprefillhistory; statecarryover hypothesis, not proof ofbadGDNkernel. No snapshots ofinitialstatehere.

FrozenoriginalCPUQ8_0T9projectiongates pass: Q4_0QKV4.88149871e-8 and Q4_0z5.60138932e-8. OriginalHFoperandidealCPUdistances7.46455191e-9/1.14780064e-8; actualoutput-to-HFideal4.0975371e-8/4.42376897e-8. Thus outputHFrounding dominates testedearlyplainQ4projections evenwhilelocal1e-7 passes. Next targetedablation OFFdefault GGML_HEXAGON_HMX_Q4_0_RESIDUAL_REFERENCE extends existinghigh+low worker to synchronousplain(single/multiweight)Q4_0 ONLY, Q5pathstillON. Sameownedrawweightscratch+deferrednextDMA/capacityguards; pipelinesrejectselectedtypes; gatheredexpertpathsunchanged. DefaultOFFnormalbehaviorunchanged. ExistingbuildhelpersresetQ4residualOFF; newbuild-hmx-plain-q4-residual-reference.cmd setsQ4ON plusprior5816controlsON. BuildstartedafterCPUterminal/noinference; awaitterminal0/archivecompiledcontrols/newcapture ->frozenQKV/z andpairedrecurrentinputdrift ->strict128repeat/CPUgate. No performance or fullmodelpromotion.

## Paired-input oracle explains firstdecode projection drift

Malformedpairedlogitsfile negative correctly exits1 Invalidtensorheader; originalQ5T4upregression unchanged3.57194014e-14/exit0. New recurrent-input capture started qwen-layer0-recurrent-inputs-controlled-npu n2/5816candidate/compiledcachecontrolguard: retain initialnorm/QKV/z/beta/alpha/gate/qkvpredelta/attn_output/final_output/linearattnout, NOT large recurrentstate plane. Need pairedCPUafterNPUterminal; inspect where prefillGDNinput/gate/norm drift begins before designing ablation. No projectionkernel accusation or relaxedgate.

Extended existing frozen matmul oracle with optional --paired-trace, currently strictQ8_0referenceonly. Refactored same strict88-byte framed capture reader forbothfiles; same occurrence/tokens, duplicates/dtypes/dims/finiteness validated; GGUF originalweights shared; optionalsampledrows retained. Reports referenceQ8code/FP16scale changes, actualcrossbackendoutputdelta, predictedquantized-inputdelta, denseinputdelta, residualdifference, pairedCPUlocalmath. Original declaredreference1e-7 exitgate unchanged; pairedmetrics are mechanism diagnostics, NOT newgate/runtimequantbyteproof. Pythoncompile0.

T1originalNPU/CPUcapturedinputs, all3localnative0: ssm_out7/4096codes and11/128scaleblocks change; predictedoutputNMSE7.44754169e-8 vsmeasured7.44765471e-8, residual1.33813962e-14 (fraction1.79672618e-7). Sharedup28/2048codes,14/64scaleblocks; predicted2.20428538e-6/measured2.20432293e-6, residual2.26795111e-14 (fraction1.02886518e-8). Down43/512codes,12/16scaleblocks; predicted2.51004751e-5/measured2.5100513e-5, residual9.1467726e-15 (fraction3.64405803e-10). CPUcapturedoutput on owninput matchesoracle~1e-14 too. Quantization-boundary input effects therefore explain these testedT1projectiondifferences to nearfloatingroundoff; this does notidentifyoriginalprefill/statecause orprovewholegraph.

T9ssm_out464/36864codes/522of1152scaleblocks; measured3.60053792e-6/predicted3.52862984e-6/residual8.60802187e-8, residualfraction0.0239075995. Sharedup1407/18432codes/376of576scaleblocks; measured1.1556133e-5/predicted1.15243484e-5/residual7.23241684e-8, fraction0.00625850951. Predictedinputeffects dominate, localHMXHFerrorstill contributes. Self-pairT1down native0: allcode/scaledifferences0 and predicted/measured/dense/residual/fractionexact0, localoriginalgate unchanged5.07756701e-15. Malformedpaired-logitsfile negative run and originalQ5T4regression run next; preserve nativefailures. Baselineinstalled/no liveinference. Next causaltarget: earlierlayer0prefillQKV/gate/recurrentnorm input precision, notalreadyCPU-matchingT1GEMVs.

## Layer0 frozen projection gates pass; upstream drift remains

PairedCPU17888 terminal0/native0; first2selectedtokensagree andsamecontrolledlogitdistances asall-layercapture. StrictT9/T1framecomparison finite/required/layoutscope exits0(reportingonly). T9 final_output inputNMSE5.36185455e-7 ->projectedlinearout3.60053792e-6; postnorm2.92471968e-6 ->sharedup1.1556133e-5/gate5.04779038e-6 ->swiglu2.10074595e-5 ->shareddown7.14148731e-5. T1 final_output1.16536161e-9 ->linearout7.44765471e-8; postnorm8.34112844e-8 ->up2.20432293e-6/gate1.05554232e-6 ->swiglu3.83428222e-6 ->down2.5100513e-5. Combined withlocalT1projectionCPUoracles~1e-14 onowninputs, this strongly motivates explicit paired-input quantization-boundary analysis rather than accusing theseGEMVkernels. Quantizationdiscontinuity is a hypothesis until actual Q8 codes/scales and pairedfloat64projections are compared.

No livebuild/inference after pairedcapture; baseline installed. Next safe action: extend frozenprojectionoracle optionalpairedtrace usingstrictsame-occurrence/tokens capturevalidation, quantifyactualcrossbackenddrift vsCPUQ8_0 projections fromeachinput, and int8code/scalechanges versusdenseinputeffect. OriginalCPU1e-7 gate remains separate/unchanged. Then target causallyidentified prefill/state precision withablation; do not promotelocalonlypass orweaken fulltokenrepeats.

Narrow NPUcapture29782 terminal0/native0, qwen-layer0-projections-controlled-npu; baseline restoredC9E4/D2D. AuthoritativeGGUFmetadata: layer0QKVQ4_0; ssm_outQ8_0(4096->2048); shared up/gateQ8_0(2048->512), downQ8_0(512->2048). Do NOT treat layer0sharedFFN asQ5_K. No modelrewrite.

OriginalCPUQ8_0 local1e-7 gates (fullmatrices oncapturedNPUinputs) all6native0: T9ssm_out8.60597095e-8/max0.000351205115, sharedup7.23297217e-8/max0.000829861239, shareddown9.75732102e-8/max0.000229695384. T1ssm_out2.69740167e-14/max4.88378333e-8, sharedup1.89918133e-14/max4.17186129e-7, shareddown5.07756701e-15/max2.10823146e-8. IdealHFoperanddistance atT9(~4.1e-8) vsoutput indicates HFexportrounding contributes, but localprojections passunchangedgate. T1nearCPUexact on owninputs despitecrossbackendlayerdrift means cannotblametheseprojectionkernelsalone. Firstlayerglobaldrift couldbe differentupstreaminputs/state/quantizationboundarychanges; no causalproofyet.

Paired narrowerCPUcapture launched17888, qwen-layer0-projections-controlled-cpu, samefilter/n2/prompt/context. Awaitterminal then compare first2selectedtokens and T9/T1per-name frames, specifically final_output-0 vs linear_attn_out-0, postnorm/up/gate/swiglu/down; optionalpaired-input frozenoracle can quantify quantizationdiscontinuities rather than assumekerneldefect. No newkernelchange, benchmarkwin orpromotion thisturn. Fullgoalactive.

## Layer drift localized before more kernel changes

Paired layer trace NPU56903/CPU76012 terminal0/native0, sameprompt13/context512/n2/filters. NPUwrapper baseline restoredC9E4/D2D. Selectedtokens agree bothfirst2steps; fullcomplete logits0/1 match previously reported controlledrun values (not claim for latersteps). Extended existing compare_tensor_traces with optionalname-regex/zero-basedper-nameoccurrence/all-differences, strictlayout/order/framing/requiredtensors and finiteness remain. Pythoncompile0. Reporting-only crossbackendcomparison exits0 means scope valid, NOT equality/promotion.

ActualT9occ2: layer0attn_norm NMSE1.6755037e-14; QKV5.3435466e-8; convraw1.81169192e-8/SiLU1.66231129e-8; linearattnout3.60053792e-6; attnpostnorm2.92471968e-6; MoEFFN2.78276354e-5/sharedFFN7.14148731e-5; l_out0 6.70163098e-6. l_out1 0.000160669571, layer21 0.00243472748/layer22 0.0219368806; final39 0.00174338279. This is accumulateddifferent-inputcrossbackenddrift, not an operation-localCPUoracle or proof of erroneousMoErouting.

ActualfirstdecodeT1occ4: norm1.20602232e-14/QKV4.72180951e-15/convraw2.25775969e-15/SiLU9.0699812e-15, linearattnout7.44765471e-8, MoEFFN1.79164045e-5/sharedFFN2.5100513e-5, l_out0 1.64501025e-6/final39 0.00459001093. This points to attentionoutput+FFN work rather than initialnorm/QKV in thisfirstdecode; recurrent state came fromdifferentprefill, so cannot blame outputprojection alone.

Next narrower qwen-layer0-projections-controlled-npu capture running n2/same5816candidate/RequireWholeCache: retain attn_norm/qkv/final_output/linearattnout/postnorm/up/gate/swiglu/shared/MoE/ffnout/l_out0. Need actual frozen outputprojection and sharedQ5 up/down CPUoracles on capturedNPUinputs, then pairedCPUwithsamefilter ifneeded; do not conflate upstreaminputdifferences with kernelerror. Fullgoal remainsactive/unqualified.

## Controlled Q5_K local success does not improve full CPU token prefix

Full128 session52729 terminal0/bothnative0; labels wholecache-hmx-q5-scale-residual-repeat128-a/b. Strict repeat verifier --require-identical exits0: all128 complete logits rows bit-identical and selectedtokensidentical. Both original matching CPU --require-tokens comparisons exit1 atstep15, CPU4602/NPU264, CPUmargin0.00659942627/oppositemargin0.106884003. Step0CPUlogitNMSE0.00854300854; no laterprefix comparison afterfirstdifferenttoken. Traced whole-cache rates14.02/13.91tok/s are not performance qualification. BaselineC9E4/D2D restored verified. Candidate5816995F remains NOT promoted; sixlayer5projectionpasses are not fullmodel agreement. Previous93tokenQ6-only remains best intended-placement reference prefix, not qualifieddecode.

Next changed from more shared-projection factors to layer drift localization: qwen-layer-drift-controlled-npu captureTokens2 uses same candidate5816 and normalfusion1, retains all40l_out plus firstlayer attn_norm/qkv/conv/linearattn/residual/postnorm/expert/shared/ffnout, excludes largest recurrentstate snapshots. Paired CPU capture must use identicalfilter/prompt/context/firsttwotokens; snapshotretention altersgraph so resultslocalonly, not bench. run-q5-weight-control now accepts CaptureFilter and RequireWholeCache; new trace requests require archivedbuild.ninja actualcompilerdefine. Archive helper can explicitlyRequireWholeCache/RequireScaleF32 and archivesbuild.ninja; defaults no longer imply wholecache in name. No newmodelmutation/installpromotion/MTPclaim.

## Explicit cache control and F32 Q5_K scale capture

Oracle21916 terminal0: all6 originalCPUQ8_K1e-7 gates PASS. T9up/down7.60079386e-8/4.37161369e-8; T4up/down3.57194014e-14/1.53254675e-14; T1up/down3.35092179e-14/1.66677706e-14. Maxabs T4up/down4.09228732e-7/6.08574169e-8; T1up/down3.45246361e-7/2.94364555e-8. T4upinput is identical to earlier failedF16scale capture. Actual T1down would fail previous F32weight-with-F16activation-scale representation oracle1.07695065e-7, while new fullF32scale runtime matches originalCPU locally; threshold unchanged. This is layer5 projection proof only. Controlled full128 repeat started next, labels wholecache-hmx-q5-scale-residual-repeat128-a/b; compare strictbits+selectedtokens, then bothCPUmatchingprefixes, no speeds promoted.

Build21480 terminal0, sign/archive0: backup-hmx-q5-scale-residual-controlled-candidate, host2226402B762C8E2233059CEEF1D686BE51731FBAD70B24BFA23A75F2C08A2CDD, DSP5816995FEDDE92D0D2B683BD8AAE3B348C4FD846FB1F895BC3DB6F5744F14C9C, catalog1A54A006A4EF30FFE0F7BCBB8C9FC9E18B3938FAEC7A53BDF1E4E29357945B4A. Actual build.ninja compiler definitions independently verified scale/cache/residualON and archived alongside CMakeCache/exactsources. Capture37824 native0/wrapper0, shared-q5k-scale-residual-controlled-npu.tensors.bin407989bytes; includes firstdecodeT1, baselineC9E4/D2D restored verified. gitdiffcheck0 (CRLF warnings only).

New check-q5-projection-capture.ps1 checks six actual layer5 up/down frozen projections atT9occ2/T4occ3/T1occ4, always originalCPUQ8_K1e-7, reports each nativeexit and aggregatefailure; does not weaken gate or claim full-model scope. Oracle21916 still running: T9up7.60079386e-8/down4.37161369e-8 bothnative0; T4up3.57194014e-14/max4.09228732e-7 native0. T4upinput f7bf002e... identical to previous F16scale failingcapture1.96769885e-7, strengthens activation-scale explanation. T9 high+low results unchanged numerically despite cachecontrol being explicitlyON. Await remainingT4down/T1gates/aggregateexit before full128 repeats. No speed/fullmodel/MTP promotion.

## Control provenance correction and small-M scale diagnostic

Standalone CPU-versus-armB --require-tokens verifier exits1 atselectedtoken60 (CPU1061/NPU353); armA exits1 at93. These uncontrolled results do not establish a reproducible new best decode. New small-M scale diagnostic currently uses unchanged original CPU oracle for promotion; any fullF32factor sweep is supporting mechanism evidence only.

IMPORTANT: residual archives FB5AB267... and04AC5B9D... and their labels say wholecache, but archived main.c has NO per-operation whole-cache control. They used normal batch-level cache flushes only. This invalidates treating their full repeats as whole-cache-controlled. Keep artifacts/names immutable; do not blame residual export for instability. Repeat session42160 terminal0, both128 rows/native0; strict same-backend compare exits1, bits differ atstep0/selectedtokens first differ60. CPU versus armA firstselecteddifference93; armB must be compared separately. Rates19.90/20.01tok/s are unqualified uncontrolled diagnostic traces, not qualified speedups. Baseline restoredC9E4/D2D. Local captured T9 oracle/inputhash findings remain local evidence only, not repeat-robust proof.

Added explicit OFF-default GGML_HEXAGON_PER_OP_WHOLECACHE_REFERENCE in DSP CMake/main.c, exactly earlier control flush-all+dirty-range-clear before source flush. Added OFF-default GGML_HEXAGON_Q5_K_SCALE_F32_REFERENCE: Q5_K HVX quantization writes full F32 Q8_K scale into existing128-byte activation scale vector (only with_offset && q8_k_base); Q5_K dot helper multiplies appended F32weightscale by this F32vector. No allocation/layout/host-format growth; depends matched F32weight diagnostics. Default path retains FP16 scale and existing behavior. This directly addresses T4 oracle exposing remaining FP16activation-scale error, without relaxing CPU1e-7. Existing8referenceflagsON plus HMXresidualON/newscaleON/wholecacheON/retainprobeOFF build21480 live. No install, baseline remains. New build wrappers explicitly set scale/cache flagsOFF except new build-hmx-q5-scale-reference.cmd sets bothON, preventing stale-cache diagnostics. Next terminal0 ->new archive/new captured T9/T4/T1 ->original CPU gate ->controlled128repeat/CPUcomparison. Full goal incomplete.

## Q5_K model residual-export diagnostic in progress

Extended synchronous multi-weight build8960 terminal0/sign0, archive backup-hmx-q5-residual-nx-wholecache-candidate: matched host2226402B..., DSP04AC5B9DA95B43659BD4F92E5255A088887DF9745BA236C45DCE328F9016FAE1, catalogD72D2243.... Capture60354 native0/wrapper0, shared-q5k-hmx-residual-nx-prefill-npu.tensors.bin384964bytes, baseline restored verified. T9 original CPU Q8_K fixed1e-7 gate: up7.60079386e-8/max0.000414958258 PASS; down4.37161369e-8/max7.45557402e-5 PASS. Ideal Q8_K->HF operands/float64dot distances up1.03217506e-14/max3.16780643e-7, down4.23631012e-13/max5.46424417e-8. Up input bc700937... is byte-identical to previous F32-weight/HF-output failing capture (up1.22052029e-7); removing output HF rounding explains this local improvement. Down input differs51942a8f..., so down is not a same-input A/B claim.

T4 occurrence3 up CPU Q8_K1.96769885e-7 FAIL; down8.53313437e-8 PASS. Existing F32-weight HVX oracle matches up2.75984901e-14/down1.17253919e-14, unlike HMX operand ideal; this suggests routing to unchanged HVX small-M path, not failed residual export. Full-model promotion REJECTED by T4 gate. Full128 repeat diagnostic still launched to measure aggregate impact, not qualification: session42160, labels wholecache-hmx-q5-residual-nx-repeat128-a/b. Await terminal then compare bit-identical repeats and matching wholecache-chat-cpu-reference128 until first divergent prefix; preserve native failure if any. Next numerical work: route-verified small-M/HVX Q5_K activation-scale precision diagnosis and captured T1. All future build-q5-weight-reference, build-hmx-q8k-reference, build-hmx-retain-probe now explicitly reset residualOFF; restore helper resetsOFF. No performance claim.

First residual build session41310 exited0 and signing/archive succeeded: backup-hmx-q5-residual-wholecache-candidate, host2226402B..., DSPFB5AB2679F08AED903BEA3B8A9E3D6B23E9B328F4857282A13DD6039455AFC2E, catalog9538D2FF.... Capture shared-q5k-hmx-residual-prefill-npu session49954 failed native -1073740791 / wrapper1 with DSP INVAL-PARAMS during actual13-token prompt; 91106-byte partial capture is NOT a numerical result. Baseline hostC9E4/DSPD2D restored and verified by wrapper. Normal fused graph can enter multi-weight Q5_K, deliberately rejected by first diagnostic. Source now extends serialized raw-weight scratch ownership/export across each synchronous multi-weight output; pipeline guard remains. Incremental residual-enabled rebuild started; must archive under a NEW name and capture with NEW label after terminal0. No performance claim or correctness promotion.

Added OFF-default GGML_HEXAGON_HMX_Q5_K_RESIDUAL_REFERENCE for plain synchronous HMX Q5_K. After dequantization, the raw-weight VTCM region owns three 2048-byte tiles (high, low, negative identity). Next weight DMA is delayed until HMX export finishes; capacity is checked. Existing activation/dequantized-weight/scales/bias regions remain separate. High is stored with accumulator retention, high*(-I) extracts low, and scalar F32 high+low is written to valid output elements with optional F32 bias once. No shared layout or weight-format change from the existing matched F32-weight diagnostic. Pipelined Q5_K and multi-weight Q5_K reject explicitly; gathered expert paths are not covered.

Build script build-hmx-residual-reference.cmd restarted in live session 41310 with final sources; baseline installed binaries have not been changed. First session 50790 exited 1 after its first DSP link because the called CMD file changed while CMD was reading it (stray truncated command); no residual-enabled build success claimed for that attempt. Do not edit running CMD files. No actual model correctness or speed result yet. Next: await build exit, archive signed matched host/DSP candidate, capture layer-5 T9 up/down and compare original CPU Q8_K at unchanged 1e-7 gate, then repeated full logits/token checks only if local gate passes. This scalar/no-prefetch diagnostic is not a performance candidate.

## Long-K retained residual mechanism passes32cases across2repeats

Extendedexistingprobe dotcounts1/8/32/64 ->K32/256/1024/2048, eachfull/paddedM9N17 andcorrect/shiftedidentity. HMXdeep loadsmax32tiles/range65535, K2048splits2deep loads preservingacc beforeHFretain export/correction. Ownedexistingactivationprep scratch now131tiles+256scales=268544bytes, capacitycheckedrelease; no sharedlayout/hostchange/VTCMgrowth. Scalarfloat64reference usesactualHFoperands acrossallK; F32high+low comparedsame. Decoder --fixtures8 reads256metadataevents; new --require-identical comparesmetricframes only(notcycles).

Build23632 terminal0/sign0, DSPACB04E9C13BF574AB516C0F412A1200416CBD1595E336B7D80994D58D95665D1 archived exactsources/so/catalog backup-hmx-retain-longk-candidate. hmx-retain-longk-v1-a/b native0/exactly1normalbackendtesteach/HMXroutingverified. Strictdecoder --fixtures8 --require-identical exit0 all32checks+framesidentical. CorrectedNMSE K32zero; K256padded/full1.05567275e-14/1.08507148e-14; K1024 1.45337598e-15/1.58272606e-15; K2048 2.23116017e-15/2.28793043e-15. Nonzeroresidual counts114/836,122/906,98/702,122/906. ShiftedidentitynegativesallNMSE~1.09..1.175 correctlyrejected, mechanismsamefixed<1e-12 /negative>.001. Dyadicfixtures notactualQwen frozenmatrix/no generic accumulatorprecision claim.

BaselineD2D/catalogrestored/hashverified, root/nestednormalconfigs restoredterminal0 all9OFF/certnormal. No livejob/inference/build. Next actualprojectionresidual integration: reserveownednegativeidentitytile inpermanentGroupA, separate lowHFoutputplane perpingpongGroupB output, preservehighplane; sharedlayoutcost size_per_mn currently(pipeline?2:1)*2bytes mustaccountbothplanes; overheadaddidentity/padding. Roothostprecomputes sameheaderasDSP, so newlayoutflag requiresMATCHEDhost/DSP config+archive/install/restore (notDSP-only). Threadlowptr throughHMXjob andoutputtransferworkers; high+low F32 thenoptionalbiasaddonce, retaindefaultpath unchanged. Auditrow/chunk/gather/groupedpipelinelifetimes beforeenablinganypath; originalCPU1e-7projection/fulltokens/repeatbitsgates remain. ExtraHMXidentityproduct/store/HVXsum mustprofileaftercorrectness; no speedwinclaimed. Goalactive/incomplete.

## Retained HMX residual extraction passes focused hardware fixtures after bias0 correction

Build28924 terminal0/sign0, exactDSP931AD6F8F4FCEA69A555512A7922092F1DE9B3DDFB4CBB570503031C4BC60F1A archived backup-hmx-retain-bias0-candidate. hmx-retain-bias0-v1-a/b native0, exactly1existingbackendtesteach, hostlogsverifiedhmx-tiled (q4_0/f32m16n8k4096). Strictmetricdecoderbothexit0: paddedfixture rawNMSE7.69357271e-8/corrected0/nonzero114, full7.49146736e-8/corrected0/nonzero836; shiftedidentitynegatives corrected1.09103572/1.17524505/nonzero136/992, bothcorrectlyrejected. Numericfieldsidenticalacrossbothruns (timingnotcompared). Probe thresholds unchangedcorrected<1e-12, negative>.001; scalarfloat64dyadicHFoperandoracle. Bias1fixturebug, notretentionhardwarefailure, accountsforpriorhuge rawerror; baselineproductionalreadyusesbias0.

Mechanismvalidatedonlyfor2controlled32x32,K32HFtiles (paddedM9/N17 andfullsigned/zeros), not modelprojection/longK accumulation/biasfusions/pipeline/replay/aliasing. Existingbackendtest5e-4 normaloutcheckalso passesbutnotCPUmodel1e-7 gate. Residual usesretainHFhighstore +HMXhigh*(-I) +HFlowstore, F32high+low equalsfloat64referenceforthesedyadicfixtures. NounsupportedF32storeinstruction, no modelhack/thresholdrelaxation.

WrapperrestoresbaselineD2D/catalog/hashverified, hostC9E4unchanged. Restorehelperterminal0 resetsall9diagnosticflagsOFF/certnormal; no livejob. Next implementOFFdefaultfullprojectionresidualexport with explicitownednegativeidentity/high/low buffers, layout/VTCM accountingmatchedhost+DSP ifsharedlayoutchanges. ExtraHMXidentityproduct+secondoutputstore/HVXsum havecost; measurelaterafterfixedsameinput projection/originalCPU1e-7 gates. PreserveF32Q5weight/Q8Kactivationcontrols. NeedlongK32/64tiles/paddedtail/zero/negative/nearULP/overflow-underflow/fusedbias andwronglayout controls, notassumetinyfixtureworksuniversally. Fullgoalactive/incomplete.

## Numeric HMX probe metrics recovered; fixture bias initialization corrected, build28924 live

Existing trace-event metadata now transports fourfields perfixture via FENCE-tagged frames: 0x7000|(case<<4)|field followedby3chunks of32bitvalue(15/15/2bits). Actualcyclefieldstimestampsunchanged; diagnosticmetadata notperformancetiming. TwoF32 metrics raw/correctedNMSE +nonzero_low/pass,64events onctxtrace[HTP_MAX_NTHREADS]. No host/RPCformat change; existingPROFILE3/OPTRACE1024 outputs beforestatuscheck evenonDSPfailure. Newstrictread_hmx_retain_probe.py validatesframe order/length/range/finiteness/count/pass; exits1onactualfailedchecks, doesnotrelaxprobe/modelgates. Pythoncompilepassed.

Instrumented18860 terminal0/sign0, exactcandidateA47D263C2AA9B807A78163D6E09879ECDDE4D784EFB63D45E216F066954C2464 archived backup-hmx-retain-trace-candidate. HMXtrace-v1-a confirmedroutefailsnative-1073740791; decoderexit1 withraw/correctedNMSE padded5.81730127/5.81728411/nonzero128/pass0, full0.721534133/0.721568465/nonzero961/pass0; shiftedidentitynegativepasses(corrected6.90812016/1.89647114). BaselineD2D/catalogrestored/hashverified. Thesehuge rawerrors precederesidual extraction, so doNOTattributefailuretohardware retention.

Sourceaudit identifiedfixturemistake: probeQ6_V_vsplat_R(0x3c003c00) initializes scale1/bias1, workingkernel uses0x3c00 scale1/bias0. Correctedprobe tosame0x3c00; no gatechange/no productionarithchange. Build28924 live linking, process955460CPU21.21875 verified. InstalledbaselineD2D; no inference. Pollsamehandle, no restartontimeout; thensign/archiveNEWcandidate andrerunfreshlabel PROFILE3/routing/decoder. OnlyRETAINflagON/all8precisionOFF. Needcleanmetricsbeforeclaimretainworks; preservefailed bias1fixturesasdiagnostic negativeevidence. Fullgoalactive/incomplete.

## Retained-accumulator hardware probe compiled; actual HMX case fails probe

Added OFFdefault GGML_HEXAGON_HMX_RETAIN_PROBE inDSP only. Existing plainHMXentry runs queuedHMXworker using5tiles+256scales inexistingownedactivationprep scratch BEFOREnormalA/Wloading. NoVTCMgrowth/hostformat/newtestsfile. Fixtures full32 andzero-paddedM9/N17, signed/mixeddyadicHFoperands, firstrow/colzero. Exporthigh :after:retain.hf, subtracthigh vianegativeidentity, exportlow :after.hf, comparehigh+lowF32 vs scalarfloat64sameHFoperanddot; nonzeroresidual/rawrounding required. Shiftedidentitynegative mustproduceNMSE>.001. CorrectedNMSE<1e-12 mechanismprecisioncheck, separatefromCPUmodel1e-7 gate. Internalfailure returns-1; FARFmetricsnotvisible inhoststderr so failingfixture/metriccurrentlyunknown.

Build55454 terminal0/allCobjects+link, signhelperterminal0/signing0. Archived exactsources/so/catalog in backup-hmx-retain-probe-candidate DSP4A2D0617EBB514CE6392471529CBBCF23F7D24C7B47134D13E4AA4B4CDC1DAE4; standardbaselinehostcompatible, no wholecacheperopcontrol inthisprobe. Candidateflags onlyRETAINON/all8precisionOFF. Updatedrestore/precisionbuilders explicitlyRETAINOFF topreventcachedoptionleak. Sourceexistingcore_mma usesoutputtileasactivation withidentity, usefullayoutprecedentnotproofnewretentionsemantics.

Runnerinitialstderrredirect+ErrorActionStop aborted onnormalnative stderr; fixedusinghiddenStartProcess/processexcodes. v2 lackedADSP_LIBRARY_PATH andfailedsession0x80000406; fixedexplicitpkg librarypath. v3 f16/f32 m16n8k4096 bothnativeexit0/1of1 passed BUT hostlog sayshvx-tiled (nonrepackedN16not32aligned), so NOTprobeexecution/proof. Keptallfailureartifacts. v4 q4_0/f32 samedims hostlogCONFIRMS hmx-tiledvtcm788480, thenDSP INTERNAL-ERROR/native-1073740791 beforetestcompletion. Probechecks orruntime mayfail; doNOTclaimretainedresidualworks/no roundingfixqualified. ArmBnotlaunched afterfailure. Existingtest_nmse5e-4 notCPUmodelgate; intendedverificationisstrictinternalprobeplusrouting, notbroadtestpassalone.

WrapperfinallyD2D/catalogrestored everyattempt, hostC9E4 reverified. Normalroot/nested configs restored afterv3; v4installedarchivedprobe withoutbuilding, installedbaselineagainfinally. No livejob/inference/build. Neednext checknew9flagsOFF cache, obtainnumericfixturemetrics (FARFALWAYSnotforwarded), separatecorrect/negativefixturefail andretention/outputlayoutcause, thenrerunfreshlabel. Nofullmodeltest/performanceclaim. Fullgoalactive/incomplete.

## Split-K export rejected offline; two-half residual representation remains a hardware hypothesis

Added optional --hmx-k-split-sweep to existing oracle: full256Q8_K quantization preserved BEFORE splitting, originalGGUFweights/inputFP16, float64partialdots->FP16exports->F32sum, countstores peroutputtile. CPU gate remains explicitoriginalreference, nobestchunkpromotion. Actual1A0Dup/T9 chunk32..2048 (64..1stores) originalCPU NMSE1.12011394e-7..1.22052614e-7, allfail. IndependentCPUupinput allchunks1.07537929e-7..1.14839559e-7, allfail. Downalltestedchunks32..512 pass6.77241843e-8..8.66159987e-8 butuprejectsfix. Nohardware splitK implemented, avoidextraoutputstore/traffic work absentaccuracybenefit.

Canonicalinputhash reportsadded: old9A66/new1A0Dup bothbc7009372593cef50ea66977fb5a53da34beef594ef7c9ba18a07b27385c33f6; now verifiedsameinputbytes (sameGGUFweight). Thus upweightdequant improvementnotupstreaminputconfound. CPUupinputdifferent9e6a2f44dded7ac1deab525b80c11dd4db3b4ec21548e186e8864cee494b0878; downnew37fd6733ebec6c274748fa33a8d0fe0c08469cc62ee3b803c6ed6828ac2b162a. This doesnotisolate scaleF32 vsqf16bypass, bothchanged.

Explicit1A0D q8_k_fp16_output_math gatesexit0 upNMSE0/down2.73738399e-16, originalCPUupstillfails. RegressionCPUoriginalQ8_Kexit0. Offline idealhighHF(dot)+lowHF(dot-high) summedF32 givesupCPU NMSE7.60070296e-8/pass; computedfromidealoperands only, notexpectedcapturedoutput. Representationcapacityonly: actualresidualexportunproven. SDK :after:retain.hf suggests testretain-store/subtract-rounded-output viaHMXnegativeidentity/exportresidual, butmustverify retainedacc semantics/layout/rounding/bias andDMAownership in focusedexistingharness beforeproduction path. No :after.sf unsupportedassembly assumed. PreserveoriginalCPU gate andwrong32negative.

No livejob/build/inference; all8flagsOFFverified, baselineD2D. Next inspectoutputtile-to-activationlayout forretainedresidual spike; use existing backendops/harness, notfullmodelperformanceclaim. Needtestzero/negative/near-halfULP/paddedM/K andwrongidentity mapping, repeatability. Fullgoalactive/incomplete.

## F32 HMX weight dequant closes operand gap; up fails final-output rounding gate

Capture44256 terminal0/CLI0, all3baselinefilesrestored/hashverified hostC9E4/DSPD2D/catalog. Actual layer5occurrence2/T9 F32weightHMX: up originalQ8_K1.22052029e-7/max0.00073123109 nativeexit1; down8.1984324e-8/max0.000404052266 nativeexit0. Fixed1e-7 unchanged. Idealoriginalweight/inputFP16 oracle withfinalFP16output: upNMSE0/max0, down2.73738399e-16/max5.96046448e-8 (valuesprinted by commonoracle, explicitreferenceexitpending). Withoutoutputrounding residual4.53253323e-8/4.428068e-8. OldF16D/Mweightoracle nowfails5.91854657e-7/3.44809712e-7. This capture matches intendedideal operand/output representation closely; no inference of generic HMX accumulation equivalence outsidethisshape/capture.

Thus tested up's originalCPUfailure is accounted for by idealFP16operands/output onitsowncapturedinput (oraclecpu_contract_nmse1.22052614e-7). Down passesoriginalCPUlocalgate. Do notlaunchfull128 asfullylocallyqualified orrelaxthreshold. Need F32output/correction path investigation next, preservefrozeninputandnegativecontrol; inputhashreport could establish old/newupinputidentity beforecausalcomparison. This prototype bypassesqf16 pluspreservesF32effectiveD/M, cannotattributeeachseparately. Restorehelperterminal0/configureonly resetsall8flags androotcertnormal; verifycacheflagsnextturn. No livejob. Fullgoal active/incomplete.

## Matched F32-weight HMX frozen capture44256 live

Separate whole-cache build/sign5801 terminal0/signability0/signing0. Archived backup-hmx-q5-f32weight-wholecache-candidate exact host/DSP/catalog/main/matmulC+H/HMX/HVX/CMake/hostsource/hashmanifest. Host2226402B762C8E2233059CEEF1D686BE51731FBAD70B24BFA23A75F2C08A2CDD identical priorF32weighthost; DSP1A0DDCB6B770FE426FCE07D2F52BB64C0B9F0607F327C662E0A5976AC6C7AC58; catalog2DB5C2CF31331348FAF5633DF9803CEE80A54944B85306906EADAF9B677B90DD. Temporary per-op maincachepatch removed afterarchive.

Generalized existing matchedwrapper optionalCandidateDirectory/LabelPrefix/CaptureTokens while preserving originaldefaults; parsercheckpassed, retainsall3candidatehash/install/restoreguards. Live44256 shared-q5k-hmx-f32weight-prefill-npu Tokens1 retained layer5prefill. Wrapper verifiedall3manifesthashes and installedall3; finallyrestoresbaselinehostC9E4/DSPD2D/catalog afterterminal/no liveinference. Pollsamehandle, no restartontimeout/no competinginferencebuild. All8nesteddiagnosticflagsON, rootF32weightON/certtemporarilyempty; configure restore only aftercaptureterminal.

Next actual up/down occurrence2/T9 originalCPUQ8_K unchangedgate plusq8_k_fp16_math/output. Newweightdequant bypassesqf16 andconsumesF32effectiveD/M; notclean scale-onlycause, actualinputmaydifferfrompriorcapture. PriorCPUweightFP16ideal+output alreadyexceedsupgate slightly, no qualificationpromise. Full128 onlyafterlocallyvalid. Goal active/incomplete.

## F32 effective-weight HMX diagnostic implemented; matched build18865 live

Local SDK target hmx_hexagon_protos.h assembly-syntax audit lists HF accumulator stores before/after/retain and cvt.hf, no SF accumulator-store intrinsic found. This is header evidence only, not proof v81 hardware has no higherprecision route. SDK qfloat_tool.c explicitly describes VonNeumann rounding/implicitone fractionalfield; IEEE float emulation does not reproduce qf16 intermediate chain. Do not invent :after.sf assembly or claim supported F32store.

New OFF-default GGML_HEXAGON_HMX_Q5_K_WEIGHT_F32_REFERENCE requires existing matchedQ5weight1024flag. Replaced exactly16Q5-only dequant calls with helper: enabled branch reads appended32F32D/M at768/896, maps each unpacked64int16 vector's duplicatedrow indexj/2, scalarF32multiply/add thenfinalHF. Disabled branch preserves exactoriginalqf16 multiply/add/HFconvert expression. Q4/Q6 dequantfunctions untouched. Stackarrays ownedpercall, normaltile768 unaffecteddefault. This isolates weight-scale representation plus qf16dequant bypass, NOT clean scale-onlyfactor and NOT performancekernel. Need actual weight/dequant oracle beforeclaims.

build-hmx-q5-weight-reference.cmd builds matchedhost and nestedall8referenceflagsON, noinstall. Restore/Q5weight/HMXQ8Khelpers explicitly resetnewHMXweightflagOFF topreventcachecontamination. Hostbuild output2226402B762C8E2233059CEEF1D686BE51731FBAD70B24BFA23A75F2C08A2CDD matchesprior1024weightcandidatehostexactly; installedhost remainsC9E4 hashverified. AllCobjectscompiled, session18865 live linking, process1049232 CPU24.015625 verified. Pollsamehandle, no restarttimeout. BaselineDSPinstalledD2D/no liveinference.

Final poll18865 terminal0/allobjects+DSP linked. No livebuild/inference, noinstall. Next separate whole-cache maincontrol build/sign ->archive MATCHED2226DLL/DSP/catalog/source/hashmanifest ->remove temporarymainpatch ->new matchedwrappercapture layer5prefillT9 with3-filehashverify/install/restore. ExistingDSP-onlywrapperunsafe1024format. Generalize run-q5-weight-control.ps1 optionalCandidateDirectory/LabelPrefix/CaptureTokens or dedicatedequivalent; preservedefaults. Need compare q8_k_fp16_math/output andoriginalq8_k unchanged1e-7; idealFP16outputalreadycanfailup so no passpromise. Fullgoalactive/incomplete.

## Same-input HMX weight/output precision isolation measured

Extended existing frozen oracle with explicit q5_k_hmx_repacked_fp16 / _output contracts: reconstruct Q5 quantintegers and FP16 effectiveD/M prefix, compute dequant weights->FP16, Q8_Kactivation->FP16, float64dot, optionally finalFP16output. Qf16 intermediates and hardware accumulation not emulated; these references diagnose representation, not redefine CPU gate. Added q8_k_fp16_output_math ideal originalGGUFweightFP16 plus outputrounding, and cpu_contract_nmse reports on sameinput. Pythoncompile passed; existingCPU layer5up/T9 originalQ8_K still1.87267724e-14/exit0, validating unchanged original reference.

Actual9A66 up/downT9 repackedweightideal vs runtime: unroundedoutput3.26255159e-7/2.04726165e-7, roundedoutput3.655886e-7/2.73273342e-7 bothnativeexit1. Sameinput repackedideal vs originalCPUcontract5.64974897e-7/3.23633203e-7 (rounded6.05167958e-7/3.95940123e-7). Thus effectiveweight storage helps explain residual but does not fully model actual kernel. Ideal originalweightFP16/inputFP16 vs CPU7.60071426e-8/4.43354105e-8; adding finalFP16output1.22052614e-7/up and8.24191408e-8/down. Separate frozenCPUup repeats this pattern: idealoperands7.15077127e-8 vs withoutput1.14839559e-7. This tested ideal outputrounding exceeds gate forup; not a universal lower bound or proof other errors cannot cancel. Do notpromise F32weights alonepass.

Explicit originalCPUgate on9A66 up nativeexit1; wrong32block negative ondown nativeexit1. BaselineD2D/all7cacheflagsOFF reverified; no livejob. Source HMXQ5 dequant uses Q6_Vqf16_vmpy/add thenHFconvert, consuming oldF16D/M prefix evenwhenF32appendedweights enabled. HMX_STORE_AFTER_F16 is mxmem(...):after.hf=acc, outputthenexpandedtoF32. Need inspect SDKsupported higherprecision accumulator export/correction, not merely renameFP16bufferF32. Next controlled matchedhost/DSP F32effectiveweight HMXdequant prototype can isolate weight loss, preserving CPU gate and actual localoutput checks; qf16/hardware accumulator residual requires itsown probe. Goal active/incomplete.

## HMX Q8_K prefill capture closes block-size gap but fails fixed local gate

Session86308 terminal0/CLI0, baseline D2D/catalog restored/hash verified. Actual layer5Q5_K occurrence2/T9: up q8_k_fp16_math8.02458706e-7/max0.00130720665 nativeexit1, originalQ8_K8.19285623e-7; down fp16_math4.57709997e-7/max0.000396051724 nativeexit1, originalQ8_K4.76661285e-7. Fixed1e-7 unchanged; do not launch128pair as locallyvalidated. Wrong32block reference up0.000297467578/down7.83709568e-5, dense0.00030096855/6.92270078e-5, much farther but no CPUequivalence. Explicit originalQ8_K/wrongQ8_0 exit tests still pending (values printed by common oracle already showfail).

New complete256block contract improves closeness to CPUquantizedinput vs dense, but residual not explained by originalweightdequantF16 ideal alone. Next isolate actual HMX Q5 effectiveweightF16 D/M repack/dequant precision plus accumulator/output packing on same capturedinput, ratherthanrelax gate or mix upstreaminputs. Keep localrepresentationoracle distinct from originalCPU promotion gate. Guarded/signcandidate9A66 archived exactsources, nohost/GGUFchanges. Restore helper executed aftercapture; verify returned configterminal and flags before nextbuild. Goal active/incomplete; no liveinference.

## Signed complete-block HMX Q8_K capture86308 live

Guarded rebuild25507 terminal0; separate whole-cache control build/sign72217 terminal0/no signing errors. DSP9A66FD8CFC6E1240B257AA1FC152CDCA19F5B5C3DC1F4A16566D21EE8E05E583 archived exactso/catalog/main/matmulC+H/hmx/hvx/CMake in backup-hmx-q8k-wholecache-candidate. Temporary per-op cachecontrol source removed after archive. Baseline host C9E4 hash verified unchanged, standardQ5tile768, noGGUFmutation.

Live session86308 shared-q5k-hmx-q8k-prefill-npu layer5/Tokens1 retained frozen prefill capture; latest poll live. Wrapper finally restores D2D/catalog after terminal/no liveinference. Poll samehandle, do not restart on timeout; no competing inference/build. After terminal: oracle input attn_post_norm-5/output ffn_up-5/weight blk.5.ffn_up_shexp.weight occurrence2/T9 --reference q8_k_fp16_math, then originalq8_k explicitgate and q8_0 wrong32block negative. Verify actual weight name against prior Q5 capture if needed. Down input ffn_swiglu-5/output ffn_shexp-5/weight blk.5.ffn_down_shexp.weight same occurrence2/T9. EffectiveweightF16scale/offset rounding remains independent; no relaxed1e-7/no claimed kernel-runtime bytes/fullmodel qualification. Current flags HMXQ8_0/HMXQ8_K/Q8F32/Q6ON, allQ5flagsOFF. Restore helper resets all7flags after archive/capture, configureonly. Fullgoal active/incomplete.

## Complete-block K-quant HMX activation diagnostic implemented; build31448 live

OFF-default GGML_HEXAGON_HMX_Q8_K_ACT_REFERENCE added. Four plain quantized HMX callsites gate by Q4_K/Q5_K/Q6_K; F32/F16 and Q4_0/Q8_0 unaffected by new flag. Existing HMXQ8_0 diagnostic remains separately gated. New diagnostic stages complete row pairs via DMA0 into owned existing scratch, waits DMA completion, computes signed-first-absmax256/inv=-127/signedmax/CPU nearest_int magic-add/min127, dequantizes with F32 d before normal HMX F16 packing. Original input never modified. All padded M rows to32 written as zeros, no padded input DMA/read. No gatherK support claimed: current Qwen expertsQ4_0 use existing32block path, Kshared plain paths covered.

Layout audit min scratch=act_threads*4*k*sizeof(float), new staging needs2*k*sizeof(float), no VTCM growth/hostformat changes. Current assertions require k%256==0/valid==k and scratchcapacity; add explicit release-mode guard before capture because NDEBUG removes assertions. New build-hmx-q8k-reference.cmd explicitly sets HMXQ8_0/HMXQ8_K/Q8F32/Q6ON and allQ5diagnosticsOFF; restore helper includes newflagOFF. Q5-weight helper should also explicitly disable newflag before reuse to avoid cache contamination.

Session31448 terminal0/allCobjects+link successful. Added release guard aborting incompleteK blocks/mismatchedvalidK/shortstride/insufficientrowpair scratch; Q5weight helper explicitlynewflagOFF. Rebuild session25507 live, matmul recompiled; latest session poll confirmslive. Build only, no install/signing/runtime claim. Poll25507 samehandle, no restart on timeout. Baseline installed unchanged. After terminal separate known whole-cache maincontrol -> build/sign/archive exactsources/catalog/DSP -> remove temporarypatch -> actual layer5 T9 frozen Q5_K q8_k_fp16_math and originalQ8_K math gates. K effectiveweightF16 rounding remains distinct, unchanged1e-7 gate. Then full128 repeat/CPUtokens only if local evidence valid. Sequential activation staging is diagnostic, not performance candidate. Goal active/incomplete.

## HMX Q8_0 full pair repeats exactly but fails CPU token60

Session64398 terminal0, bothCLI0. wholecache-hmx-q8-mm2-repeat128-a/b strict --require-identical exit0: all128 complete selected tokens/logits bit-identical. Matching chatCPU128 --require-tokens native exit1, firstdiff60 CPU1061/NPU353,61 shared-prefix rows; CPUmargin0.141433716/NPUopposite0.0226650238. Step0 NMSE0.0127697778/max0.895251065. Diagnostic14.32/14.19tok/s, not paired untraced performance qualification. Reject promotion: local HMX Q8_0 projection improvement is real, but Q6-only control remains longer CPU matching prefix93. Per-op whole-cache control still required for these repeats; no coherence fix claimed.

Baseline D2D/catalog restored/hash verified by wrapper; normal root/nested flags restored OFF/certificate via restore helper terminal0, configure only. Host unchanged C9E4 hash reverified. No live inference/build. Source routing ggml_hexagon_matmul_is_hmx_eligible requires M > HTP_MM_HMX_MIN_NROWS; matmul-ops.h constant is4. Thus source selects HVX for T4, explaining the near-exact Q8_0 result; do not label this HMX evidence. T9 is the relevant HMX diagnostic. All six nested diagnostic cache flags verifiedOFF.

Next contracts: K256 HMX prefill needs signed first-absmax activation quantization with full256 blocks preserved across column splitting. Current column worker splits k/32 among threads, so naively replacing32 quant with256 is unsafe. Full-row scratch path can quantize complete blocks; column path must use256-aligned partition and revalidate scratch capacity/VTCM accounting. Gathered expert32vector loader cannot compute a shared256 scale alone. Keep F32 router/F16 weights unaffected and test actual Q5 layer5 T9 plus negatives before full decode. Separately combine locally validated F32 Q5 weight decode with HMX Q8_0 prefill as a matched host/DSP factorial candidate; no automatic compatibility or token pass assumption. Full goal active/incomplete.

## HMX Q8_0 frozen capture passes; full decode gates next

Build67092/signing completed0, candidate DSP588D8F4FD094CCECB25CB5D31DB6FB2811870EB79C3D641E124B60400E6A6E34 archived with exact sources/catalog in backup-hmx-q8-wholecache-candidate. Temporary whole-cache source patch removed after archive. Capture14128 actual-matrix-block0-prefill-hmxq8 terminal0/CLI0; baseline D2D restored and hash verified. Host C9E4 unchanged; standard weight format, no model mutation.

Actual block0 Q4_0 qkv occurrence2/T9: q8_0_fp16_math NMSE4.0975371e-8/max0.0148152709 passes unchanged1e-7; original q8_0 NMSE4.88149871e-8/max0.0133991132 also passes, dense2.59892576e-6 fails. Occurrence3/T4: fp16_math1.82038511e-8/max0.00539808581, original q8_0 1.25008026e-14/max5.55126462e-6, both pass; dense2.28961228e-6 fails. Different chunk behavior does not prove identical kernel routing; retain provenance and inspect routing before attributing T4 exactness to HMX. Own captured inputs, not cross-backend same-input causal proof. Full128 repeatbits/CPU selected-token gates remain required. K256 HMX activation coverage remains absent.

Live session64398 wholecache-hmx-q8-mm2-repeat128-a/b using DSP-only wrapper and archived compatible candidate; process1100880 verified warming up, latest session poll remains live. Poll same handle, do not restart on timeout. Restores D2D/catalog finally. No competing inference/build. After terminal compare strict repeatbits and matching wholecache-chat-cpu-reference128 selected tokens, report native exits. Goal remains active/incomplete. Also made Q5-weight build helper explicitly disable HMX diagnostic to prevent cached-option contamination; no build/install during this inference.

## HMX Q8_0 activation diagnostic implemented across plain and expert-gather paths

New OFF-default GGML_HEXAGON_HMX_Q8_0_ACT_REFERENCE emulates CPUQ8_0 activationquant/dequantbeforeHMXF16packing onlyforQ4_0/Q8_0weights. PlainDMArow/columnchunks modifyonlyownedF32scratchafterDMAcompletion; cols32alignedmatchquantblock. Gatheredexpertflat/nonflatpaths useprivate32float vectorcopies, nevermodifyaliasedoriginalinputs. Allfourplainquantizedweightcallers routebyweighttype; groupedF16weights explicitlyfalse, F32routerweightsfalse. Kweights remainunchanged:256blockreference notyetimplemented, nofullCPUcontractclaim. Currentcandidate combinesHMXQ8_0/F32HVXQ8/Q6activationreference; Q5activation/sum/weightflagsOFF, standard768Q5layoutcompatiblebaselinehostC9E4. NoGGUFchanges.

Build53814/plainterminal0,68903/allgatherterminal0. Headergathereditchecked16loads/2signatures, stategating/callerpropagationcompiled. Separatewholecachecontrol temporarymainpatchaddedafterfullbuild; build-cacheline-candidate live newhandle (nextpoll), mayinstall/signDSPwhencomplete, do notlaunchinferencewithbuildactive. Next archiveexactDSP/catalog/main/matmul/hmx/hvx/Cmake thenremovetemporarymainpatch, frozenfirstlayerQ4prefillcapture withQ8_0FP16math andoriginalCPUQ8_0gate. Thenfull128repeat/CPUtokens, stillKprefillcontractpending. InstalledhostbaselineC9E4/DSPD2Dverifiedbeforebuild; no liveinference. RestoreconfighelpernowalsoresetsHMXflagOFFafterarchive. Fullgoalactive/incomplete.

## Weight-scale fullpair repeats but fails15; HMX activation emulation feasibility measured

58138 terminal0/bothCLI0, matchedhost2226402B/DSPA9F5894A, full128repeatbits/selectedstrictcomparator exit0. MatchingCPUgate exit1 firstdiff15 CPU4602/NPU264,16sharedprefixrows, CPUmargin0.00659942627/NPUopposite0.0267410278. Step0NMSE0.00486678944/max0.706166118 (betterdistance butshortprefix). Diagnostic14.11/14.16tok/s notperformancepromotion, HTP19142MiB/Host1MiB. BaselinehostC9E4/DSPD2D/catalog restored/hashverified. AllbuilddiagnosticflagsOFF/certnormal; no livebuild/inference. Q6-only intended-placement diagnostic93 remainslongestprefix, nonefullcorrect. Matchedhostrebuildconfoundsisolatedfullmodelweight-onlycausality; actuallocalweightmathpreviouslyverified.

Added q8_0_fp16_math/q8_k_fp16_math references: CPUquantizedactivationdequant ->FP16, originalGGUFdequantweight ->FP16, float64dot. OnsamecapturedCPUinputs Q4firstlayerqkvoccurrence2/T9 idealQ8_0FP16math7.46310225e-9NMSE/max0.00295967516 passesfixed1e-7 (dense2.49906236e-6fails). Q5layer5shared-upoccurrence2/T9 idealQ8_KFP16math7.1506022e-8/max0.000398792967 passes (dense0.000287106688fails). These omitHMXaccumulator/outputpacking andeffectiveF16D/Mweightrepackloss, so onlyrepresentationfeasibility, notruntimequalification. No thresholdchange.

Next HMX activation reference must be weight-type-aware (Q4/Q8 CPUQ8_0 vsKCPUQ8_K, notquantizeF32routerweights), operateafterDMAonprivateF32scratch beforeFP16packing, coverrow/colchunk/gatheredpaths. Columnchunkscurrently32alignedcanhandleQ8_0 butQ8_K256needsfullalignedblockcoverage/halo; do notquantize256frompartial32chunk. activation_transfer_params has5plaincallers, expertgather separate2paths; auditallbeforechanging. ExistingHMXdequant Q5 stilloldF16prefix andcoreoutputFP16, separateprecisionlimitsremain. Goalactive/incomplete; CPUtoken/performance/MTP/corpusnotqualified.

Live matchedfullpair58138 confirmedbysessionpoll+llama-cli process; firstarm wholecache-q5k-f32weight-mm2-repeat128-a. Pollsamehandle, no competingbuild/inference. MatchedcandidateDLL2226402B/DSPA9F5894A+catalog installedtemporarily, baselineC9E4/D2D restoredtogetherfinallyafterbothterminal. Strict128repeatbits/matchingCPU128token gatespending; normalbuildflagsOFF/certrestored, no livebuild.

## Matched weight-scale candidate passes original CPU-local gates; fullpair launched

32611 terminal0/CLI0, matchedhost2226402B/DSPA9F5894A withQ5effectiveF32D/M+F16activationD/F32sum andwholecache. Layer5firstdecodeoccurrence4/T1 up originalCPUQ8_K3.67594073e-8NMSE/max0.0002432939 passes1e-7, down5.8261387e-8/max4.6972883e-5passes. Intendedq5_k_f32_weight_q8_kstorageoracle2.97831386e-14/up and1.33250698e-14/down; oldF16weight/F32sumreference4.9937619e-7/5.29243945e-7 fails. ActualQ5weightrepresentationgap closed to localgate, not bitidentity/fullmodelqualification; own capturedinputs used. HTPmodel19052MiB (+15MiB asderived), HTP19145MiB/Host1MiB reported; HMXprefillmathstilloldF16prefix. NoGGUFchanges.

WrapperrestoredandhashverifiedbaselinehostC9E4+DSPD2D+catalog together. Configure-only restore-q5-weight-build-config.cmd exit0: rootweightflagOFF/originalcertificatepath restored, nestedall5diagnosticflagsOFF. Buildartifactsstillcandidateuntilnextrebuild, installedbaselinebeforefullpair. Newfullunretained128 matchedpair launched via run-q5-weight-control.ps1 -Mode repeat, fresh wholecache-q5k-f32weight-mm2-repeat128-a/b, restorebaselineall3finally. Next pollnewhandle, strictrepeatbits+matchingCPU128tokens; no competingbuild/inference. Fullgoalactive/incomplete.

## Matched F32 weight-scale capture32611live; both-package restore guarded

Previous turn matchedhost/DSPbuild completed. Build71611 terminal0/signing0 addsseparatewholecachecontrol; archived backup-q5-f32weight-wholecache-candidate exacthost/DSPsources/CMakefiles/so/catalog/DLL+hashes.json. Matchedhost2226402B762C8E2233059CEEF1D686BE51731FBAD70B24BFA23A75F2C08A2CDD, DSPA9F5894A0459345CC180588C2AB664E41D8EC20D6B42072E0FEF24DDEAF6B565, catalog806B55BD869ED3151488882352A8197B5ECCFA998BA201926E816608447A943C. Temporarywholecachemainpatchremovedafterarchive.

New run-q5-weight-control.ps1 validatesarchive3hashes, installs/validatesmatchedDLL+DSP+catalog beforelaunch, restoresall3baselinefileswithhashchecksfinallyafterterminal. Live32611 shared-q5k-f32weight-decode-npu layer5Tokens2 capture, processverifiedlive. NeveruseDSP-onlyrunner forthis1024tileformat. Next poll32611, actualup/down occurrence4/T1 compareq5_k_f32_weight_q8_kstorageoracle andoriginalCPUQ8_Klocalgate. Full128pair -Mode repeatonlyafterlocalvalidation. RootdiagnosticweightflagON/certempty stilltemporary; nested5flagsON; restoreconfigOFF+originalcertafterarchive/runterminal usingconfigureonly, noallarchitecturebuild. No competingbuild/inference, noGGUFchanges; fullgoalactive/incomplete.

## Matched Q5_K F32-effective-weight prototype builds; not installed

RevalidatedbaselineC9E4host/D2DDSP/no livejobs. OFF-default GGML_HEXAGON_Q5_K_WEIGHT_F32_REFERENCE inhost+DSP CMake expandsQ5tile768->1024, appendsD/M F32 vectorsat768/896 preservingoldF16prefix/highbitplane. Hostrepackstoresoriginal GGUF d*scale/-dmin*minimum F32; HVX32x1/32x2 consumesappendedvectors withF16activationD/F32sum. Fourkernelinstantiations/twoHVXstrides use sharedtilemacro, normal/alignedsizesagree. Q4writesunchanged; placementofnewstoresverifiedinsideQ5repackbeforebuild. HMXusesalignedstridebutoldF16prefixmathunchanged, so noHMXprecisionclaim. NoGGUFmodification.

Build92487 terminal0/hostDLL+v81DSP compiledandlinked; no install/signature/hardwaregatesyet. Existing rootbuild ggml-base dependencyrebuiltinbuilddir only, notcopiedtopkg. Baseline installedhostC9E4D8ADE6884DA62DF7A597DCEAC6582AC036236C06134231A405BAF8E6399E andD2DDSP hashverified, backup-q5-weight-prechange includeshost/DSP/catalog/rootCMakeCache. New build-q5-weight-reference.cmd builds onlyhostbackend+nestedv81, noinstall. Rootbuildcache diagnosticweightflagON andHTP_CERTempty temporarily toavoidallarchitecture signingdependency (originalcertpath backedup); nested5diagnosticflagsON. Do notfullbuildallarchitectures withthistemporaryconfiguration. Restore rootflagOFF/certoriginal aftercandidatearchive; normalinstalledruntimeunchanged.

New q5_k_f32_weight_q8_k oracle emulatesF32weightD/M+F16activationD+F32sum, threshold1e-7 unchanged. Next separatewholecachecontrol build/sign/archive MATCHEDhostDLL+DSP/catalog/exactsources, then dedicatedwrapper mustswap/verify/restoreall3files together beforehardwarecapture. ExistingDSP-onlyrunnerunsafe fornewformat; neverinstallonlynewDSPoronlynewhost. No currentlivejob. Fullgoalactive/incomplete; CPUtokens/performance/MTPstillunqualified.

## F32 offset sums repeat128 exactly but CPUtoken gate fails60

83709 terminal0/bothCLI0, wholecache-q5k-f32sum-mm2-repeat128-a/b strictrepeatbits exit0 all128complete logits/selectedexact. MatchingCPUgate exit1 firstdiff60 CPU1061/NPU353,61sharedprefixrows; CPUmargin0.141433716/NPUopposite0.114160538. Step0NMSE0.00891350656/max0.847534359. Q5/F16sumfirstdiff15 ->F32sum60 isusefulisolatedimprovement, stillworseQ6-only93/notqualified. Diagnostic14.03/14.20tok/s notpaireduntracedwin. BaselineD2D/catalogrestored/hashverified, no liveinference/build. AllprecisioncandidateoptionsremainOFFdefault; nestedbuildcache4ON butinstalledbaseline.

Next weightprecision audit completed: Q5 tile768bytes fullyuses512nibbles+128interleavedF16D/M+128highbitplane. F32D/Mneeds2additional128bytevectors ->1024tile, preservingoldprefixforHMX. Mustpairhostrepack/DSPstridechanges: hardcoded768 in4 matmul-ops.c kernelinstantiations and2HVXdotkernels, headernormal/alignedtilemacros; HMXdequantusesstatealignedtile_size andoldF16prefix, so appendedF32 alonewillnotfixHMXprefillmath. Derivedexisting60Q5sharedweights62,914,560params implyadditional15MiBrepackedstorage(+33%Q5tiletraffic), excludingpadding/workspaces; verifybeforeclaimperformance. RPCIDLhwinfo/starthasnoformatcapabilityhandshake, nevermixnewhostformatwithbaselineDSP. Prototype mustarchive+restorematchedhostDLL/DSP/catalogset andverifyhashpair beforelaunch. No GGUFrewrite needed. Goalactive/incomplete.

Live fullpair handle83709 confirmed bypoll+llama-cli process, firstarm wholecache-q5k-f32sum-mm2-repeat128-a. Pollsamehandle; no restartontimeout. CandidateB725AA temporarilyinstalled, finallyD2Drestoreafterbothterminal. Required nextgates strict128repeatbits andmatchingCPU128tokens. No competingjob/build.

## F32-sum runtime verified; full128 pair launched

83679 terminal0/CLI0/D2D restored/hashverified. B725AA F32sum capture layer5 firstdecode occurrence4/T1: up intendedF32sum storageoracle2.60046469e-14NMSE/max2.82132532e-7, down1.02513295e-14/max4.20841388e-8; originalCPUQ8_K up5.27571371e-7/down6.44089737e-7 stillfails1e-7. F16sum alternativealso passeslocalgate4.6664833e-8/up and2.08421336e-8/down, so use residuals+implementationevidence, notpassalone forformatclassification. Own inputs differ acrosscaptures, no sameinputfullmodelcausality. New full unretained pair launched wholecache-q5k-f32sum-mm2-repeat128-a/b withB725AAcandidate; wrapperrestoresD2Dfinally, no competingbuild/inference. Next pollnewhandle, strict128repeatbits/matchingCPUselectedtokens; originalgatesunchanged. Goalactive/incomplete.

## Signed F32-sum control capture83679live

Build48496 terminal0/signing0, B725AA0845B7621D9DCBDD07BEDCDBA3E9B331941885F405A3AA98F70224DD85 signedcandidate archived exactsource/so/catalog backup-q5k-f32sum-wholecache-candidate. Adds onlyQ5activationoffsetsumF32 topriorQ5+Q6/F32Q8 reference, separateknownwholecachecontrol; activationD/weightD/MremainF16. Temporarywholecache mainpatchremovedafterarchive. Live83679 shared-q5k-f32sum-decode-npu layer5/Tokens2, processverifiedlive, wrapperrestoresD2Dafterterminal. No competingbuild/inference; pollsamehandle, oracleoccurrence4/T1up/down.

Reference discriminability control: priorF16sumcapture67931 upalso passesF32sumoracle1e-7 (NMSE4.66678307e-8/max0.000216411958), vsits intendedF16sumoracle2.53306955e-14. Thus fixedpass/failalone doesnotidentify sumformat; compareactualresiduals/runtimeimplementation, do nottightengateafterseeingdata orclaimnegativecontrolrejected. OriginalCPUtokenqualificationunchanged. Fullgoalactive/incomplete.

## Q5_K F32 offset-sum isolation implemented; build19806 completed

Previous turn completed8precisionfactor sweep and rejectedQ5+Q6tokenregression. Revalidatedcurrenttree/no liveinference/baselineD2D. New OFF-default GGML_HEXAGON_Q5_K_SUM_F32_REFERENCE dependsQ5activationreference; writes onlyQ5_K activation sum vector as32F32lanes inexisting128-byte slot, consumes it throughQ5-only32x1/32x2 helpers, exactF16weightoffset->F32 thenmultiply. Activation scale remainsF16, effectiveweightD/M unchangedF16, nohostrepack/model/ABI changes. This isolates sum rounding ratherthancombineweight/scale fixes. Allobjectscompiled, session19806live linking aslastpoll; noinstallation, nohardwaremath/tokenclaims. InstalledD2Dhashverified. Nestedflags4ON; mainwholecachepatchabsent.

Final poll19806 terminal0/linkcompleted; no livebuild or inference. Explicit q5_k_tiled_q8_k_f32_sum oracle added with unchanged1e-7 gate, usesF16weightD/M+F16activationD+F32sum. Pythoncompilepasses. Next build/sign/archive separatewholecachecontrol candidate, capture layer5Tokens2freshlabel and compare originalCPUcontract plusnewstorageoracle; strictfull128repeat/CPUtoken gatesneedediflocallyvalid. Weightprecisionfloorstillunfixed, do notpromiseactivation-sumchangeclosesCPUgap. Goalactive/incomplete.

## Q5_K prototype rejected for CPU-token regression; precision factors isolated offline

84214 terminal0/bothCLI0/D2D restored. wholecache-q5k-q6k-mm2-repeat128-a/b strict repeatbits exit0/all128complete logits/selectedexact. MatchingCPUtoken gate exit1 firstdiff15 CPU4602/NPU264,16sharedprefixrows, CPUmargin0.00659942627/NPUopposite0.045583725. PreviousQ6-onlyfirstdiff93; newQ5reference is accuracyregression, notpromotion despitelocaltiledmathpasses. Step0NMSE0.0100185885/max0.892374039 (Q6only0.0171299684), anotherexample numericaldistance !=greedytokenqualification. Diagnostic14.12/14.06tok/s, HTP19127MiB/Host1MiB, notpaireduninstrumentedperformancewin.

New opt-in --q5-precision-sweep evaluates8 independent F16/F32 combinations on identical capturedCPUinput/Q8_Kintegers, originalGGUFweightbits. CPUup/down occurrence4/T1: allF32 4.26010899e-14/2.17717314e-14NMSE; weightD/Mhalfonly4.92813982e-7/5.27653671e-7 failsboth; activationDhalfonly2.54219168e-8/5.15472743e-8 passesboth; sumhalfonly2.07721302e-7 failsup/7.23918467e-8 passesdown; allhalf7.75026904e-7/6.65129773e-7 failsboth. This is same-input offline precision evidence, not modelcausality orruntimeperfactor proof. Fixed1e-7 retained; originalCPUtoken gateunchanged. Next precision experiment should distinguish F32activation sum/scales (DSP-only existingtile capacity) from effectiveweightD/M precision (hostrepack/DSPformat agreement needed). Do not expectactivationonlyfix tocloseweightprecisionfloor. Keep Q5flag OFFdefault/unpromoted; archivedQ6-only bestdiagnosticprefix93, stillnotfullcorrectdecode. No livejob/build, installedbaselineD2D, nestedcandidateflagsON/ON/ON, sourcewholecachepatchabsent. Fullgoalactive/incomplete.

## Q5_K deployed math verified, storage precision still fails CPU; fullpair84214live

67931 terminal0/CLI0/D2D restored, shared-q5k-reference-decode-npu occurrence4/T1 layer5 actualweights. Up explicit tiled Q5_K/Q8_K oracle passes2.53306955e-14NMSE/max2.86148861e-7; originalCPU Q8_K fails5.53934304e-7/max0.00108476099. Down tiledoracle passes8.34684425e-15/max1.06974767e-8; originalQ8_K fails1.61946852e-6/max0.000456264282. Thus prototype implements intended storage math, but packedFP16weightscale/min andactivationoffsetsum contract differs from originalCPU, not threshold relaxation or fullcorrectness. Models unchanged. New full unretained pair84214 live, wholecache-q5k-q6k-mm2-repeat128-a/b, signed6FF0937candidate. Firstarm launched, pollsamehandle and strictrepeatbits/matchingCPU128gate aftercompletion. Wrapper restoresD2D finally. No competingbuild/inference. NestedflagsON/ON/ON, sourcewholecachepatchremoved; no productionpromotion/fullgoalcompletion.

## Q5_K activation prototype signed; decode capture live67931

Previous turn actual Q5_K decode contract evidence justified new OFF-default GGML_HEXAGON_Q5_K_ACT_REFERENCE. Requires Q6_K+F32Q8 diagnostic controls; switches only Q5_K row/block workers to signed256scale/int8 math. Existing1280-byte tiledgroups keep FP16scale and FP16(scale_F32*signed_int8_sum32) for offset correction. Q4 and other offset types unchanged, no modelweights/ABI changes. Builds54525/44502 terminal0, signing0. Signed6FF0937FDE1E530FA9BEDBB5600BA22B5BF55FC95035F87E1E49F30C5F3932DF exactsources/so/catalog archived backup-q5k-q6k-wholecache-candidate; main temporarywholecache patch removed afterarchive. Nestedflags ON/ON/ON, installedcandidate whilelive67931 capture shared-q5k-reference-decode-npu Tokens2/layer5, skipCPU reuses matching priorCPUcapture. Wrapper restores D2D afterterminal. No competingbuild/inference; poll67931.

New explicit q5_k_tiled_q8_k oracle accounts for FP16 effectiveweight D/M, activationD andsum32, using actual GGUF Q5_Kbits. On priorCPU upT1 sameinput/int8 gives7.75026904e-7NMSE/max0.00118807303, fails fixed1e-7 (originalCPUQ8_K3.80178643e-14). Thus activationmatchingalone is not sufficient for originalCPUlocalmath: packing/offsetprecision can matter. Neworacle is a storage-contract diagnostic, not replacing CPUtoken gate. Pythoncompilepasses; runtimeoracle andfull128gates pending. Goalactive/incomplete.

## Actual Q5_K single-token decode projections measured

87082 terminal0/bothCLI0/D2D restored/hashverified, shared-q5k-math-decode-{cpu,npu} Tokens2, first ordinary decode occurrence4/T1. Up actual blk.5.ffn_up_shexp.weight Q5_K: CPU Q8_K NMSE3.80178643e-14/max3.7150928e-7; FP16-scale emulation1.41239921e-8 passes. NPU Q8_0 closest9.04594154e-6/max0.00399925084 but fails1e-7; dense6.03603857e-5/Q8_K0.000233863046 fail. Down actual blk.5.ffn_down_shexp.weight Q5_K, ffn_swiglu-5 ->ffn_shexp-5: CPU Q8_K1.74026544e-14/max1.07308474e-8 passes, FP16scale2.04310522e-8; NPU Q8_01.36427688e-5/max0.000678129113 fails, dense3.0730796e-5/Q8_K8.76209763e-5 fail. Own captured input per backend, not same-input bisection. These are complete512/2048 outputrows localprojections, not head/fullmodel qualification.

Next implement separate opt-in Q5_K 256-element reference, preserving existing tiled1280-byte groups: each32group needs both FP16 scale and FP16(scale_F32 * signed_int8_sum32) for offset correction. Audit effective packed weight scales/offsets: Q8_0 oracle uses original GGUF dequant, not proof of exact Q8_1 offset runtime rounding. Existing Q6_K reference/source unchanged arithmetically; only indentation and unused-parameter suppression cleaned while no build ran. No live inference/build; nestedflagsON/ON but installedbaselineD2D. Fullgoalactive/incomplete, no promotion.

## Confirmed layer5 Q5_K prefill: CPU Q8_K exact, NPU closest dense but local gate fails

21620 terminal0/bothCLI0/D2D restored. shared-q5k-math-prefill-{cpu,npu}, confirmed blk.5.ffn_up_shexp.weight Q5_K512x2048, attn_post_norm-5 ->ffn_up-5 occurrence2/T9. CPU Q8_K oracle NMSE1.87267724e-14/max3.24172455e-7 passes; FP16-scale emulation3.65846444e-8/max0.000367014235 also passes. CPUdense0.000287106688/Q8_00.000276057781 fail. NPU/mm2 dense closest8.11425801e-7/max0.0015161175 but fails unchanged1e-7 gate, Q8_05.96679247e-5/Q8_K0.000297911741 fail. Do not call this a passing dense reference or evidence for the HVX offset quantizer: multi-token prefill can use HMX. Next capture single-token decode projection (Tokens2, new labels) to exercise relevant GEMV/offset path and compare before implementing Q5_K activation replacement. Earlier layer0 selection corrected/retained asQ8control, notQ5evidence. No live inference after21620, no modelchanges, fullgoal active/incomplete.

Live corrected layer5 shared-Q5_K capture session21620, CPU first/NPU next serialized; first CPU process confirmed live. Fresh labels shared-q5k-math-prefill-{cpu,npu}; poll21620 before math/build/another inference. Wrapper restores D2D after terminal. Current nested diagnostic flags ON/ON, main whole-cache source patch absent; archived243C used only NPU arm. Next actual blk.5.ffn_up_shexp.weight Q5_K oracle using attn_post_norm-5 -> ffn_up-5, occurrence2/T9 after capture verified. No fullgoal completion claim.

## Shared projection metadata corrected: layer0 Q8_0 control, layer5 Q5_K next

11229 terminal0/bothCLI0/D2D restored. Prior shared-math-prefill-{cpu,npu} captured layer0, but actualGGUF metadata proves its up/gate/down shared weights are Q8_0, not Q5_K. Do not count this capture as Q5_K evidence. All shared weights: Q8_0 63, Q5_K 60, F32 40, BF16 1; layer5 up/gate/down are confirmed Q5_K (up/gate2048x512, down512x2048). Existing layer0 capture is useful control: actual shared-up occurrence2/T9 CPU matches Q8_0 NMSE1.85794092e-14/max6.36764526e-7, dense2.9394859e-5. NPU/mm2 matches dense7.49492825e-8/max0.00142260568, Q8_0 fails2.85641926e-5. Matches earlier first-layer Q4 prefill pattern, not full-model causality. Runner now accepts explicit layer and fresh prefix, defaults layer5/shared-q5k-math-prefill. Next actual Q5_K captures use those fresh labels; strict shape/type/oracle gates unchanged. Goal active/incomplete.

Live shared-expert frozen capture session11229, CPU first then NPU serialized, run-q5k-frozen-control.ps1. First CPU process verified live after launch. Labels shared-math-prefill-{cpu,npu}, chat Tokens1, retained attn_post_norm-0/ffn_up-0/ffn_gate-0/ffn_swiglu-0/ffn_shexp-0. Wrapper temporarily installs archived243C candidate only for NPU and restores D2D finally. Poll same handle; do not restart on timeout. No competing build/inference. Next oracle actual blk.0 shared weights and each capture's own input; validate type/shape/occurrence before math. Full goal remains active.

## Q6_K full pair repeats all128 bits, CPU gate extends to93 but still fails

65884 terminal0, both CLI0. wholecache-q6k-mm2-repeat128-a/b strict repeat comparator exit0, all128 complete logits and selected tokens identical. Matching chat CPU gate exit1: first selected mismatch93, CPU60239/NPU3812,94 shared-prefix rows. CPU margin0.0516681671, NPU opposite margin0.246034622; step0 NMSE0.0171299684/max1.02961874. Previous mm2 F32-Q8-only control first failed60, so head contract change extends this workload's agreement but is not full correctness. Diagnostic rates14.45/14.49tok/s, HTP19127MiB/Host1MiB, no paired uninstrumented performance claim. Baseline D2D/catalog restored/hashverified, no live inference at pair end. Next frozen shared-expert Q5_K projection captures: its offset Q8_1 quantizer still uses FP16 intermediates, untouched by F32_Q8 control. Verify actual math before further replacement. Goal active/incomplete.

## Q6_K runtime head gate passed; full repeat pair next

Signed candidate243C6F450E05C126679E5258D9A20D73245A00CB3187615E2A1375A1D1F7A392 combines the opt-in Q6_K activation contract with separate whole-cache repeat control and existing F32 Q8 reference. Exact source/binary/catalog archived backup-q6k-wholecache-candidate; temporary whole-cache main patch removed after build. Build52386 terminal0/signing0. Head capture59007 terminal0/CLI0, baselineD2D restored/hashverified. Actual output.weight Q6_K, occurrence3/T1, same eight rows: Q8_K FP16-scale oracle passes NMSE1.56846704e-9/max0.000643871641; CPU-style F32-scale2.76802276e-8/max0.00325390497; oldQ8_0 fails1.80682729e-5/max0.0802623889. Local runtime contract now matches intended emulation, not proof of full-vocabulary accuracy or CPU token identity. HTP19127MiB/Host1MiB/no CPU_REPACK reported. Next run-q6k-repeat-control.ps1 launches serialized unretained128 mm2 a/b, restores baseline finally; strict bits and matching CPU tokens both still required. Goal active/incomplete.

## Opt-in Q6_K activation prototype compiles; runtime gates pending

Implemented GGML_HEXAGON_Q6_K_ACT_REFERENCE (OFF by default, requires Q8_F32_REFERENCE). Q6_K row and block-parallel activation workers use signed-first-max 256-element scaling and scalar nearest_int magic-add, stored in the existing Q8 tiled layout with FP16 scales. Each 128-element worker reads its aligned 256-element pair and writes only its owned tile. Q4 and offset Q5_K retain the existing F32-reference control math; no GGUF changes or ABI changes. v81 build43340 terminal0; all objects/link completed. Installed DSP still D2D, hash verified; no candidate installation or runtime success yet. Nested build options currently ON/ON for the candidate, unlike installed baseline. Next add the separate known whole-cache repeat control to a diagnostic build, archive exact source/binary/catalog, then captured-head gate before full128 repeat/CPU-token gates. Goal remains active and incomplete.

## Q8_K tiled-scale feasibility check completed

Added an explicit q8_k_f16_scale local oracle, preserving the fixed 1e-7 gate. Same eight head rows and occurrence3/T1: CPU capture passes, NMSE 1.84103435e-8, max_abs 0.00189209663, native exit0. Existing CPU Q8_K F32 oracle remains 4.6172844e-15. NPU capture rejects the new reference as expected, NMSE 3.55743478e-5, max_abs 0.0795360577, native exit1; existing Q8_0 remains its matching reference. This supports a tiled FP16-scale prototype but does not establish runtime byte agreement, full-vocabulary accuracy, or selected-token identity. No kernel or model weights changed, no inference/build live, baseline DSP retained.

Dispatch audit: quantization has both row and 128-element block-parallel workers. A 256-element reference must read the complete aligned pair even when a worker owns only one 128-element output tile; do not write a neighboring worker's tile. Q5_K offset handling uses the separate Q8_1 path, so a Q6_K-only first prototype is narrower than a blanket K-type replacement. Preserve Q4 activation math and keep whole-cache control separate. Goal active, not qualified or complete.

## Frozen sampled head distinguishes CPU Q8_K from NPU Q8_0 activation contracts

Prior turn progress: complete precision-control gates and stronger CPU prefixes.26214 terminal0/CPUandNPUCLI0 head-math-prefill-{cpu,npu}, samechat Tokens1, retained result_norm/result_output, NPU23E6/mm2/wholecache/F32reference; finallyD2D/catalogrestored/hashverified/no liveinference. Lastprompt head selectsoneoutputvector: occurrence3/T1, not4; mistakenT4 selection rejected beforemath, corrected. Oracle extended with explicit sampledweightrows and scalarCPUQ8_K math (signedmax256,nearest_intmagic-add,F32scale); byte/runtime contract remainsunproven but actualoutput agreement measured.

output.weight Q6_K eight sampled vocabulary rows9419,0,11,4602,264,60239,3812,99731. CPUvsQ8_KNMSE4.6172844e-15/max1.5040772e-6; dense1.5941992e-5/max0.0594688336, Q8_02.22190851e-5/max0.0714856156. NPUvsQ8_0NMSE1.49551345e-9/max0.000616932325 (passes1e-7), dense6.46726899e-6/max0.0332822538, Q8_K3.58102057e-5/max0.0806196689. Testscover8rows only, not fullvocabulary or token60context. Capturedinputs differ acrossCPU/NPU; eachoracle uses its own actualinput, so contractclassification notsameinputcausal bisection.

Next selectiveK-quant CPUstyle256element activation quantization experiment using existing tiled8bit storage, preservingQ4Q8_0math and GGUFweights. TiledF16scale can still differfromCPUQ8_KF32scale, mustquantify ratherthanpromise exactness. Inspect taskdispatch/packing and add optindiagnostic only after relevantcode audit. Wholecache remainsseparate costlyrepeatcontrol; MTP/corpus/fastcorrectNPU notqualified. Goalactive/incomplete.

## HMX-enabled pair completed:128 repeatbits pass, CPUtoken60 fails

4770 terminal0/bothCLI0, wholecache-q8f32-mm2-repeat128-a/b; repeatbits strictcomparison exit0,128complete rows/selectedtokens exact. Bothplacements HTP19127MiB/Host1MiB/noCPU_REPACK. Diagnostic14.32/14.23tok/s not optimized/paired untracedrate. CPUgate stillfirstdiff60 asbelow. BaselineD2D/catalog restored/hashverified, no liveinference; sourcebaseline/referenceoptionOFF. Next precision investigation includes CPUQ8_K vs NPUQ8_0 activation math for K-quant/head weights and HMXprefill Q8-reference emulation, plus narrowingwholecache scope once correctness distinguished. Keep full performance/MTP/corpus goalsactive; neither23E6 norwholecache promoted.

## F32 reference extends CPU token agreement to93; still not qualified

Previous turn progress: realinput quantizer gap closed and fullpairlaunched.81678 terminal0/bothCLI0 wholecache-q8f32-mm1-repeat128-a/b signed23E6. Strictrepeatbits exit0 all128complete rows/selectedexact. MatchingCPU128token gate exit1 firstdiff93 CPU60239/NPU3812,94commonprefixrows; CPUmargin0.0516681671/NPUopposite0.155864716. FirstlogitsNMSE0.00947833913/max1.05203199. PriorMM1 referenceOFF firstCPUdiff15; newreferenceON matchesCPU through92, usefulaccuracy improvement but not thresholdrelaxation/promotion. HybridCPU_REPACK16560MiB, diagnostic3.64/3.34tok/s no pairedspeedclaim. D2D/catalog restored/hashverified.

Live4770 wholecache-q8f32-mm2-repeat128-a/b uses samearchived23E6 with HMX enabled, no retained probes. FirstarmCLI0/128complete rows, CPUtoken gate exit1 firstdiff60 CPU1061/NPU353,61commonprefixrows, margins0.141433716/0.0886878967. FirstlogitsNMSE0.00610490297/max0.828184128. Diagnostic14.32tok/s, HTP19127MiB/Host1MiB reported/noCPU_REPACK, muchmore representativeNPUplacement thanmm1; not yet a pairedspeedclaim or fullcorrectdecode. Secondarm stilllive; repeatbits pending. Finally restoresD2D/catalog when terminal/no liveprocess; sourcebaseline, F32optionOFF. Goal: recover intended NPUplacement and assess token agreement, not substitute slow CPUfallback forrequestedNPU goal. No competing NPU/build job. Fullgoalactive/incomplete.

## F32 activation reference closes frozen first-layer quantization gap; full128 gates live

Prior turn progress: actualweight math and mm1hybrid rejection. Build27778 terminal0 signed23E6A676D5935C8F8F4F81FA9EA31301FE5451A384F029DB49010A1987FF6A40 combines existingGGML_HEXAGON_Q8_F32_REFERENCE=ON with temporarywholecache. Exactmain/so/catalog archived backup-q8f32-wholecache-candidate. Buildhelper fixed stale nonexistent rebuild-gdn-state.cmd call to existing DSP-only build-cacheline-candidate.cmd; no hostlibs installed. ConfigresetOFF beforetest, temporarymainsourcepatchremoved; buildartifactsstilldiagnostic butinstalled baseline restored after firstcapture.

88080 terminal0/CLI0 actual-matrix-block0-prefill-q8f32, sameMM1probe/wholecache: realQ4 qkv projection prompt9 vs Q8_0 referenceNMSE1.32239585e-14/max8.33359081e-6, passes1e-7, versus oldMM1 6.42224989e-7. Dense2.53194076e-6 as expected forQ8quantizedactivations. Actualinputs/opweights captured, isolates quantizer difference more directly than syntheticgates. Does not establish modeltokenidentity, speed or pureNPU placement; CPU_REPACK16560MiB remains.

Live81678 wholecache-q8f32-mm1-repeat128-a/b, no retained/snapshot probes; archived23E6 temporarilyinstalled, finallyrestoresD2D/catalog afterterminal/no liveinference. Next poll81678, compare complete128repeatbits and matchingCPU128selectedtoken gate. No competingjob/build, goalactive/incomplete.

## HVX-preferred combined control repeats128 bits but still fails CPUtoken15

72468 terminal0, frozen projection and both128CLI0; E956 restored toD2D/catalog/hashverified. Labels wholecache-allhvx-repeat128-a/b are historical shorthand: mm_select1 is HVX->CPU fallback, not proven allNPU/allHVX. Runtime reports CPU_REPACK16560MiB and Host~800MiB, so substantial hybrid allocation; cannot claim pureNPU throughput. Diagnostic4.91/2.76tok/s additionally exposed to concurrent host-side oracle computation, not a paired rate benchmark. Strict128bits exit0; matchingCPUtoken gate exit1 at15 CPU4602/NPU264, CPUmargin0.00659942627/NPUopposite0.0619049072;16commonprefixrows. FirstlogitsNMSE0.00992727325/max1.15881896. Notcorrectness/performance promotion.

Frozen mm1 prompt9 Q4projection relative referenceQ8_0 NMSE6.42224989e-7/max0.0275074409 (fails1e-7); relativedense2.15424129e-6/max0.0314404177. Thus HVX-preferred projection closerQ8reference but not reference-identical. CPUprompt4 correctoccurrence3 Q8referenceNMSE1.2086363e-14/max6.77694334e-6, dense2.31103131e-6. An attemptedoccurrence2/T4 pairing was rejected by shapeguard before math; corrected to3, not counted as numerical failure.

No live inference/build. Next test existing F32-reference activation quantizer against this actual frozen projection under wholecache (prior synthetic/raw rejection not proof for this combinedcase), or independently check first-layer output/head math. Wholecache repeatability does not solve tokenaccuracy. Fullgoalactive/incomplete.

## Frozen real Q4 projection distinguishes CPU Q8 activation math from NPU dense-like path

Previous turn progress: completed matchingCPU128 gate failure.47424 terminal0/CPUandNPUCLI0 actual-matrix-block0-prefill-{cpu,npu},Tokens1 samechat/settings, retained embedding/attn_norm0/qkv0/z0/l_out0, NPUwholecacheE956/mm_select2. Nonwarmup prompt9/4 scope10records: embeddings exact, normsNMSE1.6755037e-14/1.36628724e-14, qkvCPU/NPU NMSE2.56288407e-6/5.77898258e-7, z3.0822015e-6/6.77592267e-7, l_out0.000215329883/0.000351563412. No strict crossbackend bits gate; CPU/NPU local numericdifference is expected to be quantified, not erased.

New check_frozen_matmul_math.py reads explicit occurrence/shape/finiteF32 input/output and actual GGUFweight, compares float64 dot with dense or referenceQ8_0 activation quantization (GGUF Python implementation, not runtimebytecapture). Real blk0.attn_qkv.weight Q4_0 [8192,2048], prompt9 CPUQ8referenceNMSE1.29274106e-14/max9.52580012e-6 vsdense2.49906236e-6/max0.0447002245. NPUdenseNMSE5.13357858e-8/max0.0147698165 vsQ8reference2.59709846e-6/max0.0473847004. Both pass their declared local1e-7 oracle, but different activationmath; no selectedtoken threshold relaxed, no proof that one wholemodelstream is correct.

Live72468 first captures actual-matrix-block0-prefill-hvx withmm_select1/wholecache, then wholecache-allhvx-repeat128-a/b logits-only/samechat128. Goal: see if matching quantized CPU arithmetic resolves separate tokenfailure under passingrepeatcontrol. E956 temporarilyinstalled, sourcebaseline; finally restoresD2D/catalog after no liveprocess. No competingjob/build. Fullgoalactive/incomplete.

## CPU reference complete: whole-cache repeatability pass is not token correctness

7498 terminal0/CLI0 wholecache-chat-cpu-reference128,128 complete CPUrows. Strict selectedtokencomparison against wholecache128-a exit1, firstdiff15 CPU4602/NPU264,16commonprefixrows; same metrics below now verified with complete aligned logs. CPUdiagnostic0.53tok/s, prompt11720.43ms13tokens/eval240053.20ms128; not a CPUperformancebaseline. No liveinference. D2D installed. Next frozen actual early-matrix CPU/NPU input/output captures and independent quantized-weight oracle; fullgoalactive/incomplete.

## Matching chat CPU prefix disproves whole-cache token qualification; reference still live

Previous turn progress: wholecache128repeatpass and cheaper range-control rejection.7498 remains live, CPUreference advancing. Non-strict partial comparison at21 recordedCPUrows vs128NPUrows (not complete128gate) already supplies same-input counterexample: firstselected15 CPU4602/NPU264, scores CPU24.4032612/24.3966618 margin0.00659942627 vs NPU24.440197/24.4690914 margin0.0288944244;16 commonprefixrows, step0 NMSE0.0369790405/max2.03249264. Same stage CLI/chat/model/settings; this is NOT the excluded rawCPU comparison. Wholecache improves repeatability but does not establish selected-token correctness. Keep reference running to completion; do not kill/restart or claim terminal/full128results. BaselineD2D installed, no competing inference/build.

Read-only audit: init_ctx allocates unified ctx/main-stack/work-queue/DMA-ring/HMX-ring block via memalign4096 (ctx->footprint), separate512KiB DDR scratch via memalign128. DMA descriptors reside in unified block, not activation tensor ranges. Existing batch-start wholecache flush plus footprint prefetch; perop wholecache covers nonactivation allocations whereas rejected activation range control does not. Hardware coherence/return status not established by allocation audit; no proven descriptor bug. Next narrower controls can separate metadata-footprint/DDR scratch, weight-source coverage, API scope and tracker reset. Accuracy next: independent frozen real matrix input/weight/output oracle, separating activation quantization from repeatability, after CPU reference terminal. No new deployed mutation this turn; goalactive/incomplete.

## Narrower activation-range control rejected; matching CPU128 reference running

96723 terminal0/bothCLI0. perop-activation-unretained128-a/b strict comparison exit1,128 complete rows each; firstbits0 NMSE0.00569143334/max1.09128118, firstselected15(4602/264), margins0.0494861603/0.148881912. Only16rows common prefix. Diagnostic25.30/25.03tok/s not promotion. Rangeflush(nonweight/nonfence) plus syncht does not reproduce wholecache's128bit pass. Different coverage/API/dirty-tracker reset/timing/layout prevent a clean weightcache rootcause inference. Candidate3324 archived; source temporarypatchremoved and installedD2D/catalog restored.

Fresh wholecache-chat-cpu-reference128 live session7498, same run-mtp-stage CLI/chat/c512/Hello13-token prompt/model/seed/temp settings, device none. Compare selectedtokens (notbits) against E956 wholecache pair after complete128rows. Historical rawCPU reference excluded. No competing NPU/build job while CPU reference runs. MTP/fullcorpus/raw vschat/fastercachecontrol still unqualified; goalactive/incomplete.

Runner robustness: run-mtp-stage now guards existing label artifacts and live inference before writing; run-decode clears new snapshot environment to prevent accidental probe inheritance. Both PowerShell parsers pass. This does not change deployed inference code.

## Whole-cache unretained128 strict bits pass; narrower activation control live

Previous turn progress: matched32 cache diagnostic. Session23800 terminal0/bothCLI0 perop-wholecache-unretained128-a/b, signedE956, hostC9/llama3597 unchanged, normalGGUF/Hello chat/FA1/GDN1/routerHVX/replay1/fusion1/UB128, logits only/no retained or snapshot probes. Strict comparison exit0 all128 complete rows bits and selectedtokens identical. Diagnostic17.09/16.74tok/s, not paired benchmark or production qualification. D2D/catalog restored/hashverified before nextbuild.

Next narrower candidate build35644 terminal0 signed3324E7D94DECEA3AFB67D0DBA85F2DE035A2ED644683A6EA1C80ABD1B62BA63B: flush every nonweight/nonfence nonempty activation src/dst range with hex_l2flush regardless dirty tracking, followed by syncht(memory clobber), existing tracking preserved. Does not flush entire cache or weight ranges. Archived source/so/catalog backup-perop-activation-candidate; prechange main backup-perop-activation-prechange. Temporarysource patch removed after archive/build. Live96723 perop-activation-unretained128-a/b sameflags/no retained tensors; finallyrestoresD2D/catalog only after no liveinference. Do not build or run competing inference until96723 terminal.

Historical alignment-cpu128 reference cannot qualify current chat runs: run-decode.ps1 uses llama-completion -no-cnv/c256/rawHello1token; run-mtp-stage uses llama-cli single-turn reasoningoff/c512/chatHello13tokens. Attempted comparison differedstep0, excluded as mismatched workload, not model accuracyfailure. Need fresh same-runner CPU reference for crossbackend qualification; prior raw CPU39failure belongs only that raw workload. Fullgoalactive/incomplete.

## Whole-cache control passed matched32 bit gate; unretained128 control live

89083 terminal0/bothCLI0, perop-wholecache-boundaries32-a/b strictlogits exit0 all32bits/selected identical. All363nonwarmup earlyboundaries exact/finite. Matched baseline11897 firstbitdiff6. Instrumented16.43/16.50 vs22.47/22.49tok/s not an optimization. D2D restored/hash verified; main source matches prechange ignoring line endings (hash differs only formatting). Cache API adds syncht/timing/code-layout confounds; passing32probe not full-model qualification.

Reinstalled archivedsignedE956 for live session23800 perop-wholecache-unretained128-a/b run (no retained/snapshot probes, logits only, same diagnosticflags). Orchestratorfinally restoresD2D/catalog after terminal/no liveinference; sourcealreadybaseline. Do not start competing NPU/build jobs. Goalactive/incomplete. Next poll23800 and compare complete strict128rows, then design narrower cache/barrier/ownership controls if passing.

## Early block boundary pair complete; whole-cache per-op diagnostic running

Prior turn progress: independent float64 real-state oracle and layout-localization limits. early-block-boundaries-repeat32-a/b session11897 terminal0/bothCLI0, unchanged llama3597/hostC9/DSPD2D, FA1/GDN1/routerHVX/replay1/fusion1, snapshots disabled, retained embedding and l_out0..9. All363 nonwarmup records (33frames) finite/bit-identical. Logits strict comparison exit1:32 aligned selectedtokens allmatch, firstbits6 NMSE0.000118477632/max0.206396103; rawargmax remains same. Thus this retained layout observes instability after block9 despite identical early outputs, not universal model bisection.22.47/22.49tok/s instrumented diagnostic, not rate promotion.

Next diagnostic build67613 terminal0 signedDSP E9560F297EFB470BC70A61615BE6130F0E35CD6366FDE52A84F2C2D588FC5160 temporarily installed: unconditional QURT whole-data-cache flush/invalidate and dirty-range clear immediately before existing per-op input prep flush. Existing batch-boundary whole-cache calls unchanged. SDK computev81 qurt/qurt_memory.h states API adds syncht on v60+, so this control changes cache/barrier/timing, not a clean separation between them. No arithmetic changes. OriginalD2D/catalog/main backed up backup-perop-wholecache-prechange; exact diagnosticso/catalog/main archived backup-perop-wholecache-candidate. Temporary source lines removed already, main restored to prior logic; diagnostic installed only for pending test.

Session89083 live perop-wholecache-boundaries32-a/b matches11897 probe/settings. finally restores prechange D2D/catalog only after no live inference. Do not build or start another NPU job until89083 is terminal. Current candidate is a deliberately expensive diagnostic, not optimization/promotion. No result yet.

## Deeper snapshot changes earliest boundary; independent real-state recurrence matches model

Previous turn was progress: actual-state captures, replay gates and negative controls. This turn44284 terminal0/bothCLI0 recurrent-snapshot18-repeat32-a/b, unchanged llama3597/hostC9/DSPD2D, same diagnostic FA1/GDN1/routerHVX/replay1/fusion1. Snapshot18 plus l_out9/10/14/17/18/19 and attn_output18. Logits strictexit1,32 complete rows, bits differ0 NMSE0.00507050919/max0.874156475; firstselected15(4602/264), margins0.0151443481/0.0878658295. Fourteen records/frame. Prompt scope skip28/max28:1equal27different, earliest retained l_out9 prompt9 already differs, whereas frozen initialstate18 prompt9 remains exact. Thus snapshot10 prior downstream-localization is layout-specific, not transferable. Changed layout reproduces upstream variation; do not blame isolated GDN from different inputs.

Added check_frozen_gdn_math.py float64 recurrence over immutable actual inputs, checks recorded model attention AND next graph's frozen carried state, modulo16 Q/K head mapping, threshold1e-7 with finite/framing/shape/missing/duplicate gates. Layer10 prompt4 botharms attentionNMSE6.66431474e-14/state1.22510262e-13; firstdecode1 attention1.97280864e-14/state1.43833466e-14. Wrong-head-map h//2 negative fails attentionNMSE3.01318991/state0.874687496, expectedexit1. Layer18 botharms prompt9/prompt4/decode1 sixchecks allpass: attentionNMSE3.19896254e-14..4.71086639e-13, state4.91694959e-14..2.19796319e-12. Actual state carry and arithmetic match locally despite inter-run input/output variation. Next-state assumption valid only plain consecutive graphs without reset/rollback; not MTP qualification or bit-exact oracle.

No live inference/build. No new installed artifact this turn. Next move toward frozen actual matrix-input/reuse oracle or earlier block-stage ownership localization, not another identical isolated GDN campaign. Fullmodel repeatability/MTP correctness and performance remain incomplete; goal active.

## Immutable layer10 inputs captured and actual-state GDN gate passes; model instability remains

Prior turn was progress (completed ABBA and actual stage profiles). This turn: opt-in LLAMA_RECURRENT_SNAPSHOT_LAYER / runner SnapshotLayer duplicates conv window and six GDN inputs, marks independent copies as outputs and expands before consumers, no consumer substitution. DUP not allocator-inplace eligible. Build39961 terminal0 installed only llama.dll SHA359708332567BB4BDB2A0CCD1170261CF3BB6062ED9847E174A4A42AA0088FF8; old llama.dll981E backed up backup-recurrent-snapshot-prechange. HostC9/DSPD2D unchanged.

recurrent-snapshot10-repeat32-a/b session85782 terminal0/bothCLI0, FA1/GDN1/routerHVX/replay1/fusion1, snapshots layer10 plus l_out9/attn_output10/l_out10. Each350 records/35 graphframes, finite. Logits strict exit1:32 complete rows, first bits0 NMSE0.00149631852/max0.457410336, firstselected19(1438/12622), margins0.0342884064/0.173408508. All210 records covering frame2 prompt9 through frame22/step19 are bit-identical, including actual frozen state and all five activations, conv window and layer9/10 outputs. Laterframes not compared because inputs diverge. This layout localizes observed instability downstream of retained layer10 output, NOT a universal production bisection result. Copies/retentions change layout and prevent some layer10 fusions;17.31/16.37tok/s diagnostic only.

State-aware GDNFX002 prompt4 fixture exported from graph3 in both arms,2229272bytes, identical SHAB6127A737CFB5FD9C480253CCE5EB53369CD3968D38A7095287923936273C6D0. Actualstate523776nonzero/maxabs1.05241084 (not zero/seeded proxy). Frozen gate is4D[1,32,T,1], initial exporter wrong3D expectation rejected before creating output, corrected. Explicit wrong-token-count exporter negative rejected/no output.

Existing fixture loader extended for002 actualstate;001 preserved. Captured-state tests exclude synthetic seeded arm; chain loader rejects002 to avoid falsely treating independently captured states as chain carry. Build52092 terminal0/testexe only. gdn-actual-state-prompt4-replay1/replay0 each6/6 pass32 repeats (384 focused comparisons), including2 partial-prefix cases eachsetting. Wrongslot control4/6 passes/expectedbackendexit1, both partial cases fail state independently CPU/NPU, attentionstable. V1 regression12/12pass. No model/MTP qualification from focused gates.

No live inference/build. Next retain boundary between10 and19 or snapshot layer18 with identical-prefix guards; alternatively compare isolated actual-state output to frozen model output (a fresh independent oracle, not only CPU/NPU agreement). Default-disabled snapshot code remains installed, originalDLL backed up; full goal active/incomplete.

## Actual candidate stage profiles completed; logits remain non-repeatable

concat-dim0dma-stage16-a/b bothCLI0, second wrapper99202 terminal0; each15 complete1384-op one-token DSP profile batches. Batch means33.971/33.851ms, head2.969/2.968ms, SSM_CONV+WIN1.493/1.460ms summed attribution across30 layers. No standalone concat in steady decode; prefill/warmup concat source transposed outside candidate eligibility. Instrumented wall rates1.34/1.35tok/s unusable for rate promotion. Same installed hostC9/DSPD2D.

Strict logits comparison exit1:16 complete aligned rows, all selected tokens match, first bit difference0 NMSE0.000559164819/max0.300038815. This short instrumented pair does not override unretained128-token selected failure15. Comparator now matches ASCII selected-token markers in raw log bytes because unrelated verbose log bytes were not valid UTF-8; binary finite/layout and complete selected-row checks unchanged. First wrapper stale LASTEXITCODE falsely stopped after successful a; b separately executed using script success. No live inference after both arms. Next immutable actual pre-GDN/conv-state snapshot probe with paired allocation controls; full goal remains active and incomplete.

## Completed generic concat ABBA: local cost reduction, not full-model qualification

2026-09-27: concat-model-shape-abba-v2-{a1,b1,b2,a2} completed all four correctness/performance arms, wrapper exit0. Each correctness arm12/12; each performance arm12 nonempty timing cases. A=DSP27 baseline, B=DSPD2D candidate, identical host/test executable throughout v2. Candidate restored and installed SHA256 D2D98EDEA1BDFD8652EE0766C4B7707AE1209E335164E6115478091C304B68CD verified; no live inference process after completion.

Packed v0 F32 concat [3,8192,1,1] plus T rows, dim0: T1 A98.31/98.31us, B79.22/77.87us (means98.31/78.545,20.10% reduction); T4 A181.87/174.70us, B61.30/60.40us (178.285/60.85,65.87%); T9 A288.24/296.01us, B64.86/66.21us (292.125/65.535,77.57%). All12 shapes improved in both candidate arms versus both baseline arms. Row-strided source variants also measured, but not production transposed qkv physical strides.

These are existing perf-harness amortized batched us/run (OPBATCH3072/replay1/fusion0/profile0/default workers), not single-token submission latency, whole recurrent-stage latency, or end-to-end speedup. Calibrated run counts differ. First attempted non-v2 perf arm returned zero cases because eval/perf factories are separate; rejected by runner and excluded. Full-model selected-token/repeat failure at15 remains open; MTP not qualified. Next: profile actual candidate model-stage placement/cost without promoting this microbench to model throughput.

## Hardware campaign recovered; focused DMA concat fix does not fix full-model repeatability

78828 terminal0, both model CLI0, normal Q4_0 GGUF/Hello128/FA1/GDN1/routerHVX/replay1/fusion1/UB128/headcap0, no retained tensors, logits trace. concat-dim0dma-model-repeat128-a/b strict compare exit1:128 complete rows each, first bit difference0 (NMSE0.00307839269/max0.902320385), selected tokens first differ15 (4602/264), margins0.12786293/0.121154785. Compared16 common-prefix rows; later rows not same inputs. Diagnostic27.17/27.06tok/s, not paired baseline/uninstrumented promotion. Thus focused generic-CONCAT/cache discrepancy resolved by candidate, but full-model instability remains. No full-model/MTP correctness or speedup claim.

No live inference/build. User-approved pending-process recovery used, all logs preserved; fresh replay0 focused18/18, default concat48/48 eachreplay, wrongtail0/18 eachreplay verified. Installed hostC9/DSPD2D remains experimental, baselineDSP27 backup available. Next paired stage latency/correctness assessment of dim0 DMA scope, then actual model state/input ownership investigation rather than more unchanged isolated-chain repetitions. Full requested goal remains incomplete; user explicitly resumed work despite previous blocked bookkeeping.

## User-approved recovery succeeded; candidate hardware checks resume

User explicitly approved terminating only pending test and one fresh session. Validated PID231784 executable/commandline, terminated it;10553 terminal1 wrapper, replay0 exit-1 externally interrupted before graph execution (not numeric failure). Logs/stacks preserved. Fresh conv-state-chain-dim0dma-approved-retry fusion0/replay0 terminal0 all18/18 pass32 repeats. Combined first replay1 and fresh replay0 total1152 focused comparisons, including384 bypass/no-convolution cases; no full-model qualification.

concat-dim0dma-default terminal0 replay1and0 each48/48 supported cases pass (96 total; unsupported types remain explicitly skipped). Candidate wrong-tail control terminal0 wrapper, expected backend exit1 both replay1and0 each0/18pass; rejects wrong semantics independently CPU/NPU. HostC9/DSPD2D unchanged. No reboot/driver reset, approved recovery consumed. Continue full-model repeat/token and stage latency gates; candidate remains experimental.

## Recovery decision pending; hardware campaign at impasse

03:49:30 revalidated10553/PID231784 live, CPU0, still remote_handle64_open since03:33:50. Not terminal; do not restart from timeout. Same unavailable-session/recovery-choice condition has persisted across three consecutive turns since user question. Safe source/descriptor mapping audit, non-invasive stack, WCT inspection and regression preflight exhausted; no meaningful hardware qualification can proceed without session return or user direction on recovery. Marking goal blocked (not complete), preserving full objective, live process and logs. Pending choice: keep waiting or explicitly terminate only pending test and attempt one fresh session; no reboot/driver reset. Never infer approval from automatic continuation.

Installed experimental DSPD2D98EDEA1BDFD8652EE0766C4B7707AE1209E335164E6115478091C304B68CD/hostC9, baseline DSP27 backup-concat-dim0-prechange. First replay1 arm18/18 pass; replay0 no graph execution/unqualified. Resume: first poll10553, inspect pending choice, then default CONCAT regression/negativecontrol/focused profiles/full-model repeat+CPU token/MTP gates. Do not label candidate a speedup or full-model fix.

10553/PID231784 re-polled live around03:46-03:47, still session-open since03:33:50. Nonblocking user question submitted: continue waiting or explicitly terminate only pending test and retry one fresh session (no reboot/driver reset). No answer/authorization yet; default continue waiting, no restart. Do not repeat question or infer approval from continued goal messages. Goal active/incomplete.

## Windows wait-chain inspection: blocked threads, no detected cycle

10553/process231784 remains live at session open. Added read-only inspect-wait-chain.ps1 scoped to campaign processes, using SDK wct.h ABI (280-byte nodes onARM64) and closing/freeing handles. Terminal0 inspection: main624300 and driver79576 each one WctThreadType/Blocked node, DETECTED_CYCLE0; no exposed dependency chain. This is not proof absence of a driver/DSP deadlock; WCT cannot traverse the relevant driver dependency here. No resets, termination or competing inference. Next re-poll10553; candidate still replay1-only qualified, full goal incomplete.

10553/process231784 still live at Qualcomm remote_handle64_open, re-polled this continuation. Offline independent address mapping audit:512 randomized dim0/dim1 cases/646620 bytes pass versus scalar tensor indexing; hardware/default regression still pending. No restart or competing inference. Goal incomplete.

## Non-invasive stack confirms wait inside Qualcomm session open

10553/process231784 still live. ARM64 cdb -pvr (documented non-invasive/nonsuspending) inspected stacks and detached, terminal0; dim0dma-replay0-init-stack-v2.log retained. Main thread: ntdll ZwWaitForSingleObject -> KERNELBASE WaitForSingleObjectEx -> libcdsprpc remote_handle64_invoke -> remote_handle64_open -> ggml_hexagon constructor. This pins current wait to session open, before graph execution; does not establish cause or exclude prior DSP state/cleanup interaction. Other driver thread waits on SRW synchronization; no private driver symbols, symbol nearest-export names not exact internal function identities. No reset/termination/competing inference. Initial debugger invocation rejected a switch before attaching; v2 succeeded.

Next poll10553; default CONCAT regression script ready, negative control and model/profile gates pending. Candidate first replay1 arm remains18/18 only; goal incomplete, no promotion.

## Standard concat regression runner prepared while FastRPC initialization remains live

10553 re-polled live, process231784 CPU0 still at session allocation (started03:33:50). No restart/termination/competing NPU work. Added run-concat-regression.ps1 for existing CONCAT suite, fusion0/replay1and0: hashes exe/host/DSP, clears opt-in fixtures, refuses existing logs and competing inference, demands exit0 and nonempty all-pass counts. Parser/preflight terminal0; actual invocation safely rejected live test before launching another process (terminal1 expected guard). This is runner validation, NOT concat regression execution/qualification.

Installed testexe F5DAFDDBEE102E64029903549A584616F4B0EA26EDF17DBC814A5F5F49EBA1FB, hostC9, DSPD2D98EDEA1BDFD8652EE0766C4B7707AE1209E335164E6115478091C304B68CD. Next poll10553; when terminal run regression prefix concat-dim0dma-default, then wrong-tail control and model/profile gates. Goal incomplete.

## Dim0 row-DMA concat candidate passes first focused arm; replayoff live

Continuation revalidated10553 and process231784 live, CPU0, still HTP/FastRPC session allocation before graph execution. Recent System-event read found no matching Qualcomm/DSP/WHEA/power error in inspected15-minute/30-event scope (not exhaustive). Observation delay is not terminal; no competing inference/build launched. Replay1 log terminal18/18 remains first-arm evidence only. Previous goal turn progress; this continuation verified wait plus candidate/source audit, not blocked or complete.

Build54715 terminal0, signed DSP candidate D2D98EDEA1BDFD8652EE0766C4B7707AE1209E335164E6115478091C304B68CD installed; hostC9 unchanged. Backup baseline DSP27/catalog/source in backup-concat-dim0-prechange. concat_rows_dma generalizes existing dim1 contiguous DMA helper to dim0 row-contiguous F32/F16/I32 sources/destination with matching outer dimensions, sane row strides, one device and existing descriptor-width/stride/height guards. Two strided DMA copies concatenate row segments; transposed and unsupported cases retain existing fallback. No arithmetic change.

conv-state-chain-dim0dma-fusion0 session10553: replay1 terminal0 all18/18 pass32 repeats, including previously failing generic direct persistent-cache raw/SILU. Replay0 confirmed live process231784, currently at HTP session allocation with no test output yet. Poll10553 before competing build/inference; do not restart solely on observation timeout. Candidate not promoted: replayoff, negativecontrol, existing CONCAT regression, paired stage profiling and full-model repeat/token gates remain required. Goal incomplete.

## Convolution-free data-movement control passes; simple copy/read explanation not established

Expanded WrongTail control fusion0/replay0 wrapper terminal0, expected backend exit1,0/18 pass; all bypass and convolution cases reject wrong tail independently on CPU/NPU at repeat0. New control's semantic gate validated.

Build52393 terminal0/testexe only. Added six bypass_conv1 controls to existing opt-in chain (18 total). Replaces SSM_CONV with CONT(transpose(first T elements of each T+3 window)), observing oldest window samples without arithmetic, while preserving concat/state-tail CPY/cache policy. Full14 prefix reference matches split9/4/1 delayed stream; state gate unchanged. Descriptor remains SSM_CONV for group selection, but bypass graphs contain no SSM_CONV/SILU; do not count as convolution-kernel comparisons.

conv-state-chain-bypass-fusion0/replay0 and fusion1/replay1 both terminal1,16/18 pass: all six bypass variants pass32 repeats, including generic direct persistent-cache reads. Only two non-bypass generic direct persistent cases fail NPU-only at finaltoken13 plus state. Thus copy/read reuse alone in this control does not reproduce failure; removing conv also changes DMA scheduling/lifetimes/allocation. Next inspect convolution-containing dependency/alias order and DMA/cache interactions, not assume a stale-read or arithmetic bug. No live jobs, no production change, goal incomplete.

## Immutable-read control suppresses generic persistent-cache failure

Expanded negative control conv-state-chain-snapshot-wrongtail fusion0/replay0: wrapper terminal0 with expected backend exit1,0/12 pass, all variants reject wrong tail on both CPU/NPU at repeat0. Snapshot option does not weaken semantic gate.

No live jobs. Build87448 terminal0/testexe only; reverted cpy source rebuilt in build tree, clearing rejected candidate there. Installed hostC9/DSP27 unchanged/hashverified. Opt-in conv matrix now12 cases: previous8 plus4 persistent-cache snapshot_read1 variants (raw/SILU x generic/transposed). After each cache commit, snapshot_read1 CPYs to a fresh immutable tensor before next chunk reads state; persistent backing still overwritten each stage.

conv-state-chain-snapshot-fusion0/replay0 and fusion1/replay1 each terminal1,10/12 pass. All immutable-read variants pass32 repeats; only generic persistent direct-read raw/SILU fail NPU-only at token13/state (repeat0/4 in fusion1,0/0 fusion0). CPU full/split remains identical. Additional copies change allocation/timing and cache transitions, so this is ownership-seam evidence, not root-cause proof or production workaround promotion.

Source descriptor audit: host add_tensor deduplicates same data+shape/strides and propagates flags from extras; sampled input/kernel/state descriptors flags0 (not incorrectly weight-exempt). Generic CONCAT uses scalar reads/writes; transposed path uses DMA/VTCM. Next independent minimal strided-write -> scalar-read reuse oracle without SSM_CONV, plus descriptor order/range audit of failing case. Full-model instability remains open.

## Convolution layout control narrows failure; pre-copy flush candidate rejected

No live jobs. Build81052 terminal0/testexe only adds transposed_input0/1: same logical chunk bytes repacked cont(transpose(chunk)) then transposed back to match production source layout. Eight cases. Baseline conv-state-chain-transposed-fusion1 replay1 and restored conv-state-chain-transposed-restored-off fusion0/replay0 both terminal1, each6/8 pass: all four transposed-input variants and two generic distinct-cache variants pass32 repeats; only generic persistent-cache raw/SILU fail NPU-only, first attention token13 plus finalstate. CPU full/split remains bitidentical. Layout materialization changes graph/timing too, so do not identify one implementation as root cause from this alone.

Hypothesis: scalar reads may populate clean cached lines outside dirty-range tracking before DMA reuse. Tested unconditional hex_l2flush of destination in cpy_dma_sametype_sameshape before DMA. Build84452 terminal0, DSP installed/signed temporarily; conv-state-chain-cpyflush-fusion1/replay1 still6/8 pass, same generic persistent failures. Candidate rejected; patch removed, git diff cpy-ops.c empty, installed DSP and catalog restored from backup-cpy-clean-read-prechange. DSP27 SHA reverified. Build output still contains rejected candidate until rebuilt; do not copy it to installed package. Source restored (line-ending-only hash difference from backup).

Next isolate generic CONCAT reads from strided CPY/cache reuse using immutable snapshots or an independent CPU-readable seam; validate graph alias/order and actual tensor descriptors. Do not promote a production fix or model throughput from this layout-specific test.

## New convolution-cache reuse discrepancy, not yet a proven production root cause

No live inference/build. 88099 terminal0: corrected GDN fusion0/replay1 and0 each32/32 pass; together corrected fusion1/fusion0 yield4096 chain comparisons. Builds88311 and69749 terminal0, testexe only; installed hostC9/DSP27 hashes reverified unchanged.

Added opt-in conv_state_chain in existing test-backend-ops plus run-conv-state-oracle.ps1. Frozen synthetic F32 input[17,8192,1], kernel[4,8192], initial3-token history; full14-token convolution compared bytewise per backend with9/4/1 chunks, strided tail CPY, distinct/persistent[3,8192,1] cache, raw/SILU,32 repeats. CPU/NPU component NMSE remains1e-7; repeats also bit-gated. This uses generic CONCAT (nontransposed chunk source), not yet production's transposed-input specialized path.

conv-state-chain-fusion1 replay1 terminal1: distinct variants pass32 repeats, persistent variants fail on NPU only. Detailed fusion1/replay0 and fusion0/replay1 both terminal1, distinct pass, persistent fail: first attention mismatch token13 (final1-token chunk), finalstate mismatch, CPU full/split remains bitidentical. Repeat failures begin at1/2 or3/5 depending run. Worker1/fusion0/replay1 still fails persistent cases (repeat1/0), so multiple workers are not necessary. Do not yet attribute to a kernel/cache bug; graph alias/ordering and tensor lifetimes need audit.

WrongTail negative control terminal0 wrapper with expected backend exit1 for both replay1/0: all4 cases fail independently CPU/NPU from token9 and state mismatch. Confirms semantic gate rejects wrong tail even when backends agree. Next: production-like transposed-source control, isolate strided CPY->scalar read of reused cache, inspect actual descriptors/alias ranges and ordering. Full-model correctness/performance/MTP remain incomplete.

Corrected slot0 fusion0 control session88099 is live (replay1 then0); poll88099 before competing inference/build. Goal incomplete.

## Corrected slot0 chain passes; production state indexing audit agrees

15321 terminal0: fusion1/replay1 and replay0 each32/32 pass, 2048 corrected-chain comparisons. This supersedes the prior live-job notice below. Correct forward carry now uses newest slot0; whole-chunk restore/replay and per-backend repeat bits pass with component CPU/NPU NMSE gates. Synthetic initial state remains explicitly declared; this is not full-model/MTP qualification.

Source audit: llama-graph.cpp build_rs selects rows using state_copy_main; llama-memory-recurrent.cpp s_copy returns idx*mem->size+src0 and resets pending rs_idx to0. Normal idx0 selects newest group; seq_rm requests the rollback-distance group. Together with delta-net-base.cpp's ordered snapshot copy, no equivalent written-1 forward-carry mistake was found in this production path. No production code changed.

Correctedslot0 chain session15321 confirmedlive; poll15321 before competing build/inference. Wrongslotnegative logs4times attention0/0 state1/1,8/12pass, so failure specifically state-slotselection. Goal incomplete.

## Correction: newest snapshot is slot0; prior K2 chain forward-carry claims withdrawn

CPU ops.cpp10955/11037 andDSP gated-delta-net-ops.c885 establish newest-firstslots:0final,1onetokenback. Harness incorrectly used written-1 forforwardcarry inK2. Fixedbothnormal/replayedcarry offsets to0; noDSP/modelchange. EarlierK2chainpasses demonstratedbackendagreement/consistencyonolder-stateinput, NOT correctforward-carry semantics. K1unaffected; wholechunkrestore checks remainedsamehistoryconsistencybut do notrescueincorrectK2forwardclaim. 38130terminal0oldfusion0/replay1and0 each32/32pass, nowqualifiedbythiscorrection.

Addedindependentpartial-snapshot oracle usingT4K2 vsseparateT3K1prefix withsharedinitialstate: slot1stateafter3 mustbitmatchprefixstateonbothbackends, alsofirst3attentionbytes. CapturedT4fixture normal30435terminal0/replay1and0 each12/12pass (8ordinary+4partialvariants)=768comparisons,256partialchecks. Wrongslotcontrol chooses0instead1: terminal1expected,4partialfail/8ordinarypass; logsstatebitmismatchbothbackends. Build40475terminal0/testexeonly. Runnerfaultflagdefaultcleared andnegativecontrolmetadata distinguisheswrongslot/wrongrestore.

Correctedslot0chainmatrix launched fusion1/replay1then0, recordhandle/poll before competing build/inference. Goal incomplete.

## Wrong-restore negativecontrol validates samebackend rollback oracle

85621terminal0 fusion1/replay1and0 each32/32pass=2048 strictchaincomparisons (16rollback+16normal variants32repeats). Build57162terminal0/testexeonly added opt-in GGML_TEST_GDN_WRONG_RESTORE/runner WrongRestoreNegativeControl: restorespost4state insteadpre4state, intentionallyincorrect. Runner clearsflag fornormalruns. 67577terminal1 EXPECTED: all16wrong_restore1rollbackvariants fail, all16unaffectedcontrols pass,16/32summary. Bothattentionandstate original-vs-replay bitchecks report backend1=1/backend2=1; negativegate catches wrongsemantics independentlyonCPU/NPU, notmerelycrossbackendagreement. HostC9/DSP27unchanged/reverified.

38130confirmedlive normalfusion0suite withfaultflagcleared, same32variants/replay1then0. Poll38130 before competing inference/build. Wholechunkrestore/syntheticinitialstate only; partialacceptance andconv/KV/MTPintegration remainunqualified. Goal incomplete.

## Whole-chunk snapshot restore/replay oracle added; replayoff live

85621 confirmed live replay0 after replay1 32/32pass. Poll85621 before competing build/inference. Build89249terminal0/testexeonly. Existing opt-in chain now rollback0/1: snapshotafter9chunk, process4chunk, restorepre4state, replayidentical4activations, thenprocess1decode. Snapshotdependencies explicitlyforce copies before backingoverwrite. Output retains original/replayed4attention plus immutablepoststatecopies/finalstate; samebackendoriginal-vs-replayed attention/state byteidentityrequired, plusCPU/NPU componentNMSE/repeatbits. Sixteenrollbackvariants plus16normalcontrols,32repeats. Replay1all32pass=1024chaincomparisons. Syntheticinitialstate; wholechunkrestore only, notpartialacceptance/convstate/KV/MTPqualification. Next negativecontrol intentionallyrestorewrongpoststate toproveoracle rejects it, thenfusionoffcontrol. Goal incomplete.

## Persistent-cache carried-state chain passes2048 strict comparisons

Existing opt-in chain now tests persistent_cache0/1: cachepool reused across9/4/1 CPYs vs distinctbuffers, same validlastsnapshot indexing. K2 finalT1 copiesonevalidslot, reads slot0 ratherthan staleunusedslot1. Initialsyntheticstate resetbetweenrepeats, caches initially randomized; notactualmodelallocation/rollbackcoverage. Sixteen variants(layout x state x K x cachepolicy)32repeats. Build3022terminal0/testexeonly, hostC9/DSP27 hashesunchanged/reverified.

24367fusion1 and40056fusion0 bothterminal0/replay1and0 each16/16pass. Total2048 strict CPU/NPU chaincomparisons, perchunkattention/finalstateNMSE plus perbackendrepeatbits. Runtime addressaudit confirms persistent CPY0/1/2 sharecachebacking (example000002CDB04C2000 forK1); firstchunkfusionactual underfusion1. Negativeevidencefor simpleHVXGDN+CPYpersistentreuse ontheseactivations/states, not fullmodelqualification. Next rollback/snapshotrestore oracle oractualprestatecapture remainsrequired. No live inference/build, goal incomplete.

30907 terminal0 fusion0/replay1and0 each8/8pass=512 comparisons. Togetherfusion1+fusion0 chains total1024 strict comparisons, repeatbits stable. No live inference/build. Goal incomplete.

## Carried-state actual-activation GDN chain passes fused/unfused controls

Opt-in GGML_TEST_GDN_CHAIN existingtest extension loads exact ordered9/4/1 fixtures, consumes precedingCPY output state(last written snapshot), concatenates eachattentionchunk+finalstate, gates eachchunk/state NMSE and samebackendrepeatbits. Eightcases contiguous/permuted x zero/seededinitialstate x chunkK1/2,32 repeats. Initialsyntheticstate reset onlybetween completechains, notbetweenchunks. Cachebuffers distinct perchunk; notinplace-modelallocator/MTProllback replay. Runner clears bothfixtureenvs outsideopt-in and printsallfixturehashes. Builds91259/13262terminal0/testexeonly, hostC9/DSP27 unchanged.

Fusion1 session26024terminal0 replay1/0 each8/8pass=512 chain CPU/NPU comparisons. Runtimeverified HVX and actualfirstchunk fusedGATED_DELTA_NET+CPY; laterCPYs separate, so do notclaimall3copiesfused. Fusion0 session30907 replay1 8/8pass, replay0 completion polled separately. No fullmodel promotion; next useful ownershipcontrol is in-place persistentcache reuse and snapshots/rollback, or actualprestate capture, notanotherunchangedisolatedkernel pass. Goal incomplete.

## Component gates expose masked HMX attention error; strict HVX1536 pass

Fixture-only oracle now returns worst NMSE of aggregate, attention, and each state slot (same1e-7limit), plus requires same-backend byte-identical output across frozen-input repeats. Default random cases unchanged. Build75905 terminal0/testexeonly; hostC9/DSP27 unchanged/reverified. T9strictHVX replay1/0 each8/8pass512; T4prompt andT1decode fixtures likewiseeach512pass (session65674terminal0). Total1536 strict CPU/NPU comparisons with repeatbits stable perbackend, not crossbackendbitidentity/fullmodelqualification.

Auto/GDN2 strictT9 replay1and0 independently4/8pass: both zero and seededK1/layouts fail; K2passes. Seeded attentionNMSE1.62984832e-7 while stateNMSE1.30817204e-8 explains prior aggregatepass dilution; zeroattention2.46165239e-7/state3.04427295e-7. Failedcases stop first accuracy failure, so their repeatbitidentity is unqualified. No limit relaxation or rootcauseproof for fullmodelGDN1 variation.

New fixture T4 records104..108 SHA B19E14F3C42F5ACD9E3D2185D64FD83CB7926B0916C5B7F67C0B5E828404F76B/132120bytes; T1firstdecode records135..139 SHA DBEE0916B4CBB4F9213278EB4761804BC9B749F615D1B38A2DA89AB27DA879FF/33048bytes. State remains explicitly synthetic. Next useful extension: chain captured9->4->1 chunks with CPU/NPU state carried through real GDN+CPY ownership transitions instead of resetting each isolatedcase. No live inference/build, goal incomplete.

Default GDN regression98182 terminal0: replay1/0 each11/11pass,704 comparisons. Fixturemode cleared for regression; default tests unchanged. No live inference/build. Goal incomplete.

## Frozen actual-activation GDN harness implemented: HVX passes512; auto zero-state K1 fails

Existing test-backend-ops now has optional GGML_TEST_GDN_FIXTURE GDNFX001 import, exact shape/length/finite checks, no extra Q/K normalization. Eight cases: contiguous/permuted x zero/seeded syntheticstate x K1/2,32 identical-input repeats each. Default cases unchanged; fixture opt-in only. Runner validates fixturepattern/path/op, prints SHA/statepolicy, clears fixtureenv for normal tests. Build67375 then5701 terminal0; only testexe installed, hostC9E4D8A/DSP27DC648C unchanged/reverified.

gdn-fixture-prompt9-hvx replay1/0 both8/8 pass =512 CPU/NPU comparisons, Fusion1/GDN1, frozen real layer10 prompt9activations SHA5DE02A32. Auto/GDN2 separately replay1/0 each6/8 pass, bothzero-stateK1/layouts fail NMSE~3.04e-7 >1e-7 stockgate. Exact failed components attention2.46165239e-7/state3.04427295e-7 bothlayouts. Runtime verified K1 hmx-chunked-scalar VTCM7690240; K2 hvx-recurrent-scalar VTCM1048576. SeededK1passesaggregate but not bitidentity/individualcomponent qualification. No CPU threshold relaxation, no fullmodelcauseproof: diagnostic fullmodel usesGDN1 yetstillvaries. Operator actual-state not replayed. No live inference/build; goal incomplete.

## Internal block11 stages stable under new probe; frozen captured GDN fixture exported

29702 terminal0/bothCLI0; strict logits/tensors1,all31requirednames131each/4061records. Warmup0..61,prompt9 62..92,prompt4 93..123,step1 124..154. Firstdifference89=L19(prompt9) NMSE4.05889625e-05/max0.166457951. Scope62/max27 all27 finite/identical0 includes embedding, earlyboundaries, all12 block11 internalcaptures, andL14. Block11 stages only differ aftertoken15 divergence in this pair; no local kernel fault reproduced under changed retention. L14 later differs beforeprefixend (7extra differing decodeframes), so not globally stable15boundary. Logitsbits0 NMSE0.0058011769/max0.947532177, selected15(264/4602), margins0.10185051/0.000352859497. Diagnostic21.28/21.26tok/s; no promotion. No live inference/build.

New export_gdn_fixture.py extracts first matching captured layer/T Q/K/V/loggate/beta F32 logical data, validates framing/shape/finiteness, refuses overwrite and missing tensors, prints recordindices/hashes. Layer10 prompt9 A/B fixtures both297240bytes SHA5DE02A320F09AB42BD29CB4607FFC44040B316A381D5EA356C82D2A1E3977569; actualrecords73..77, byte-identical five activationinputs. State explicitly NOT_CAPTURED, never pretend postgraph stateview is preinput snapshot. Next add opt-in fixture GDN cases to existing test-backend-ops using already-normalizedQ/K(no extra normalization), declared zero/seededsyntheticstate, contiguous/permutedlayouts,K1/2, CPUgate/repeats. This is actual-data operator reproduction, not fullmodel replay or correctness qualification. Goal incomplete.

Block11linearstages session29702 confirmed live in arm a; poll29702 before competing inference/build. Previous72409 terminal. Goal incomplete.

## Fixed19-layout third arm: B/C first variation in block11 attention residual

72409 terminal0/CLI0, both AC/BC strict logits/tensors1/all19names present131each. AC logitsbits0 NMSE0.004786738/max0.757945538, selected60(1061/353), margins0.00156021118/0.0250968933; firsttensor43=L7(prompt9), samepatternasAB. BC logitsbits0 NMSE0.00347167167/max0.708541393, selected15(264/4602), margins0.0288944244/0.151729584. BC firsttensor45=attn_residual-10(prompt9) NMSE2.24305207e-06/max0.00836181641; sameframe embedding+earlythroughL9 records38..44 strictfinite/identical0. Identical layout has distinct earliest varying boundaries among runs; not exclusive block8 defect. BC gives focused block11 attentionpath/storedstate candidate with identical captured current input and prior observed boundary history, not uniqueopproof. Diagnostic23.24tok/s, no promotion.

Next preserve19/add block11 normalized-input/projection/conv/predelta/gate/output boundaries. Avoid state_predelta readback as input-state proof: it is a view and graph may update underlying persistent state before retained postgraph read; snapshot needed for frozen-input harness. Same128policy, recordhandle/poll before competing inference/build. Goal incomplete.

Blocks11to12stages armc session72409 confirmed live; poll72409 before competing inference/build. Previous15800 terminal. Goal incomplete.

## Stage retention shifts first variation to block8; repeat identical layout next

15800 terminal0/bothCLI0; strict logits/tensors1,all19 required captures131each/2489 records. Warmup0..37,prompt9 38..56,prompt4 57..75,step1 76..94. Firstdifference43=L7(afterblock8,prompt9) NMSE0.000114174124/max0.0111532509. Same-frame embedding+L0/L1/L3/L5 records38..42 finite/identical0; L7 record43 strict1. Thus attn_residual/ffn_out layers10/11 already receive varying upstream input; no attention-vs-MoE culprit there proved. Logitsbits0 NMSE0.00762000924/max0.976641655; selected15(4602/264), margins0.00382423401/0.0288944244. Diagnostic23.39/23.09tok/s. Retention again changes first varying range; more bisection by changing outputs is not guaranteed narrowing. Next armc identical19-capture layout/policy128, comparea/c andb/c, no added tensors or build. Goal incomplete.

Blocks11to12stages session15800 confirmed live in arm a; poll15800 before competing inference/build. Previous51935 terminal. Goal incomplete.

## Block12 already varies during prompt with identical block10 boundary

51935 terminal0/bothCLI0; strict logits/tensors1,14 required names131each/1834 records. Warmup0..27,prompt9 28..41,prompt4 42..55,step1 56..69. Firstdifference35=L11(afterblock12,prompt9) NMSE1.43940033e-06/max0.00520630553. Same-frame embedding+earlythroughL9 records28..34 finite/identical0; L11 record35 strict1. L12/L14 andlater vary too. Logitsbits0 NMSE0.000792570272/max0.428418756; selected15(4602/264), margins0.0829963684/0.0277080536. Diagnostic23.42/23.00tok/s, no promotion. Underprobe suspectrange afterblock10 throughblock12/states; next preserve14/addL10 and attention-residual/FFNoutputs at layers10/11 to split attention/MoE and identify which block first varies. Same128policy, recordhandle/poll before competing inference/build. Goal incomplete.

Mid12to13boundaries session51935 confirmed live in arm a; poll51935 before competing inference/build. Previous12764 terminal. Goal incomplete.

## Block15 prompt output varies with identical block10 boundary

12764 terminal0/bothCLI0, strict logits/tensors1/all12 required tensors131each.1572 records: warmup0..23,prompt9 24..35,prompt4 36..47,step1 48..59. First difference31=L14(prompt9) NMSE7.30027696e-06/max0.0151357651; embedding+L0/L1/L3/L5/L7/L9 records24..30 finite/identical0; L14 record31 strict1. DownstreamL19/L29/L39 vary, actual headinput47(prompt4) NMSE0.00519365353/max1.10332632. Logitsbits0 NMSE0.00557900575/max0.829092979; selected15(4602/264), margins0.0225811005/0.0288944244. Underprobe variation afterblock10 throughblock15/state alreadyinprompt. Diagnostic23.74/24.00tok/s, no promotion.

Next preserve12/add L11/L12 (afterblocks12/13) to split range11..15, same128 policy. Recordhandle/poll before competing inference/build. Goal incomplete.

Mid15boundaries session12764 confirmed live in arm a; poll12764 before competing inference/build. Previous2645 terminal. Scoped prefix22/max187 has120equal/67different, confirming all7 early boundaries stable through selected15. Goal incomplete.

## Downstream pair localizes prompt variation afterblock10 throughblock20

2645 terminal0/bothCLI0; strict logits/tensors1, all11 required captures present131 each.1441 records, warmup0..21,prompt9 22..32,prompt4 33..43,step1 44..54. Logitsbits0 NMSE0.0138678706/max1.29138541; selected15(4602/264), margins0.0387077332/0.236562729,16 common-prefix rows. First tensor difference29=L19(prompt9) NMSE3.40025016e-05/max0.139535546. Same-frame embedding+L0/L1/L3/L5/L7/L9 records22..28 finite/identical0, required names verified; L19 record29 strict1. L29/L39 also vary there; result_norm prompt9 empty so no meaningful head-input claim there, actual prompt4 record43 differs NMSE0.00366111492/max0.670762539. Scope22/max187 includes prompt+steps1..15, later changed token inputs excluded. Under this plan first retained variation afterblock10 throughblock20/states, not uniquely faulty op or unprobed bisection. Diagnostic23.99/23.98tok/s, no promotion.

Next preserve11 and add L14 afterblock15,128 same policy; record handle and poll before competing inference/build. Goal incomplete.

Fullboundaries session2645 confirmed live in arm a; poll2645 before competing inference/build. Previous46752 terminal. Goal incomplete.

## Corrected embedding capture: all7 early boundaries equal on common prefix

46752 terminal0/bothCLI0.917 records/7 per frame: warmup0..13,prompt9 14..20,prompt4 21..27,step1 28..34. Required names model.input_embed,L0/L1/L3/L5/L7/L9 verified. Logits bits0 NMSE0.00205683904/max0.457926899; selected104(9764/47), margins0.363044739/0.0863609314,105 common-prefix rows. First tensor changes756..762=step105 AFTER token104 divergence. Scope skip14/max742 prompt+steps1..104 all742 finite/identical0 with required-name gate. No varying early block observed under this plan despite logits0 variation. Diagnostic24.50/24.44tok/s, not promotion.

Added optional --require-tensors exact names in selected scope; defaults unchanged. Present two-name test0, missing embedding1, name outside scope1 verified. Next preserve7 and add L19/L29/L39/result_norm to examine downstream boundaries within same new plan. Goal incomplete.

Inputplus4 session46752 confirmed live in arm a; poll46752 before competing inference/build. Previous26053 terminal. Goal incomplete.

## Early-plus-mid terminal; actual embedding name corrected for next probe

26053 terminal0/bothCLI0; strict logits/tensors1. Hello128 intended embedding+L0/L1/L5/L7/L9 actually emitted only5 boundaries: qwen35moe names embedding model.input_embed, not qwen3next model.embed_tokens. No embedding stability claim.655 records, warmup0..9,prompt9 10..14,prompt4 15..19,step1 20..24. First difference12=L5(prompt9) NMSE0.000140893768/max0.0148127079; same-frame L0/L1 records10/11 finite/identical0, L5 strict1. L5/L7/L9 all prompt+shared-prefix frames vary; L0/L1 first changes after token divergence (108 single-token differences each). Logits bits0 NMSE0.00337936964/max0.72514087; selected19(1438/12622), margins0.0701026917/0.106679916,20 common-prefix rows. Diagnostic24.82/24.90tok/s. This probe shows range blocks3..6/state variation in prompt, not universal root cause. Next corrected model.input_embed plus existing5 and L3(block4) midpoint, same policy128. Goal incomplete.

Earlyplusmid session26053 confirmed live in arm a; poll26053 before competing inference/build. Previous12796 terminal. Goal incomplete.

## Added block8 shifts earliest variation to block2 at step11

12796 terminal0/bothCLI0; strict logits/tensors1. Hello128 L1/L5/L7/L9 retention. Logits bits0 NMSE0.00167951521/max0.396555126; selected112(0/65915), margins0.0625495911/0.0116291046,113 common-prefix rows.524 records; warmup0..7,prompt9 8..11,prompt4 12..15,step1 16..19. First difference56..59=step11 across all4 boundaries, L1 NMSE0.0045278368/max0.00885372609. Scope8/max48 prompt+steps1..10 all48 finite/identical0; scope56/max4 all4 differ strict1. No prompt boundary difference this time. Unlike previous3-boundary plan, earliest retained variation is block2, not uniquely blocks7..10; changed retention/lifetimes prevent unconditional bisection. Logits0 difference predates all retained boundary differences. Diagnostic25.53/25.59 tok/s; no promotion.

Next retain embedding and L0 alongside L1/L5/L7/L9 to inspect input/first-block relation within new plan, same128 policy. Record handle before competing inference/build. Goal incomplete.

Boundary2to6to8to10 session12796 confirmed live in arm a; poll12796 before competing inference/build. Previous20466 terminal. Goal incomplete.

## Midpoint6 stable; block10 varies during prompt, next midpoint8

20466 terminal0 aggregate/both CLI0; strict logits/tensors both1. Hello128 retained L1/L5/L9. First logits bits0 NMSE0.00964649462/max0.995898724; selected15(4602/264), margins0.054895401/0.0100212097;16 common-prefix logit rows. 393 records: warmup0..5, prompt9 records6..8, prompt4 records9..11, step1 records12..14. First L9 difference record8(prompt9) NMSE4.31836267e-05/max0.0101788305. Same-frame L1/L5 records6/7 strictfinite/bit exit0; L9 record8 exit1. Scope skip6/max51 covers both actual prompt chunks and decode steps1..15:34 equal L1/L5 records,17 differing L9 records. First L1/L5 changes57/58=step16 AFTER selected15 divergence, not fault evidence. Variation lies after block6 through block10 or associated states under this probe; not universal unprobed bisection. Diagnostic25.32/25.91 tok/s, no promotion.

Next serial boundary2to6to8to10-repeat128-a/b retains L1/L5/L7/L9, same policy/headcap0, adds block8 midpoint. Record handle and poll before competing inference/build. Goal incomplete.

Boundary2to6to10 session20466 confirmed live in boundary2to6to10-repeat128-a; poll20466 before competing build/inference. Prior67647terminal; goal incomplete.

## Boundary2to10 differences precede token divergence; midpoint6 added

67647terminal0/allCLI0. Logitsbits0 NMSE0.001300401/max0.351103626, selected111(0/65915), margins0.0973625183/0.107374191,112sharedprefixrows.262records L1/L9 pairs; firstL9diff23=step8 NMSE0.0031238361/max0.0115880258, sameframeL1record22bitidentical/finite strict0; output23strict1. FirstL1diff52=step23NMSE0.0023416319/max0.00292944908, also beforeselected111. Meaningfulvariationinrangeblocks3..10/statewheninputboundaryidentical, andearlierboundaryvarieslater underthisprobe. Notoneuniqueop or unprobedbisectionproof.

Next serialboundary2to6to10-repeat128-a/b sameHello/policy/headcap0 Retainonly^l_out-(1|5|9)$pluslogits; addscompactmidpoint6, notallintermediates/statepayloads. Recordreturnedhandle/pollbeforeotherinference/build. Goal incomplete.

Boundary2to10 session67647 confirmed live in boundary2to10-repeat128-a; poll67647 before other inference/build. Prior30679terminal; goal incomplete.

## First two block outputs stable on common prefix; boundary2to10 probe launched

30679terminal0aggregate/bothCLI0. Logitsbits0 NMSE0.0059607497/max0.752601147, selected56(271/198), margins0.0210227966/0.249969482,57commonprefixrows.262retainedrecordsalternateL0/L1: warmup0..3,prompt9records4/5,prompt4records6/7,decode8+.120equal/142different; first120/121=step57 AFTERtoken56 divergence. Strictfiniteprefixskip4/max116 exit0/all116equal acrossbothoutputs; no first-two-blockdefect evidenceunderprobe. HostC9E4D8A/DSP27DC648C hashesverified.

Nextboundary2to10-repeat128-a/b Hello samepolicy/cap0 retainingonly^l_out-(1|9)$ pluslogits. Outputafterblocks2and10 withinoneplan, notcross-probebisectionproof. Recordreturnedhandle/pollbeforeotherinference/build; no productionpromotion, goal incomplete.

First-two-boundary repeat session30679 confirmed live firsttwo-repeat128-a; poll30679 before competing build/inference. Previous12013terminal; goal incomplete.

## First layer stable on common prefix despite logits0 variation; two-boundary128 next

12013terminal0aggregate/bothCLI0. Logitsfirstbits0NMSE0.00141180521/max0.368807673, rawargmax95, selected104(9764/47), margins0.0684356689/0.0611820221; comparator105commonprefixrows. Firstlayer131retainedrecords108equal/23different, firstdiff108=step105 AFTERselected104. Scope skip2/max106 prompt9/4+steps1..104 strictfinite/bit0/all106equal; warmup0/1alsoequal. Laterfirst-layerdifferencesfollowdifferentinputs, not firstlayerfault evidence. Underprobe logitsvariationalreadyafterfirstlayerorassociatedotherstates, notunprobedrootcauseproof.

Next serialfirsttwo-repeat128-a/b Hello samepolicy/headcap0 retainingonly^l_out-[01]$pluslogits, two boundarieswithinoneplan tochecklayer1input/output relation. Existingretentioncaveat; sharedprefixcomparisonmustderivefromselectedlogs, don'tcountafterdivergence. Recordreturnedhandle/pollbeforeotherinference/build. Goal incomplete.

First-layer128 selective repeat session12013 confirmed live in firstlayer-repeat128-a; poll12013 before other inference/build. Prior97615 terminal; goal incomplete.

## Midpoint repeat fails bits13, all64tokenssame; first-layer128 probe launched

97615terminal0aggregate/bothCLI0. Logitsfirstbits13 NMSE0.000112541827/max0.166591644, all64selectedsame. Mid20trace67records17equal/50different, firstrecord17 NMSE0.000244394218/max0.00946378708. Mapping2warmups+prompt9/4: record3=step0,lastpromptcol;record16=step13 stable/finite strict0;record17=step14 diff/finite strict1. Atfirstlogitdiff currentmid20hiddenstillidentical, variation downstreamoritsstoredstatesunderprobe, notuniqueopproof. Changedretentioncannotbisectionunprobedgraphasiflifetimesunchanged.

Addedoptional --require-finite tensorcheck(defaultoffforlegit-infmasktraces); actualmid20self67finitepass, identicalNaNsyntheticcorrectlyrejected. Nextserialfirstlayer-repeat128-a/b Hello samepolicy/HeadPrefetch0, only^l_out-0$retainedpluslogits, longer128capturesintermittency; strictgatesandsharedprefixanalysisaftercompletion. Recordreturnedhandle/pollbeforeotherinference/build. Goal incomplete.

## Mid-model boundary repeat live

Head-inputprobecompletedandchangednextaction: upstreaminputvariationunderprobe. Scopedcomparator self16pass0/incomplete3requestedonly2availablefails1verified. Serial97615 confirmed live mid20-input-repeat64-a/b Hello64 samepolicy/headcap0, RetainTraceonly^l_out-19$ pluslogits. Midpoint20/40layeroutputboundary, no all-layerretention or callback. Comparefirstdivergenceonlywithinsharedprompt/generatedprefix, fulltensorreportlaterrowsdiagnostic. Retentionmayalterlifetime/stabilize; not automaticallyfirstbadop or correctnessfix. Poll97615 beforeotherinference/build; goal incomplete.

## Head-input repeat terminal: variation already upstream of head

98885terminal0aggregate/bothCLI0, strictinput1/logits1.67retainedresult_normrecords: warmuprecords0/1identical, record2emptycheckpointidentical; actualheadinputrecord3alreadydiffNMSE0.000264430944/max0.24225235. Logitsbits0 NMSE0.000414554051/max0.204733849; selected15(4602/264), margins0.0902500153/0.152362823. Underretentionprobe headinputalreadyvariesbeforefirstgenerateddifference; nothead-only explanation, notfirstupstreamop/rootcauseproof.

Comparatoroptional --skip-records/--max-records scopeswithoutchangingdefault; strictrequestedlength, readsfullframing/counts. Scope3+16 coversactualsteps0..15commonprefix, all16inputrecordsdiffer; laterstep16+excludedbecauseprefixdiverged. Nextmoveboundaryearlierwithminimalretention; no livejob, goal incomplete.

Head-input selective repeat session98885 confirmed live in head-input-repeat64-a; poll98885 before competing build/inference. Priorheadcostsweepterminal, headcap0normalpolicy. Goal incomplete.

## Head depth cost sweep terminal; selective head-input repeat launched

56325terminal0/all3CLI0. Cap2runtimeactual2/VTCM1984512,19batches/15complete1384op-single-token samples, headmean2953.8/median2933/range2928..3094/stdev45.574us,last2933; finalopsum33.062ms. Depth8/4/2 nearlysameinstrumentedheadcost, no latencywin demonstrated; cap0normaldefaultretained, no promotion. ReducedVTCMnotwholemodelgainproof.

Nextserialhead-input-repeat64-a/b Hello expanded/MM2/FA1/GDN1/routerHVX/UB128/defaultworkers/fusion1/replay1/HeadPrefetch0, logitsplusRetainTraceonly^result_norm$. Strictinputtensor andlogitrepeatgates; retainedOUTPUT changeslifetime, stable result wouldnotprovefix/excludeobservation suppression. Varyingheadinput wouldestablishupstreamvariationunderprobe. Recordreturnedhandle/pollbeforeotherinference/build; goal incomplete.

## Head cost cap8/cap4 effectively similar; cap2 live

56325 confirmed live in cap2-stages16; caps8/4 CLI0/runtimeactual8/4 andVTCM4737024/2902016. Newread-only parse_head_profile.py filterscomplete1384-op batches with exact single-tokenheadshape, excludeswarmup/checkpoint, checksallselectedbatchcounts. Bothlogs19batches/15selectedcomplete. Cap8headmean2957.133/median2945/min2928/max3123/stdev48.022us; cap4mean2942.800/median2934/min2929/max3030/stdev25.021us. Lasthead2929/2930us, finaloperatorsum32973/33240us. Smallunpairedinstrumenteddifference notspeedwin; lowerVTCM islayoutfactnotwholemodelfix. Cap2actual2/VTCM1984512verified, results pending. Readerpositive/incomplete/wrongshape synthetic validationpasses. No competinginference/build, goal incomplete.

Head cost sweep session56325 confirmed live in cap8-stages16. Runtime DEBUG verifies cap8 actual8/VTCM4737024, matching prior trace peak8. Caps4/2 and finalbatch comparisons pending. Poll56325 before other inference/build; no promotion.

## Headcap ABBA terminal: both arms fail repeatability; cost-only profiles launched

59624 terminal0 aggregate; all4CLI0, strictcross1/cap4repeat1/cap8repeat1. Cap8repeatfirstbits0 NMSE0.000346027369/max0.213671744, selected15(264/4602), margins0.0288944244/0.00694847107, stopsafter16commonprefixrows. Cap4repeatbits0/all64selectedsame. Crossdifferencecannotisolatecapcausality becausebotharmsunstable. Diagnostic rates8a26.96/4a27.23/4b27.00/8b27.14, no speed acceptance.

Next serial basicprofile Hello16 caps8/4/2 expanded/MM2/FA1/GDN1/routerHVX/defaultworkers/UB128/fusion1, no logits/tensorretention. SelectionDEBUGlogsactualdepth/VTCM, lastbatchheadcost/ops aftereach. Cost-onlyresearchdoesnotqualifymodelrepeat/CPU/MTP or performance. Recordreturnedhandle/pollbeforeotherinference/build. HostC9E4D8A/DSP27DC648Cexperimental, defaultcap0 retained; goal incomplete.

## Cap4 repeat fails bits but keeps64 selected tokens; cap8 repeat pending

Session59624 still live aftercap4-b applicationcleanup. Independent cap4-a/b strictrepeat exit1, firstbits0 NMSE0.000177878099/max0.188775241, all64selectedtokenssame/noargmaxdifference. This arm itself is not logit-repeatable; initial cross8/4failure cannot be attributed todepth alone. Cap8-b and aggregategates pending. No competing build/inference; goal incomplete.

## First headcap cross-depth pair fails; same-depth repeats pending

Session59624 confirmed live, cap8-a andcap4-a applicationcleanup complete. Independent strictcross exit1 firstbits5 NMSE0.000193934919/max0.152682781, selected15(264/4602), margins0.0288944244/0.0435295105, stopsafter16commonprefixrows. This fails cross-depth qualification but doesnotattribute difference tocap4 before same-depth ABBA repeats. HostC9E4D8A/DSP27DC648C hashesreverified. No overlapping inference/build; pollsamehandle. Candidate/defaultpolicy notpromoted, goal incomplete.

Head-prefetch ABBA session59624 confirmed live in cap8-control64-a; poll59624 before other inference/build. Current experimental hostC9E4D8A/DSP27DC648C, headcap8 requested. Goal incomplete.

## Head-only prefetch control built and installed; ABBA64 qualification launched

Default-off GGML_HEXAGON_HEAD_PREFETCH0/2/4/8/16 caps existing VTCM selection only for named output.weight/Q6_K/input2048/output248320/oneactivation row, ordinary quantized HVX branch. No arithmetic/DSP/protocol changes. First build80771 failed undeclared ne01 before installation; corrected src0->ne[1], build49937 terminal0 installed hostC9E4D8ADE6884DA62DF7A597DCEAC6582AC036236C06134231A405BAF8E6399E. DSP27DC648C unchanged; onlyhostggmlhex copied. PriorEDB9FFDLL+source backeduphead-prefetch-prechange hashverified. Builder generatedotherarchDSP/buildcatalog but didnotinstallthem. Runner HeadPrefetch resetsdefault0/printsmetadata; rawdecode resets0. Selection debuglog reportsactualdepth/VTCM whenprobeon.

SerialABBA Hello64 expandedmodel/FA1/GDN1/routerHVX/defaultworkers/MM2/UB128/fusion1/replay1/logitsonly: cap8-a,cap4-a,cap4-b,cap8-b thenstrictcrossandbothrepeat comparisons. Recordreturnedhandle andpollbeforeotherbuild/inference. Traced ratesdiagnosticnotpaireduntracedbenchmark, 64passwouldnotqualify128intermittency/CPU/MTP. Goal incomplete.

## Head/expert per-op trace attribution validated

Previous turn progressed by complete330566-event decode capture; session74097 terminal0, no live job. Extended parse_thread_trace.py --tensor exact first-input filtering, batch-relative clocks/op-window clipping and out-of-batch rejection; fullbatch integrity checked before filtering. Head/expertgate/down real analyses exit0; wrapped synthetic clippedDMA[10,20]/HVX[12,20] oracle passes. Peak overlapping interval count added to avoid inferring prefetch depth from totals alone.

Head2.937ms, eachworker HVX1.917..1.928ms, DMAunion2.910ms, outside tracedHVX1.001..1.013ms; not measured stall. Sample blk0 gate/up126us(thread0 quant0.387us/compute63.583us), down87us(quant4.271us/compute42.043us). Head VTCM4737024bytes; max16policy does not prove actualdepth16 (layout may choose lower). Next boundedheadprefetch experiment must verifyactualdepth/layout and unchanged arithmetic, preserve signed/hostrollback plus strict modelrepeat/MTPgates. No performance or correctness promotion, goal incomplete.

Session74097 now terminal0, RUN_EXIT=0; no live inference/build. Mode3 text dump imposed minutes of host logging overhead, printed token rate invalid for performance use. Complete final trace result below is authoritative.

## Actual single-token trace complete and validated; prior batch label corrected

Final batch4 dims2048:1,1384ops/35.249ms, operator sum34.364ms, head2.937ms. Reader exit0 reported/observed counts47080,40738,40538,40538,40458,40458,40458,40298,0,0,0 exactlymatch, no saturation/unmatched/open. Batch3 dims2048:4 is prompt/checkpoint work, NOT single-token decode; earlier entries' label corrected. Its1994-vs1384 op difference is shape difference, not evidence a single-token early graph lost fusions.

Thread0 activation quant0.145ms/prep0.0438ms/HVXcompute13.493ms, DMA residence union20.601ms (sum102.300ms), overlap tracedHVX10.750ms/outside9.851ms; per-thread outside8.608..9.851ms. Outside is not DMA stall and cross-thread totals are not latency. Main-thread L2FLUSH0.435ms/INIT0.296ms. Actual replay unavailable underprofiling, no uninstrumented speed or correctness promotion. Capture74097 final application cleanup logged; poll shellhandle terminal before next inference/build. Goal incomplete.

## Single-token trace batch emitted; event dump still live

Session74097 confirmed live, batch3 reports1994ops/75.394ms and thread counts81634,78482,78482,78482,70284,70284,70284,70284 below131072. Operator rows already emitted sum74.214ms but event log still streaming; no final event completeness claim. This early graph has no MUL_MAT+ADD/SSM_CONV+WIN rows, unlike16-token final1384-op control; therefore not a same-plan steady-state comparison. Profiling disables replay via want_replay opt_profile==0 guard. Quant type concern resolved: current host EDB9FF/DSP27DC648C hashes match, model size/mtime unchanged, prior stages16 loader has same q4_1=5/q8_0=105/q5_K=60 inventory. Warmup mixed-type rows do not imply external model mutation. Poll74097 to terminal; goal incomplete.

## Prompt trace saturated; decode capture remains live, overlap analysis extended

Session74097 confirmed live. Warmup batches0/1 complete; actual prompt batch2 reports2008ops/299.088ms, thread0 exactly131072events (capacity), so reject that batch for complete event decomposition. Final single-token decode pending, no rerun or concurrent inference/build. Added same-thread DMA/HVX interval intersection and residence outside traced HVX (not a measured stall). Synthetic wrapping example gives8cycles overlap/2outside, disjoint-overlap oracle11cycles passes.

Source audit host selection chooses deepest fitting prefetch from16/8/4/2 for src1_nrows<=4, otherwise caps at2. Important: top8 expert activation rows can fall in >4 branch even for one token; do not assume all decode matvecs use16. Later experiment should isolate vocabulary head depth first and verify actual kparams/layout, preserve arithmetic/model, use repeated bit/selected gates and paired latency before promotion. Current27DC648C still unqualified. Goal incomplete.

## Trace reader validated against completed warmup batch; capture still live

Session74097 confirmed live again, same job, second warmup batch still streaming. First batch2054ops is warmup (log explicitly warming up before it), not actual prose prefill or single-token decode; previous entry's prefill label corrected here. parse_thread_trace.py --batch0 exits0, all reported/observed counts match, no capacity reached/unmatched/open events. Thread0 DMA sum131.538ms versus union33.344ms inside58.590ms batch, HVX_COMP25.322ms; overlap prohibits summing to wall latency. Not decode tuning evidence yet.

Reader now supports zero-based --batch selection and retains only selected/latest batch rather than all batches. Synthetic missing/saturated/open-event case correctly exits1; UINT32 wrap case previously passed. No inference/build restarted; poll74097 to terminal before using final decode trace. Goal incomplete, current candidate unpromoted.

## Per-thread DMA/compute trace capture live

Previous goal turn progressed via768 fused-bias checks and fresh full-model stage profile. Session74097 confirmed live: dma-contract-threadtrace2, expanded prose2 tokens, current27DC648C/MM2/FA1/GDN1/routerHVX/UB128/defaultworkers/fusion1/replay1, DeviceProfile3 and OPTRACE131072 per thread. No retained tensors/logits or arithmetic changes. Prefill batch reports2054ops/58.590ms, thread counts77498/73652/65454/... below capacity; event log still streaming, not terminal or complete decode evidence. Poll74097 before any competing build/inference; do not restart from observation timeout.

Added read-only parse_thread_trace.py: last-batch completeness/capacity checks, per-thread event pairing and32-bit timestamp unwrap, interval sums and unions by event kind. Syntax check and synthetic overlapping/wrapping clocks pass (10cycles each across UINT32 wrap). DMA is submission-to-pop residence, not DMA-engine busy time; thread/category sums cannot be added into wall latency. Need terminal actual-trace validation before relying on output. Goal incomplete; candidate still unpromoted.

## DMA-contract fused-bias controls and full-model stage profile complete

Session73399 terminal0: fused-bias replay1/0 each12/12x32, 768 passes with basic profiling. Actual MUL_MAT+ADD hvx-tiled logged, including inplace/out-of-place N32/64/256 and K32/2048. Full-model prose16 profile CLI0, final1384ops/33.197ms instrumented device sum; head2.938ms, expert gate/up4.671ms, expert down2.707ms. No speed or correctness promotion. Source audit finds bias pop_nowait followed by weight completion waits in same chain; skipped bias wait alone does not establish race. No live job; next targeted dense expert scratch/repeated-input controls or per-thread trace to separate DMA/quant/compute costs. Goal incomplete.

## DMA-contract attention/state focused controls pass; no live job

Session91496 terminal0: FA1/fusion1 replay1 and0 each8/8x32 (512 comparisons), GDN1/fusion1 replay1 and0 each11/11x32 (704 comparisons), stock thresholds unchanged. Additional1216 focused passes do not override model repeat failure at selected6. Candidate27DC648C still experimental, no performance/MTP/CPU promotion. Harness backlog and completed agent reviews saved in EARLY_ENGINE_REUSE.md and GEMMA_MOE_HEXAGON_FIT.md. Goal incomplete.

## DMA-contract full-model repeat fails; focused attention/state qualification next

Session98433 terminal0 aggregate, DMA_CONTRACT_REPEAT_EXIT=1. Both 128-token CLI runs exited0; first logit bit difference at1 (NMSE0.0027049565/max0.711775303), first selected-token difference at6 (430/321), margins1.19086647/0.4793396. Comparator stops after7 common-prefix rows, not a complete128-row equality comparison. Diagnostic26.97/26.95tok/s, no speed promotion. Candidate27DC648C remains experimental; compiler-contract repair is not sufficient to stabilize the full model. Prior signed rollback artifacts retained. Next serial focused FA/GDN replay on/off qualification; no competing inference/build. Goal incomplete.

## DMA-contract candidate installed; modelrepeat launched

Build32740terminal0, installed27DC648C23A3B0CA97947D89477B9D095E85717BD26DD8A9A731E26D7F92D20B/signedcatalogmembershipverified0. IRclobbers~memoryverified. Focusedreusefusion1/replay1and0each20/20x32pass1280, no thresholdrelaxation. Previous1932583E/catalog/headerbackupsdma-contract-prechange, EFCbackupalsoavailable. Candidateexperimental,noemittedreordering/rootcause/speedclaim.

Serialdma-contract-prose128-a/b expandedmodel/MM2/GDN1/FA1/routerHVX/defaultworkers/UB128/fusion1/replay1, logitsonly. Comparestrictcomplete128aftercompletion. Recordreturnedhandle/pollbeforeotherinference/build; goal incomplete.

## DMA compiler-contract candidate building; previous signed candidate backed up

Previousgoalturn progressedcombinedUB4/worker1negativecontrol andLTOIRaudit. No emittedreorderingproved. Saved1932583EDSP/catalog/dma-queue.h in dma-contract-prechange hashverified. Narrowdma-queue.hcandidate declaresmemoryclobberonrelease/start/linkasm andemptycompilerbarrieraftervolatilecompletioncheck, coversfastdonepath; noaddedhardwarewait/syncht/cacheflush. Pop_nowaitunchangedbecauseitdoesnotpromisecompletion. Compilercontractrepairnotmodelrootcauseclaim.

DSP-onlybuild32740confirmedlive usingexistingbuild-cacheline-candidate.cmd, hostlibsnotcopied. Headeraffects27translationunits, link/sign/installpending. NewmatmulIRinspectfor~memory; no inferenceuntilterminalhash/catalogverified. Candidateunqualified; baselineEFCand1932583Erollbackavailable. Nextreuse20casesreplay1/0plusmodelfull128repeat, CPU/MTP gates onlyafterevidence. Goal incomplete.

## DMA audit can inspect LTO IR and linked ELF; no patch yet

MatmulobjectisLLVMbitcode magicBCc0de, notnormalobject: objdumpobjectrejectedcorrectly. SDKhexagon-llvm-dis readsIR; linkedcandidateELFobjdumpworks. Firstsampledma_queue_push_fallback_2d linkedrelease/dmlinkpairat6510/6514 showsnearbydescriptor/data storesbeforepublication, notcompleteallcallsiteproof. InlineasmIRmemorycontractquestion remains; do not infer emittedreordering solelymissingclobber. No liveinference/build; currentstrictnegativecontrolrecorded. Nextauditallpublication/completionIRsites andconstructboundedcandidate onlywith defensiblecontractfix andA/Bcontrols. Goal incomplete.

## Combined control fails selected6; no live job

Handle98218terminal0aggregate,UB4_WORKER1_REPEAT_EXIT1: firstbits5NMSE.00432204872/max1.64874601, selected6(430/321), margins.451276779/.591995239. Actualthreads1/ub4/fullHTP,bothCLI0. Combinedsettingsnotstabilized;7.77/7.78tok/sdiagnostic. Findingssaved. No liveinference/build. NextinspectDMAassemblypublication/completioncompilercontracts, no speculativehardwarebarrierfix. Candidate1932583Eexperimental,EFCrollbackretained;goal incomplete.

## Combined first128 run completes; DMA compiler contract audit pending

Handle98218confirmedlive in cacheline-ub4-worker1-b afteraCLI0. ActualaHTPthreads1/hvx1/n_ubatch4/fullHTP19037.92,noCPUweights; diagnostic7.77tok/snotpromotion. Strictpairpending. Previousgoalturn progressedmodeladdressnegativeevidence/profile/stagedhandoffsourceaudit.

ReadDMAqueue push/pop/start/link code: dmstart/dmlink inlineasmvolatile registeroperands no memoryclobber; poll/wait wrappersdo clobbermemory, popfastpathvolatile done mayskip poll. Potentialcompilerorderingcontractquestion, notestablished emitted-reorderingbug or modelrootcause. InspectgeneratedHexagondisassembly beforepatching. No productionchange, installed1932583Eexperimental, goalincomplete.

## Combined microbatch4/singleworker control launched

Modeladdressauditdidnotestablishfocusedbyte-disjointhazard. Matmulsourcequant handoffinspected usesvtcm_act_raw ->vtcm_src1, noobservedDDRquantbufferhandoff; work_queue_run_async waitsbarrier/acquirebeforereturn, namealoneisnotasynccompletionbug. No newdefectclaim.

Nextserialhandle98218confirmedlive cacheline-ub4-worker1-a/b128prose expandedmodel/candidate1932583E/MM2/GDN1/FA1/routerHVX/fusion1/replay1/UBatch4/HvxThreads1,logitsonly. PriorUB4fullNPU/allHVXprofileused8workers; priorsingleworkerMM2usedUB128/HMXprefill. Combinedcontrol removesbothrequestswithoutMM1CPUrepack, actualplacement/threadcountsrequirecheck. Slowdiagnosticnotperformancepolicy. Pollsamehandlebeforeotherinference/build; goal incomplete.

## Model address profile terminal; shared-line reproducer reachability unproven

Handle21694terminal0,19segments18624descriptorrecords/8unalignednonweightpatterns/zero byte-disjointsharedlinepairs. Allreportedunalignedconv_state_last0/1, not focusedhazardproof. Finalbatch1384ops33.440msprofile;majorstagesinFINDINGS. Candidate1932583Estillinstalledexperimental,EFCbackupretained; no modelCPU/repeatqualification orspeedpromotion. Previousgoalturn progressedprosefailgate/newinspectorverification. No livejobs. Nextinternalmatmulactivationquantization/scratchownershipaudit, no speculativecachepatch. Goal incomplete.

## Prose gate fails14; model descriptor/address profile launched

Previousgoalturn progressedcodequalification/optionalverbose. Handle34208terminal1: math/codepass32, prosefailselected14(2014/15771), margins.324108124/.343389511, NMSE.00447142799/max.887376785;passwordunexecuted. Candidate1932583Estillnotpromoted; EFCrollbackavailable. Findingsupdated.

Addedread-onlyinspect_tensor_cachelines.py descriptorsegment/boundaryinspector, excludesweight/fenceflags, onlycountsbyte-disjointshared128lineswithinsegment. Testedreproducerlog20segments160recordsfinds64offsetpairs, no128offsetfalsepair. SharedlinesaloneNOTdirty/lifetime/rootcauseproof; modeladdressproofpending.

Serialhandle21694confirmedlive cacheline-model-address16 expandedmodel/prose/MM2/GDN1/FA1/routerHVX,DeviceProfile1+BackendVerbose,noOUTPUTretention/callback/logits. Addressinspectorandlastbatchprofileaftercompletion. Logging/profiletimingnotbenchmark/replayproof. Pollsamehandlebeforeotherinference/build. Goalincomplete.

## CPU suite code32 passes; prose CPU arm live; optional address logging ready

Handle34208confirmedlive nowcacheline-cpu-suite-prose-cpu. CodeCPU/NPUCLI0,require-tokens0all32, firstbits0NMSE.00146429343/max.489736795. DiagnostictracedNPU28.75tok/snotpairedspeedacceptance. Math/code2casespass; prose/passwordpending; suite stopsfirstfailure. Previousgoalturn progressedmathqualification; currentcodegateadditional evidence, no blanketpromotion.

MTPdiagnosticrunneroptionalBackendVerbose switch defaultoff, envGGML_HEXAGON_VERBOSE1andCLIloglevel5onlywhenrequested, metadata printed. ASTparseclean. No tensorOUTPUT/retain/callback changes; logging canstillaffecttiming, notspeedmeasurement. Preparednextmodeladdressaudit afterlivesuite: inspect byte-disjoint sublineviews/actualproducerconsumerplacement ratherthan assume focusedreproducer reachable. Nootherinference/build live; candidate1932583Estillinstalledexperimental, EFCbackupavailable. Goal incomplete.

## Candidate CPU suite math32 passes; code CPU arm live

Handle34208confirmedlive in cacheline-cpu-suite-code-cpu aftermathCPU/NPUCLI0. Mathstrictselectedgate0all32, numericalbitsdifferfrom0(asexpectedcrossbackend); NMSEfirst.00187819494/max.466886163. ActualmathfullHTP19037.92MiB,threads8,1932583Ecandidatehashverified, replayhit1381ops. Diagnostictraced28.70tok/snotpairedspeedacceptance. Suiteonly1/4qualifiedsofar; code/prose/passwordpending, stopsfirstfailure. Previousgoalturn progressedbycandidatefullmodelnegative/fusedoracle1280passes. Pollsamehandlebeforeotherinference/build; no productionpromotion. Goal incomplete.

## Candidate modelrepeat stillfails52; fusedoracle/CPUqualification launched

Previousgoalturn progressed viahandoffnegativecontrolsandline-overlapfix1280focusedpasses. Handle13540terminal0aggregate;modelrepeatstrict1bits0NMSE.00253665054/max.642757416,selected52(271/198). Notsolemodelrootcause, candidate1932583Eexperimental, EFCrollbackbackupsavailable. Signtoolcatalogverification0 andhosthashesunchanged. Findingsupdated.

Nextserialjob handle34208confirmedlive: dma-cacheline-fused fusion1/replay1+0reuseoracle both20/20x32pass1280additionalcomparisons,thenrun-decode-correctness cacheline-cpu-suite32tokens normalmodel/defaultFA2GDN2/routerdefault. Qualificationhashes/artifactsprinted, stopsfirstCPUselectedfailure. SeparatebaselinepolicyfromHVXdiagnosticpolicy; no claimselectedcorrectnessfromrepeatabilityalone. Pollsamehandlebeforeotherinference/build. Goal incomplete.

## Line-overlap candidate passes1280focusedcomparisons; fullmodelpairlaunched

Build60993terminal0,DSP-onlyinstalledsigned1932583EB642704F72CEA77A5FF7DDC9EFC24FB24E93627656A4F1A3A0C4B1ED. BaselineEFC+catalog+sourcebackedupcacheline-prechange, hostlibrariesuntouched. Fusion0/replay1and0reuse20/20x32 each pass1280comparisonsstock1e-7; previouslyfailing64offsetaliasesfixed. Findingsupdated. Notproductionpromotion/modelrootcauseproof.

Serialfullmodelhandle13540confirmedlive prose-cacheline128-a/b MM2/GDN1/FA1/routerHVX/fusion1/replay1/UBatch128 expandedmodel,logitsonly, noallocationfill/tensorretention. Strictpairrunsaftercompletion. Pollsamehandlebefore anyotherinference/build. Goal incomplete.

## Explicit handoff rescues same-line DMA; line-overlap candidate build live

Previousgoalturn progressed by byte-disjointreproducer/128offsetcontrol. Test-onlybuild13768terminal0. Extendedhandoffreplay1/0each18/20: onlyno-handoff64offsetaliasesfail unchangedNMSE; consumingproducerinADDbeforeDMA rescuesboth64offsetaliases32repeats,128offset/disjoint/oldcontrols pass.576passingcomparisonsperarm, failingcasesstopfirstrepeat. AdjacentADDuses16scalar table/staticDMA geometry, notoriginaldynamicglobalreadcoverage; mainissue sourcemaintenancehandoff.

Saved EFC DSP/catalog/htp-tensor.c baseline in cacheline-prechange,hashverified. Candidateis_tensor_dirty now compares128byte-line overlap using64bitbounds, zero-sizeguard; bookkeeping/clean ranges stillbyte-based conservative. No arithmetic/kernelchange. NewDSP-only build-cacheline-candidate.cmd avoidsproductionhostcopies, buildsinstalls/signsv81candidate. Handle60993 confirmedlive; pollbeforetests. Installedbinarymaystillbaselineuntilcopy; validatehash/catalogafterterminal. CandidateNOTpromoted; modelrootcauseunproven. Nextall20reusecasesreplay1/0, thenmodelstrictrepeats/CPUgates/profiles. Goal incomplete.

## 128byte-separated alias controls pass; only same-line64offset fails

Extendedcontrolreplay1terminal1,14/16: both64offsetaliasesfail unchangedNMSE;both128offsetaliasesandallothercontrols pass32.448passingcomparisons, failuresstopfirstrepeat. Sourcebyte-overlap dirtypredicate vslineflush remainscandidate; notyetexclude generic64offsetDMAissue. Nextsourceconsume/flushof64byteproducerbeforeDMA asnegativecontrol, pluskernel/flagaddressproof. No livejobs; no productionDSPchange. Goalincomplete.

## Byte-disjoint cacheline reproducer fails replay1 and0

Previousgoalturn progressed by HVX-onlyprefillnegativecontrol/cacheaudit. ExtendedexistingDMAreuseoracle optionaladjacentdefaultfalse: F32MM32x16 by32x1 produces64byteCPUoutput atbacking0, concatDMA beginsbacking64 withlargeoutput, RMSconsumes. Disjointcontrolseparateoutput. Firstattemptassertedunallocatedbackingbecausecreatedafterview; correctedcreationorderbefore execution. Test-onlybuild16209terminal0.

Replay1/0each10/12: adjacentaliasrows1NMSE.005390220,rows9.000873050failstock1e-7; disjointadjacentand8existingcontrols pass32each. Actualrows9addresses producer...1000 size64/overwrite...1040 size147456 verified byte-disjoint,same128line. NoDSPpatch/rootcauseclaim/fullmodelrelevanceyet. Added128byteoffsetcontrols toexclude genericDMAalignmentissue; test-onlybuild77229terminal0, extendedcontrolrunlaunchedprefixdma-adjacent-offset-control replay1. Recordresultnext. No productionbinarychange; goalincomplete.

## Cache audit: byte-range tracking versus line-granularity maintenance

Read htp-tensor.c dirtymark/merge/clean/flush and main batch boundaries. is_tensor_dirty tests byte-range overlap; hex_l2flush rounds to128-byte lines. Potential adjacent-disjoint view/cacheline-sharing hazard requires focused reproducer and actualmodeladdressproof; no defect established. Existing largeoverlap DMAreuse tests are not thisdistinctcoverage. FENCE excludes explicitlymanaged synchronizationtensors, notautomaticallyallstate. No speculativeproductionpatch. Next inspect128byteallocatoralignment/viewoffsets and add adjacent-line control to existingoracle ifreachable. No livejob; currentmicrobatchfailureevidence recorded.

## Microbatch4 repeats fail token33; HMX prefill not necessary

Handle33988 terminal0aggregate,bothplainCLI0, UB4_REPEAT_STRICT_EXIT1: firstbits5 NMSE.000923040539/max1.01711631, selecteddifference33(430/303), margins.198802948/1.47491455. Diagnostic27.60/27.22notpromotion. PriorprofilefullHTP19037.92and9678allHVXmatmul/zeroHMX. Findingsrecorded; no livejobs. Previousgoalturn progressed with actualHMXoracleproof and microbatchruntimeplacement. Returntocache/bufferownershipaudit; do not call thisHMXroundingalone. Productiondefaultsunchanged, goalincomplete.

## Microbatch4 profile completes; repeat pair live

Handle33988 stilllive nowprose-ub4-plain128-a. FirstprofileCLI0,actualn_ubatch4/fullHTP19037.92MiB/noCPUweightbuffer, finaldecode1381ops32.067ms. All9678profilematmulrecords hvx-, zero hmx-; warmupuses2tokens. Smallbatchchangesprefillstate/fusion/allocatorandhostcompute4.5984MiB versus17.1296, so not singlevariablekernelcausalproof. ProductionDSP EFC hash reverified unchanged. Remainingserialplain128a/b strictrepeatpending; pollsamehandle. Goal incomplete.

## HMX execution verified; microbatch4 full-NPU control live

Previousturn progressed by GDNwritebacknegativecontrol and12newprefillcases. Auditactualexecute-oplogsconfirmsall12newcaseshmx-tiled,replay0has384executions;768newcomparisons pass, sparseidentity/exactgridlimitsnoted inFINDINGS.

Runner addsoptional UBatch1..128default128, metadataandCLI printed, ASTparseclean. Handle33988 confirmedlive serialprose-ub4-profile16 thenprose-ub4-plain128-a/b MM2/GDN1/FA1/routerHVX/fusion1/replay1,expandedmodel. Profileforactualplacement thenunprofiledlogits-onlystrictrepeat. Smallprefillblocks intendedtoavoidHMXmatmulwithoutMM1CPUfallback; actualkernel/memoryverificationrequired. Batchingchangesrecurrence/fusion/allocationtoo, notcleanarithmetic-onlycausalproof. Productiondefaultunchanged,noDSP/hostbuild. Pollsamehandlebefore other inference/build. Goal incomplete.

## Exact-grid prefill replay0 also passes new cases; no live jobs

Explicitreplay0 terminal1 same18/20, all12newexactgridcases32repeats pass, twooldrandomQ4casesfail. Newcases768totalcomparisonsacrossreplay1/0. Runtimehardwarehmxavailability andsupports-op not executionproof; actualHMXkernel/profileproof remainsnextcheck. No liveinference/build. EFC productionDSP notedited/rebuilt; testexeonly. Goal incomplete.

## New exact-grid prefill cases pass replay1; old random Q4 controls fail

Initialfocusedrun terminal1,18/20casespass: all12new2048width/rows9..64 exact-grid cases pass32 repetitions each (384 comparisons). TwooldrandomQ4rows1cases fail NMSE5.454e-6/5.483e-6 vsstock1e-7; failingcasesstopfirstrepeat. Runnerstopsbefore replay0, so explicitlylaunched replay0separately. No tolerance relaxation; exact-grid result doesnotqualify randominputs or fullmodel. ActualHMXlogauditpending. No livebuild.

## Prefill oracle test build complete; focused run launched

Build38862 terminal0,testexeinstalled; DSPsubbuildsno-work. Existing activation_precision oracle MM2/fusion0/replay1then0 launched prefixprefill-exact-grid after buildcompleted.20cases (8old+12new),32repetitions each; strictstockCPU gate unchanged. ActualHMXpathnotyetqualified; run completion/results pending. No productionpolicy change or inferenceconcurrency.

## GDN-copy removal repeat fails; model-width prefill oracle build launched

Previousgoalturn progressed via convwindowprofileproof. Handle89531 terminal0: actualprofile1414ops33.270ms,30GDN1.577+CPY.927=2.504ms;WINretained. Plain128a/b strictrepeat1 firstbits10 NMSE.00036859493/max.324617147,128selectedsame. Separatewriteback notstabilized; no productionchange.

Existing activationprecision test nowoptionalrowsdefault1, oldcasesunchanged; adds F32/Q4_0 identityweights k2048, exact binarygrid inputs with each32block max127/64, rows9/13/24/32/33/64,32seeded repetitions each. Intendedtail/full HMXtile layout/precision-isolated control, actualpathmustbeverified. gitdiffcheck0. No newtestsfile. Test-only build-router-test handle38862 confirmedlive, launched onlyafter inference terminal; poll samebuildhandle, then existing activation_precision oracle MM2/replay1/0 and profile actualHMX. DSPsubbuilds no-work; no DSP/productionhost edits. Goal incomplete.

## Selective GDN writeback control launched

After terminal38589/profileevidence, nextserialjob handle89531 confirmedlive: prose-gdncopyoff-profile16 then prose-gdncopyoff128-a/b. Mask8126=8190minus64 disables GDN_CPY only, keepsconvwindowandotherflags. Source3440 matcher adds state-copy destination to GDN output list; no token1-only guard. This is a selectiveownership/submission diagnostic, not hypothesisproof; allfusionoff historicallyalreadyfails. Sameprose expandedmodel/MM2/GDN1/FA1/routerHVX, profilefirst foractualplacement, then unprofiled logits-onlyrepeat. Defaultfusion1unchanged,nohost/DSPbuild. Pollsamehandlebefore other inference. Goal incomplete.

## Conv-window profile verified; no live job remains

Handle38589 terminal0,bothCLI0. Finalbatch1384ops33.081mswith30WIN1.612ms; no-window1444ops36.233mswith30CONCAT2.971+30CPY1.278+30SSM_CONV.336=4.585ms. Actualsubstitution verified,60extraops,2.973msstageincrease; total3.152msnotpurestagecausalestimate. Profile disablesreplay and stronglyinflateshosttimings; no untraced speedclaim. Mainmatmulcounts/fusionsretained. Findingssaved. Previousgoalturn progressed with selectivefusionnegativecontrol; currentprofile evidence retains convwindowdefault and narrows remainingstate/ownershipinvestigation. No production binaries altered.

## No-window128 repeat fails; paired stage profiles live

Previous turn progressed via realprose repeat and selectivefusion runner support. Handle62744 terminal0 aggregate; NOWINDOW_REPEAT_STRICT_EXIT1 and REAL_NOWINDOW_STRICT_EXIT1. Plainno-window a/bfirstbits0 NMSE.000714954633/max.330351353,128selectedtokensidentical; diagnostic24.78/24.49tok/s. Notstabilized, no production promotion. Findings recorded.

New serial profile job handle38589 confirmedlive: prose-window-profile16 then prose-nowindow-profile16, same prompt/model/MM2/GDN1/FA1/routerHVX, masks1/8062, DeviceProfile1, no logit/tensor tracing. parse_last_batch after each. Verify actualSSM_CONV+WIN substitution and per-stagecost, profile disables actualreplay so not untraced/replay latency proof. Only one job; poll samehandle before other inference/build. Goal incomplete.

## Real prose repeat passes; selective no-window plain control live

Previous turn progressed via new-prompt plain repeat. Handle30499 nowterminal0: real-a/b all128 target logits bit-identical strict0, both59/68 acceptance; real-a alsoexactplain-a, plain-b varies starting3. Diagnosticreal25.09/25.16tok/s, not speed qualification. Findings updated.

MTP runner optional Fusion nowValidateRange0..8190 (same established decode-runner pattern), default1 unchanged, PowerShell AST parse noerrors. Source flag128 controls SSM_CONV+WIN, 1 specialenablesall;8062=8190minus128 requests everyother flag, including dormant flags whose matchers are not called. Newhandle62744 confirmedlive serial prose-nowindow128-plain-a/b MM2/GDN1/FA1/routerHVX/replay1,128tokens, logitsonly expandedmodel. Strictpair and frozenreal comparisons follow, distinctexitcodes. This tests targeted one-token convwindow fusion rather than globalfusionoff, which already failed in prior Hello controls. Runtime invocation proof still needed separately; maskrequest alone is not execution proof. No DSP/host changes or concurrentbuild. Poll samehandle; goal incomplete.

## Prose real MTP128 equals plain-a; independent real repeat live

Handle62081 terminal0 (aggregate): plain/real strict exit0, all128 target logits bit-identical to plain-a and same selected tokens. Real acceptance59/68 (includes9 rejected drafts), diagnostic25.09tok/s versus plain26.62/26.67; traced timings not performance acceptance. Plain-b differs numerically from both startingstep3 despite same128tokens, so no blanket MTP qualification. New handle30499 confirmed live prose-hvx128-real-b under identical settings, strict target compare againstreal-a aftercompletion. No production edits/builds/model switch. Previous goal turn progressed with second-prompt plain repeat evidence.

## Prose plain128 repeats tokens, not logits; MTP arm still live

Handle62081 confirmed live now in prose-hvx128-real-a. Plain a/b bothCLI0/fullHTP19037MiB, strict repeat exit1: first bits3, NMSE.00193076379/max.70274508, all128 selected tokens identical. Diagnostic26.62/26.67tok/s, not speed qualification. New prompt confirms remaining numerical repeat instability is not Hello-only; same selected sequence permits aligned numerical comparison throughout128. Real arm pending; do not attribute a plain/real numerical discrepancy solely to MTP. Poll same handle; no concurrent inference/build.

## New-prompt plain repeat and real MTP control live

Previous goal turn made progress: independent real128 target repeat strict bitidentity confirmed and user-requested review notes saved. Current serial inference handle62081 confirmed live: prose-hvx128-plain-a/b then prose-hvx128-real-a, prompt Explain why the sky appears blue in three clear sentences. Expanded MTP model shared by all three, MM2/GDN1/FA1/routerHVX/defaultworkers, fusion1/replay1/OpBatch3072,128tokens, logits only; no retained tensor/callback/fill. Compare plain a/b strict before interpreting plain a/real a, print distinct strict exit codes; aggregate shell exit is not qualification. Installed DSP EFC SHA reverified unchanged. Poll same handle before any additional inference/build. Goal remains incomplete.

## Real MTP128 independent repeat matches frozen target logits

Handle94182 terminal0: mtp-real-allhvx128-repeat CLI0, strict target comparison exit0 against mtp-real-allhvx128. All128 target rows bit-identical, no selected-token difference; both74 draft rows recorded and52/74 drafts accepted. Comparator result qualifies target rows only, not a separate draft-logit equality check. Diagnostic23.07tok/s,43.34ms/token; traced run is not a speed promotion. Existing real replay1/replay0 equality plus this independent repeat strengthens real-path repeatability evidence, not correctness versus CPU or unstable plain128. No production policy changes; goal remains incomplete. No live inference remains.

User-requested early-engine and Gemma reviews are complete. Future Zig/Rust/C++ Hexagon harness backlog is saved in EARLY_ENGINE_REUSE.md; Gemma architecture/shape review and qualification plan in GEMMA_MOE_HEXAGON_FIT.md. Keep llama.cpp reference lane; no new engine or model switch implied.

## Structured HMX controls pass; random contiguous also fails

Previous turn progressed via focused HMX GDN failure/component diagnostics. Build60637 terminal0 testexe only, optional structured controls and random contiguous9/K1 added. AutoGDN2 replay0 now9/11: random permuted1.561e-6 and contiguous1.530e-6 fail; attention/state contiguous3.28569139e-7/1.53669771e-6. Structured one-hotQ/K/gate0/beta1/statevalue1/16 grid passes32 in both layouts, runtime HMXchunked verified. Not same-logical-input layout comparison; structured cases change recurrence too. No tolerance relaxation or claimed rounding-only cause. EFC hash unchanged.

Expanded forcedHVX control handle81213 terminal0: replay1/0 each11/11x32 pass704 comparisons. No live inference/build. Goal remains full Qwen NPU performance/correct MTP, no production policy promotion. Next precision-focused frozen-input audit or allocation-preserving observation; this HMX case cannot explain already-observed GDN1 model repeat instability.

## GDN component diagnostics show larger saved-state discrepancy

Test-only build56863 terminal0; optional err diagnostics leave same total NMSE/gate unchanged and print attention/state only on failed repeated model-sized GDN. Replayoff components run CLI1 same9/K1 failure: attention NMSE2.9149949e-7, final-state NMSE1.56796905e-6, total1.561e-6. Both components exceed stock1e-7, saved state dominates tensor length. Source HMX uses FP16 intermediate tiles; precision contribution is plausible, not a proven indexing defect. Recurrent-only HVX all8case32repeats already passed. No tolerance relaxation, no DSP/library changes or live jobs. Next use same frozen input to distinguish algorithm/FP16 rounding from layout/alias effects before optimizing this path.

## Model-shaped GDN oracle isolates HMX prefill CPU discrepancy

Previous turn progressed by verified1024 direct scalar-read reuse comparisons. Extended existing test_gated_delta_net with optional32seeded repetitions, default old tests unchanged. Model metadata confirms16key/32value heads,state128; new permuted group2 cases tokens1/2/4/9 xK1/2. Test-only build initially failed misplaced initializer, corrected before successful1297 copy. No production edits.

ForcedGDN1 replay1/0 all8cases x32 pass512 complete graph comparisons at stock1e-7. AutoGDN2 replay1 and separately0 each7/8: tokens9/K1 fails NMSE .000001561 versus1e-7, other7cases pass32 each. Failing case stops repetitions on first failure, so not256 passing comparisons per arm. Runtime confirms failed9/K1 hmx-chunked-scalar VTCM7690240,9/K2 hvx-recurrent-scalar VTCM1048576. Replay removal does not fix focused arithmetic mismatch. This is distinct from remaining fullmodel GDN1 repeat instability; cannot claim sole root cause or blanket HVX promotion. EFC DSP unchanged by hash, testexe/runner optional GDN selector only. No live inference/build; next separate attention/state error and audit HMX precision versus indexing.

## Dynamic scalar-read reuse passes focused unprofiled oracle

Previous goal turn progressed via corrected coverage audit and saved user-requested harness/Gemma notes. Replaced misleading clean_read arm with scalar_read: old F32MM output reshaped[1,2048,2*rows] feeds ADD with source[32,2048,2*rows], then unrelated DMAconcat aliases old storage and RMS consumes overwrite. Source selection proves HTP_BINARY_KERNEL_SCALAR (ne0=1,ne2>1 excludes static DMA table), worker reads global src1_plane directly. No fused-MM dependence/quantized primer remains. Existing dirty-only arms retained. Test-only build25254 terminal0, installed testexe updated.

Fusion0 replay1/0 both8/8 x32 pass:512 graph comparisons, including256 scalar-read variants. Actual HTP ADD shapes verified; alias width2 old/table/overwrite same000001B31F179100 address16384bytes. Replay captures uid1..4, scalar arm4ops/9tensors alias,4ops/10disjoint, profile0. Fusion1 replay1/0 also8/8 x32 pass,1024 total comparisons/512 scalar variants. These focused transitions do not reproduce clean-read stale-output defect; not arbitrary model lifetime proof. No production changes or live inference/build. EFC DSP unchanged by hash. Next recurrent-state/lifetime coverage or allocation-preserving first-divergence observation, not blanket cache patch.

## Harness backlog saved; Gemma shape review completed; reuse coverage caveat

User-requested harness backlog expanded in EARLY_ENGINE_REUSE.md (Zig/Rust/C++ host options, C DSP ABI, frozen inputs/state, explicit aliases/ownership, per-stage timings and strict promotion gates). Agent Gemma4 26B-A4B review in GEMMA_MOE_HEXAGON_FIT.md: resident model smaller/noGDN, but active FFN1.73x and output-head1.45x Qwen coefficients; shape fit not faster-decode proof. No model download/switch.

Test builds2899/47545 terminal0, testexe only. Initial F32 primer arms512comparisons pass but rows2 unfused and18 HMX, not established direct clean-read coverage. Revised zeroQ4 primer with oracle MM1 also512pass, actual HTP HVX but rawQ4 did NOT fuse. Therefore intended direct cached-read arm remains UNPROVEN; do not label these passes clean-cache handoff qualification. Q4zero keeps strict default tolerance, no numerical threshold relaxation. EFC DSP hash reverified unchanged. No live inference/build remains. Source marks DMA outputs dirty too; no new cache bug proven.

## Clean-read reuse arm added; test-only build live

Previous goal turn progressed by negative OpBatch128 repeat and recorded selected15 failure. Source audit: proc_op_req flushes tracked dirty sources, preflushes destinations, marks all destinations dirty even for DMA; subsequent source handoff therefore invalidates output before consumers. Clean-but-valid stale read is not established as a bug. Existing oracle covers dirty dead F32 producer; added clean_read bool arm executes unrelated F32 matmul+ADD residual fromdead before DMA overwrite, intended to repopulate clean cached lines after source flush. Source HVX fused residual accesses global src2 directly; must verify actual fusion/runtime placement before claiming coverage. Existing rows1/9 x aliasfalse/true x cleanfalse/true x32, default old arms retained. Build-router-test.cmd live2899, testexe only; no concurrent inference. Poll same handle, then Fusion1 replay1/0 and runtime proof. Installed EFC DSP unchanged, no production fix claimed.

## Smaller128-op submissions still fail selected token15

Handle72187 terminal0 aggregate from Write-Output; both128 CLI0, STRICT_LOGITS_EXIT=1. Firstbits0 NMSE .000569868686/max .227771223; selected mismatch15 (4602/264), margins .00424957275 versus .145418167. Both models fullHTP19037.92MiB; runtime confirms flush128 and actual replay disabled after flush2. Diagnostic25.70/25.51tok/s, not speed acceptance. Smaller submission boundaries do not stabilize this policy. Frozen real128 versus smallbatch-a also failsbits0/selected15, but differing/plain repeat already precludes MTP attribution. No retained tensors/fill/callback. OpBatch default3072 unchanged; no live inference remains.

Next inspect focused reuse coverage for clean cached CPU reads versus dirty CPU producer output. Existing proven preflush fix addresses dirty lines; whether a clean-but-valid cache line can survive reused DMA output is an unproven separate hypothesis. Need source/cache ownership audit and focused reproducer, not another global policy promotion.

## Smaller submissions test batch-boundary sensitivity without retention

Previous goal turn progressed through all40 third-repeat intermittency and early-repo audit. No prior live inference remains. Runner adds optional OpBatch64..8192 default3072, printed in metadata; parse verified, production default unchanged. Live handle72187 plain128 a/b expanded model/MM2/FA1/GDN1/routerHVX/defaultworkers/fusion1/requestedreplay1, OpBatch128, logits only, no retained tensors/callback/fill. Splitting submissions changes DSP cache flush/queue and fusion boundaries, not graph tensor OUTPUT lifetime flags. Source flush_batch marks replay_ok false after second flush; actual replay must be recorded as disabled for this control. Replayoff alone already failed128, so this is not a clean replay discriminator. EFC hash unchanged. Require complete128 same-prefix selected/full-logit gate, inspect placement and actual flushing. No speed/policy promotion; poll same handle before any concurrent inference/build.

## Early engines reviewed; thin harness salvage identified

User-requested agent cloned clean Specula/Trident via gh under Documents/github and reviewed all branches without competing builds/inference. Audit in EARLY_ENGINE_REUSE.md. Best reuse: Trident Zig mmap/GGUF host utilities and timing decomposition, Specula persistent output/state ownership and snapshot/artifact contracts. Neither supplies optimized custom Hexagon replacement; Trident u8/QNN arithmetic and per-shape context churn are rejected as Q4_0 shortcut, shared allocator only proposed. No harness implementation yet. Existing DSP/runtime retained. Latest all40 a/c exact versus divergentb establishes intermittent process repeats; production/MTP gates remain incomplete. No live inference at this entry.

## Same-layout third repeat exact: passing probes cannot prove causal suppression

Handle74449 terminal0, CLI0; strict a/c logits0 and760tensor identity0. Thus all40-only a/c exact while b differs startinglayer1. Also all40-a versus layer1-attn-a16 full logits exact (cross-layout gate0). Passing layer1 detail/attention checkpoint pairs cannot establish those probes caused suppression; process-to-process intermittency persists with identical checkpoint selection. All captures complete16 and selected inputs match. No inference remains at this entry, no production changes. Need allocation-preserving observation or focused lifetime reproducer rather than further monotonic retention claims. Specula/Trident agent review ongoing as requested.

## Single attention-residual checkpoint also suppresses variation

Handle55536 terminal0, bothCLI0;779tensor records and all16 full-logit/selected rows bit-identical, strict gates0. All40 endpoints plus attn_residual1 retain same1384 decode ops as all40-only, while full layer1 details raised1398; extra checkpoint changes n-tensors2451->2473 and memory/lifetimes despite same op count. This is suppression or intermittent repeat outcome, not attention/FFN localization. No proof one checkpoint is a production fix. Live handle74449 all40-only third process c compared with frozena tests repeat intermittency under identical checkpoint selection. User-requested Specula/Trident review agent remains separate and no competing NPU jobs. Poll same handle.

## Layer1 full-detail retention suppresses drift in this pair

Handle98865 terminal0, bothCLI0 and strict logit/tensor exits0.1767tensor records all bit-identical;16 full-vocabulary rows/selected tokens exact. Adding all named layer1 intermediates to all40 retained endpoints changes allocator lifetimes/fusion eligibility and suppresses the previous layer1 prefill failure. This is not a runtime fix or a production repeatability pass. EFC DSP hash unchanged. Live handle55536 minimal additional checkpoint attn_residual1, retaining prior all40 boundaries to split attention/FFN only if variation persists. Unique labels mtp-plain-layer1-attn16-a/b; same diagnostic policies. Poll same handle before inference/build. Source confirms OUTPUT flags prohibit several fusions (GDN+CPY line3454, conv+window3548, chained operations3773 etc.); retention is not only a lifetime perturbation.

## All40 retention preserves drift; first observed layer1 prefill

Previous goal turn progressed via dense19 failure and strict tensor gate. Handle63767 terminal0 aggregate from Write-Output, bothCLI0; STRICT_LOGITS_EXIT=1/STRICT_TENSORS_EXIT=1. Firstbits0 NMSE .0130977413/max1.21752167, all16 selected equal.760tensor records97equal/663different. Every layer0 record and both warmup graphs exact; layers1..39 each17 differing records (9-row prefill,4-row prefill,15decode). First differing record81 l_out1 width9: rows0..5 exact,6..8 differ, NMSE .0000664059222/max .00434350595. All40 retention does NOT suppress current configuration's failure. This localizes observed onset to layer1 in this retained allocation, not production. FullHTP model19037MiB retained; diagnostic rates20.40/19.68tok/s not speed result.

Live handle98865 pair retains all40 endpoints plus every named layer1 intermediate, regex (^l_out-[0-9]+$|-1$), labels mtp-plain-layer1-detail16-a/b. Same EFC/MM2/FA1/GDN1/routerHVX/replay/fusion/defaultworkers/expanded model. Must recheck onset since intermediates alter allocation and fusions. No production library edits or optimization acceptance. Poll same handle before additional inference/build. User requested parallel agent review of Specula/Trident for harness salvage; review agent must not contend with campaign inference/builds.

## Dense0..19 is exact; variation moves to later prefill and changes token15

Previous goal turn progressed by verified sparse14 failure and dense launch. Handle75525 now terminal0 (reader); bothCLI0, strict logits exit1, firstbits0 NMSE .00032017979/max .204569817, selected difference15 (264/4602), margins .151790619 versus .0293674469.418tensor records384equal/34different; only29/39 differ (each9-row/4-row prefill plus15decode). All retained0..19 exact in every record. Earliest9-row differences at29 rows6..8; at39 rows2..8. Prefill varies before generated-prefix divergence. Dense allocation shifts onset after19, not a production localization. All tensor records are still same-input comparisons: selected mismatch occurs on final sampled token, with no subsequent differing-token decode.

Tensor reader adds optional --require-identical (default diagnostic retained), rejects empty traces. Tested prior114-record self pair exit0 and actual differing pair exit1. Generic tensor comparison allows nonfinite intermediates because attention masks legitimately contain negative infinity; sampled-logit finiteness remains separately checked. No production kernel/library edit.

Live handle63767 all40 retained boundaries pair, plain16 same current EFC/fusions/replay/policies. Historical all-layer tests used old52A DSP, fusion/replay off and only3 tokens, with final-block prefill variation/decode suppression; this pair tests current fixes/real replay/longer16 rather than treating historical suppression as current proof. Capture all boundaries in one layout, strict logit/tensor exits printed separately. No concurrent inference/build; poll same handle.

## Layer14 retention moves prefill variation before layer9

Handle51236 terminal0 from diagnostic reader; both CLI0, STRICT_LOGITS_EXIT=1, firstbits0 NMSE .0113623985/max1.09743237, all16 selected match.114 tensor records29equal/85different: all layer0 exact; layer9/14/19/29/39 each15 decode plus9-row/4-row prefill differ. First differing9-row layer9 rows0..5 exact,6..8 differ (max .00232831389). Adding14 moved variation before9 rather than splitting prior10..19 interval. Sparse retention is not monotonic bisection; strong allocation-layout sensitivity observed, causality not established. Both retain fullHTP model19037MiB; no CPU_REPACK fallback.

Live handle75525 dense pair retains all layer outputs0..19 plus29/39 in one layout, same policy/model, unique labels mtp-plain-boundary-dense19-16-a/b. This tests early boundaries without changing checkpoint selection between individual layers; allocation remains diagnostic, not production-equivalent. No DSP/library edits or promotion. Poll same handle before any further inference/build.

## Layer9 exact while layer19 varies, including prefill

Previous goal turn yielded evidence (terminal handle25570), not merely a status update. Both boundary9-16 CLIs exit0; strict full-logit repeat FAILS firstbits0 NMSE .00255326411/max .499688864, all16 selected tokens match. Tensor comparison95 records44equal/51different: only layer19/29/39 differ, each15 decode records plus the9-row and4-row prefill records. Layer0/9 remain exact throughout. In the first differing9-row prefill, rows0..5 are exact and rows6..8 differ. Thus variation is not limited to decode or dependent on token-prefix divergence. Observed interval10..19 applies only to this retained allocation, not untouched production.

Live handle51236 repeats plain16 retaining0/9/14/19/29/39, same expanded model/MM2/FA1/GDN1/routerHVX/allfusion/replay/defaultworkers. Adding14 tests that observed interval while retaining previous checkpoints. Probe retention changes allocation lifetimes, so revalidate onset rather than assume monotonic bisection. DSP hash EFC17600F16F21B612DF2A43EC53B2F46967F95492642FA7D9A963C4682FCD46 verified unchanged immediately before launch. Strict comparator exit printed separately from diagnostic reader exit. No concurrent inference/build; poll same handle. No production policy promotion or engine migration.

## Adding layer29 moves drift earlier; per-tensor totals verified

Handle95718 terminal0 only from diagnostic reader, both CLI0; strict16 FAIL firstbits3 NMSE .000873213309/max .431005478, all16 selected equal.76tensor records37equal/39different. Analyzer now prints all per-name/shape difference totals: layer19/29/39 each13 single-token records differ; layer0 and all other records exact. First three logit rows/first two decode captures exact, then all three later boundaries differ. Earlier interval20..39 from previous allocation is NOT a stable production localization; added retention changes onset/lifetimes.

Live handle25570 plain16 pair retains0/9/19/29/39 to split current1..19 interval under its own layout. Same EFC expanded-model MM2/FA1/GDN1/routerHVX/defaultworkers/allfusion/replay. Preserve probe-specific caveat and selected-prefix checks. Poll same handle, no concurrent inference/build. No kernel fix/performance promotion.

## Sparse boundary probe preserves drift; first decode differs after layer19

Handle68899 terminal0 only from diagnostic tensor reader; both CLI0 but strict16 logit repeat FAILS firstbits1 (NMSE .00014866704/max .248159409), all16 selected tokens equal. Tensor reader57 records:28equal/29different. All l_out0 records bit-identical; all warmup/system/user prefill captured boundaries0/19/39 exact. Firstdecode l_out0/19 exact, l_out39 differs NMSE .000357052164/max .071290493. Laterdecode l_out19/39 both differ. Thus first observed decode variation lies after19/before39 in this retained allocation; not an untouched-production first-layer proof. Same selected prefix all16 allows paired intermediate comparison.

Live handle95718 plain16 pair retains0/19/29/39 with same MM2/FA1/GDN1/routerHVX/allfusion/replay to split firstdecode interval; preserve caveat that new retention changes lifetimes and may move/suppress failure. EFC DSP unchanged. Poll same handle; no concurrent inference/build. Broader correctness/MTP/performance goals remain incomplete.

## One-worker128 still varies; sparse layer-boundary pair live

Previous turn progressed via no-warmup failure, MM1 placement confound, worker control. Handle91860 terminal1: both one-worker128 CLI0, HTP model19037.92MiB no CPU_REPACK placement; all128 selected tokens match but firstbits33 NMSE .0011920827/max1.09820342. Diagnostic7.71/7.66tok/s, not production recommendation. Multiworker execution is not required for numerical variation; no blanket worker-count fix. First32 output rows exact, later floats differ while same generated prefix.

Comparator now prints metrics for firstbit-differing row even when selected/rawtop still equal; strict gates unchanged. Live handle68899 plain default-worker16 a/b, MM2/FA1/GDN1/routerHVX, retain only l_out0/19/39 and full logits, strict repeat then tensor comparison. Probe preserves fused wholegraph execution but changes retained-output lifetimes/allocations, so resulting first observed boundary is probe-specific unless baseline-equivalence demonstrated. Do not call suppression a fix. EFC DSP unchanged, no competing model/build.

## MM1 repeat passes but CPU placement confounds; HVX worker control live

Handle15093 terminal0, three vector16 CLI0, a/b and a/c all16 bits identical. BUT runtime load assigns CPU800MiB/CPU_REPACK16560MiB/HTP1677.92MiB versus priorHTP19037.92MiB;42/42 offloaded alone is misleading. MM1 changes placement, not just eligible prefill arithmetic, so this is NOT evidence of stable NPU-only execution or HMX root cause. Diagnostic10.14/7.70/10.02tok/s not promoted.

Runner MmSelect default2 restored on next calls; optional HvxThreads0..8 default0 sets existing NHVX selector (0 uses backend default). Live handle91860 two128 plain MM2/FA1/GDN1/routerHVX runs with one HVXworker isolate worker scheduling/cache handoff and compare strict full logits; audit buffer placement before interpretation. No production arithmetic/policy promotion, EFC unchanged. Poll same handle91860.

## No-warmup repeats fail; matmul family isolation live

Handle25448 terminal1; both CLI0. No-warmup128 a/b firstbits3 and selected112 (0/65915), NMSE .00167233585/max .611432552, reference margin .148765564/candidate reverse .0116291046. A versus frozen real128 same firstbits3/selected112. Warmup is not required for variation; do not promote --no-warmup as correctness fix. Default warmup remains unchanged.

Runner now optional MmSelect1/2 default2, existing environment selector exposed. Live handle15093 triple plain16 vector controls use MM1/FA1/GDN1/routerHVX to remove remaining eligible HMX matmuls (prefill), complete logits and two strict repeats; allfusion/replay still1, same expanded model/EFC. This is diagnostic kernel-family isolation, not a slower substitute final state. Poll same handle before model/build. Broader128 and MTP gates remain incomplete.

## Attention dirty-alias oracle passes; no-warmup plain repeats live

Test-only build84108 terminal0, corrected allocation order and HVX-size F32 producer4096x2. Handle64373 terminal0: FA2/FA1 x replay1/0 each8/8 cases x32 passes,1024 graph comparisons. Outputalias runtime proof same data address 000001D5FFE8F500 for fa_dead_cached and batch attention32768bytes; capture6ops/16tensors in alias arms, replay hits logged. No isolated attention alias failure reproduced; prior allocation asserts are harness construction errors, preserved. EFC DSP unchanged.

Runner gained optional NoWarmup (defaultoff), uses existing --no-warmup flag; production default unchanged. Source common warmup decodes BOS/EOS then clears memory/synchronizes. Live handle25448 runs plain allhvx128-nowarm-a/b, strict full-logit repeat, then compares a against frozen real128. This is a prior-execution/cache-state discriminator, not a claimed warmup bug or fix. No concurrent model/build; poll same handle.

## Zero-fill repeat fails; attention dirty-destination alias control added

Initial alias attempts after builds27730/88235 abort before execution: tensor->view_src->buffer NULL because dead backing tensor created after batch view. Not kernel failures. Corrected construction order creates dead before batch. Producer shape revised4096x2 (still8192 F32 elements) to avoid HMX32-row producer and exercise HVX cached stores. Rebuild84108 pending/complete audit, then unique fa-batch-alias-fixed labels; preserve failed harness logs.

Previous goal turn progressed: attention1024 comparisons and MTP replay-mode exact128 evidence. Handle62125 terminal1, both fill0 plain128 CLI0; strict repeat firstbits0, rawargmax95 but selected mismatch104 (47/9764), NMSE .00582894217/max1.2520566. Fill0-a versus frozen real128 firstbits5/selected104 (real9764/plain47), NMSE .0054341586/max .937845945. Thus zero initialization is not sufficient to stabilize/correct all-HVX plain128; no production filling adopted.

Extended existing attention batch oracle with aliasfalse/true arms: batch2 output forced onto dead F32 matmul[256,32] storage, producer built before attention and not an input, same-size8192 F32 elements. Single1+1 remains independent reference within CPU/DSP. Four visible lengths x both alias arms x32 changes; checks output DMA/cache handoff beyond earlier disjoint test. Test-only rebuild live; no DSP arithmetic edit. Poll build before oracle, then FA2/1 replay1/0 profile0, profile1 only for runtime layout proof. Installed baseline EFC unchanged.

## Attention oracle also passes actual replay; allocation-fill diagnostic live

Strict real allhvx128 replay1 versus real allhvx128 replay0 PASSES all128 full-vocabulary rows/selected tokens. Real target repeatability across these two controls is verified; plain baseline still unstable. No global CPU equivalence or MTP performance acceptance.

Profile0 oracle arms FA2/FA1 x replay1/0 all4/4 x32 pass, additional512 comparisons. Runtime logs prove actual replay capture uid1..4 and hits for each policy (5ops/13tensors). Together1024 profiled+unprofiled graph comparisons; actual replay coverage is only profile0 arms. Test does not reproduce pure batch arithmetic defect. No DSP/host library change, testexe only.

Handle84985 terminal1: real allhvx128 replayoff completes128/52of74 drafts, mismatch against unstable repeated plain starts atprompt/selected104. Prior plain replayoff repeat FAILS selected104, so replay is not required for variation. Compare real replayoff against frozen real replay1 separately to audit MTP repeatability.

Live handle62125 two plain allhvx128 repeats with diagnostic AllocFill0, full logits, then strict repeat and frozen real128 comparison. This tests uninitialized RPC allocation sensitivity, NOT a proposed production fix. Source fill memset runs at allocation before model load; changes memory/timing and must not be promoted solely if repeat improves. No concurrent inference/build; poll same handle.

## Focused attention batch identity passes512 changing-input comparisons

Source audit correction: ggml-hexagon.cpp want_replay requires opt_profile==0. Thus the profile1 oracle arms do NOT exercise actual replay despite requested replay1;512 comparisons cover arithmetic/profiled execution only. Run all four arms profile0 after current model job and inspect replay capture/hits. Do not call profile1 requested replay arms replay coverage.

Handle84985 first replayoff repeat128 CLI0, strictFAIL fromprompt NMSE .000474651218/max .255891502, selected diverges104 (9764/47), NMSE .00672605373/max1.00333798. Thus variation persists without replay; earlier cross-mode128 identity is not a reproducible fix. Real replayoff128 arm still live; poll same handle.

Four oracle arms FA2/FA1 x replay1/0 each CLI0,4/4 cases x32 repetitions:512 graph comparisons total, checking within-backend batch2 versus single1+1 bits separately on CPU and DSP. Profile confirms FA2 actual hmx-pipe and FA1 hvx at both token widths. This synthetic/permuted-layout test does not reproduce a pure attention batch arithmetic defect; does not prove full-model lifetimes/masks/state safe. Profile instrumentation remains present, so unprofiled oracle control is still useful. DSP EFC hash reverified unchanged; only testexe/source and runner options changed.

Live handle84985 repeats plain allhvx128 replayoff, compares to prior replayoff strict, then real allhvx128 replayoff and strict same-policy target comparison. Earlier real replay1 versus plain replay0 full128 already bit-identical. Require repeat gate before accepting stability. No concurrent inference/build, no promotion.

## All-HVX128 replayoff matches real MTP; focused attention batch oracle built

Handle3661 terminal1: replayoff128 differs from frozen plain128 bits7/selected15 but strict bitidentity PASSES against real allhvx128 for ALL128 target rows. Unfused128 differs from frozen baseline from prompt and selected15, so no fusionoff fix established. Replayoff128/MTP128 match is new evidence; repeated replayoff still needed before attributing replay as causal (plain repeats already vary).

Added test_flash_attn_batch_equivalence in existing test-backend-ops.cpp: model-like permuted Q[256,2,16], K/V[256,256,2] with F32 Q/F16 KV and causal F16 masks at visible7/13/32/45. Same input evaluated batch2 and single1+1, concatenated output; custom metric requires internal bitidentity separately for CPU and DSP (does NOT require cross-backend floats equal).32 seeded changing inputs per case, wholegraph/replay infrastructure. Test-only build81963 terminal0, only testexe copied; installed EFC DSP unchanged. Extended existing oracle runner with pattern/FLASH_ATTN_EXT/FA selector/profile (defaults unchanged), parse passes.

Next four separate focused arms FA2/FA1 x replay1/0 with profile1; catch failures so all arms execute, inspect individual logs not final aggregate exit. No production arithmetic edit/promotion. Then repeat plain128 replayoff and real MTP128 comparison if justified.

## Repeat controls complete; all-HVX replay/fusion isolation live

Handle5660 terminal0 from final64 comparison only: repeat128 strict bitidentity FAILS (firstbits0), all128 selected tokens equal; repeat64 strict bitidentity PASSES all64. CLI both0. Do not interpret aggregate job exit0 as both gates passing. Expanded model/defaultHello actual prompt13 and same EFC DSP. Prior128 plain-versus-real still FAILS selected15; no acceptance.

Live handle3661 sequential controls run plain allhvx128 replay0/fusion1, then replay1/fusion0, each against frozen allhvx128 full logits. This tests dependencies of newly observed HVX-policy plain instability; baseline16/128 FA2 fusion/replay evidence does not cover this changed policy. No concurrent inference/build or production source change. Prompt parameter parse verified, broader corpus postponed. Poll same handle3661; individual strict results matter, not aggregate final exit.

## All-HVX128 fails; plain length/repeat sensitivity isolated

Live handle5660 repeats plain128 then plain64; first repeat128 CLI exit0, complete128 comparison fails bitidentity from prompt (NMSE .000402363347/max .206838369), but all128 selected tokens match. Thus same-length full-logit instability is verified; token-length causation is not established. Runner now quotes defaultHello; actual prompt count remains13. Second64 arm still live, poll same handle before more inference/build. Next isolate FA1 numerical repeatability with unfused/replayoff and focused attention reuse/batch oracle; broader MTP corpus deferred.

Previous turn made progress with complete strict64 bitidentity. Handle87075 terminal1, both128 CLI exit0; strict128 fails first bits7/selected15 (plain4602/MTP264), NMSE .00167076229/max .538464785, reference margin .00282669067/candidate reverse .0288944244. Real52/74 accepted. No promotion.

Unequal-length prefix diagnostics: real allhvx64 versus real allhvx128 first64 full rows bit-identical; plain allhvx64 versus plain allhvx128 first bits7/selected15 (264/4602). Thus failing128 comparison includes a changed plain baseline, not merely new late MTP drift. Length causation or repeat instability not yet distinguished. Next repeat plain128 and plain64 independently before further attribution. Prompt parameter added to runner (defaultHello unchanged; single-line/no-double-quote guard), PowerShell parse passes; broader corpus deferred pending baseline repeatability. EFC DSP unchanged.

## Real MTP64 passes full logit bitidentity with consistent HVX attention

Live handle87075 first reruns strict64 bitidentity (passes), then sequential plain/real mtp-*-allhvx128 and strict128 bitidentity. Poll same handle; no concurrent model or rebuild. Both128 use same GDN1/FA1/routerHVX expanded-model contract.

Handle72824 terminal exit0; plain and real CLI exit0. mtp-plain-allhvx64 versus mtp-real-allhvx64 complete64 target rows each, no bit difference, no selected/raw argmax difference. Strict require-identical also passes. Settings GDN1/FA1/routerHVX, same expanded MTP GGUF and EFC signed DSP; all fusions/replay remain1, real cap1 and rollback enabled, KV-only/fused verifier off. Real accepts27/36 drafts, so this is not a no-draft workaround. Changing attention selector removes remaining observed batch1/2 discrepancy under this consistent policy. Does not prove a specific HMX bug or broader CPU equivalence.

Both traced (no stage/device profile) diagnostic rates26.75 plain versus23.72 MTP; NOT a performance improvement. Plain FA1 versus earlier FA2 differs in logits from prompt and selected token61, so policy changes must be judged against their matching plain target, not substituted as arithmetic-preserving production defaults. Next extend exact plain/MTP gate to128 then broader prompts and isolate HMX attention batch equivalence with focused oracle. No production policy promotion.

## Consistent GDN fixes prompt identity; remaining MTP drift starts at32

Previous goal turn yielded new evidence: handle13443 is terminal exit1 from the strict64 target-token gate; all four CLI runs exit0. mtp-zero-hvx-prefill1 versus mtp-real-hvx-prefill1 passes strict one-row bitidentity. Real default-GDN versus real forced-HVX prefill also passes bitidentity. Runtime profile shows both nine-token target GDN paths use hvx-recurrent-scalar with forced selector1, including K1 and history K2; history GDN+CPY fusion executes. Consistent arithmetic removes the observed prompt discrepancy without disabling real drafts/history.

mtp-plain-hvx64 versus mtp-real-hvx64: complete64 target rows each, first bit difference32, first selected difference61 (1061/353), NMSE .00615652845/max .902138725. Reference pair margin .0192584991, reversed candidate .0696849823. Real MTP accepts27/36 drafts. NOT a correctness pass or promotion; diagnostic rates26.25 plain/23.70 real are traced, real additionally stage-timed, not paired benchmarks.

Installed DSP hash EFC17600... reverified. No production source policy change. Live handle72824 runs plain then real mtp-*-allhvx64 with GDN1/FA1/routerHVX and full logits, then strict selected gate. This tests attention batch arithmetic (plain1 versus verify2) after fixing GDN policy. Poll same handle; no concurrent inference/build. Real MTP remains cap1 with verification and rollback.

## GDN history changes kernel family; consistent-HVX MTP comparison live

Handle71122 terminal exit1 only because strict one-row bitidentity fails; both CLI runs exit0. mtp-zero-prefill1 versus mtp-real-prefill1 complete one selected row each, same token9419 but logits NMSE .00173994209/max .809248924. Zero cap guarded to one-token only after prior64 abort; real MTP remains cap1.

Source audit identifies actual policy difference: delta-net-base.cpp keep=false builds GDN K1; keep=true builds K=n_rs_seq+1. Host/DSP HMX GDN eligibility requires K==1 and n_tokens>=8; real MTP K2 uses HVX recurrent while plain/zero-cap nine-token system prefill can use HMX chunked. Four-token later prefill is below threshold but consumes prior states. This is a supported causal lead, not yet runtime cause proof.

Live handle13443 sequentially runs mtp-zero-hvx-prefill1 and mtp-real-hvx-prefill1 (GdnSelect1, PROFILE1/logits), strict one-row comparison, then mtp-plain-hvx64 and mtp-real-hvx64 (GdnSelect1/routerHVX, full logits; real MTP stage timers), strict64 CPU-independent same-backend target-token gate. Real64 still drafts/accepts/rolls back with K2, not a zero-draft workaround. Installed EFC baseline/referenceOFF unchanged. Poll same handle; inspect perop GDN kernel family plus selected sequence before promotion or speed claim.

## Rollback baseline restored; zero-cap prefill matches plain but control aborted

Previous goal turn was progress (F32 reference implemented/oracle-gated; full-model failure evidence). Handle13340 terminal FAILS CPU raw step19. Build92127 terminal exit0; baseline EFC17600... signed membership verified and v81 reference optionOFF. Scalar artifact retained, not promoted.

Handle52557 zero-draft64 terminal exit1/runtime -1073740791: runtime still invokes one draft with cap0 then asserts n_outputs_max <= cparams.n_outputs_max. NOT a successful no-draft64 control. Its one captured target prompt row is bit-identical to plain64 prompt row; actual target/draft n_rs_seq=0 logged. Real MTP cap1 has differing initial prompt row. This distinguishes setup/history policy before verification, but reservation/layout also changes, so n_rs arithmetic causation not yet proven.

Runner now restricts DraftMax0 to Tokens1 prefill diagnostic to avoid known abort. Live handle71122 runs mtp-zero-prefill1 then mtp-real-prefill1 on same F32-MTP GGUF/routerHVX/allfusion/replay, stage/logit probes; compare one complete selected row and initial logits. Do not claim zero draft as real MTP fix. Graph recurrent definition located src/models/delta-net-base.cpp; audit history path and trace first layer/correctly paired prefill tensors next.

## Scalar quantizer rejected end-to-end; faster baseline restored; zero-draft MTP control live

Handle13340 terminal exit1. q8-f32-raw128 fails CPU step19 (13/303), earlier than baseline39; row NMSE .000701287132/max .353906333, reference margin .0568733215/candidate reversed .0376052856. Diagnostic46.09ms/token, no paired untraced claim. Despite focused activation pass, global CPU output not rescued; scalar reference not promoted. Preserved F076 artifacts.

Build92127 terminal exit0 using build-q8-reference.cmd OFF. Installed baseline hash exactly EFC17600F16F21B612DF2A43EC53B2F46967F95492642FA7D9A963C4682FCD46 restored, v81 cache optionOFF verified. Source reference option defaultsOFF; no lingering scalar mode.

Added DraftMax0/1 parameter to run-mtp-stage.ps1, default1 unchanged. New zero-draft control uses MTP loading/hidden extraction with draft cap0; common context need_n_rs_seq returns0, while real MTP cap1 requests1. Server drafts only when n_draft_max>0. Compare64 target rows to frozen plain64; inspect actual n_rs_seq/catchup/draft behavior to validate control. This is diagnostic isolation, NOT a substitute for real MTP. Real MTP token15 divergence still uncorrected.

## F32 quantizer does not rescue prose; longer raw comparison running

q8-f32-prose32 complete exit0, strict CPU token gate FAILS same step14 (2014/15771). NMSE .00710593797/max1.15200472; CPU margin .324108124, candidate reversed .130945206 (baseline reversed .19323349). Narrow identity quantization gap corrected, but Q8 activation precision is NOT sole explanation of prose divergence. Traced diagnostic46.31ms/token vs baseline traced35.69ms, not untraced paired performance evidence; scalar path not promoted.

Handle13340 still live, now q8-f32-raw128 against frozen alignment-cpu128. Poll same handle; do not interrupt/rebuild or assume final exit proves prose pass. Installed F076 diagnosticON remains experimental. After raw comparison preserve evidence, choose first-divergent model tensor/arith isolation or faster F32 quantizer only if justified; broader CPU and MTP gates remain unfulfilled.

## Quantizer model diagnostics live after2048 focused comparisons pass

Handle7262 terminal exit0; RMS512 also passes. Candidate focused total2048 passes: activation512/MMbias768/DMA256/RMS512. Installed signed F07662BA... diagnosticON; default source optionOFF; frozen model/host unchanged. No promotion.

Live handle13340 runs q8-f32-prose32 and strict comparison against frozen prose CPU32, then q8-f32-raw128 and strict comparison against frozen CPU128. Native comparer failure does not skip second diagnostic; final/job exit alone is not both-gates proof, inspect each comparison. No concurrent inference/build. Depending on results, inspect logits reduction and first-divergence/margins, repeat candidate, profile quantization phases and untraced latency, and revisit MTP. Signed rollback artifacts retained.

## F32 activation reference corrects focused precision gap

Build55057 terminal exit0. Installed signed v81 SHA256 F07662BA46E85191506D2B6D6B14B361BE3100999B62B03D310AAF094037B8EF; signtool catalog membership verified. GGML_HEXAGON_Q8_F32_REFERENCE=ON cache verified; host ggml-hexagon/llama DLL hashes unchanged EDB9.../981E... . Source option remains defaultOFF, installed artifact is experimentalON.

activation-f32-reference replay1/0 passes8/8 cases each,512 comparisons. Previously random Q4_0 identity failed; exact and F32 controls remain passing. Since only Q8_0 quantization calculation changed and packing/dot retained, this isolates that focused discrepancy to activation quantization arithmetic. Does not establish model CPU equivalence or speed benefit. q8-f32-mm-bias768, q8-f32-dma256 pass; RMS continuation handle7262 must reach terminal before model runs. Signed diagnostic artifacts saved q8-f32-reference-candidate; EFC rollback retained alignment-preflush-candidate.

Next run q8-f32-prose32 against existing frozen CPU prose reference, and raw Hello128 against existing CPU128, then repeat/latency/profile/MTP as evidence warrants. No simultaneous inference, no promotion. Remember reset nested optionOFF for baseline rebuild/rollback.

## Four-prompt baseline complete:3 pass, prose fails; Q8 F32 reference build live

Handle48107 terminal exit0; password passes all32 CPU-selected tokens. Complete baseline corpus: math/code/password each32 pass, prose fails step14. All use original GGUF A4D2D76... and corrected EFC17600... DSP, default router/fusions/replay. Do not describe suite as passing. No remaining model process before diagnostic build.

Live build handle55057 launched build-q8-reference.cmd ON: nested v81 GGML_HEXAGON_Q8_F32_REFERENCE enables scalar F32 activation quantization but retains tiled packing/dot kernels; parent others defaultOFF. v81 compile/link completed and parent sign/install flow continues. ON cache verified; installed hash still EFC at last check (before copy). Poll same handle to terminal before inference, verify signed installed hash, then activation_precision=1 both replay modes. Signed EFC baseline artifacts preserved. Source defaultOFF; nested v81 cache persists ON after experiment until explicitly reset. No promotion/performance claim; MTP remains open.

## Prose CPU gate fails; password control live; F32 quantizer candidate prepared

Handle54867 terminal exit1 after math/code pass32 tokens each and prose fails step14 (2014/15771). Prose row NMSE .00676040833/max1.13722801; CPU selected-pair margin .324108124, NPU reversed .19323349. Password not run by stopped suite. Live handle48107 now separately runs alignment-corpus-v1-password-cpu then npu and strict gate,32 tokens. Same EFC17600... installed DSP and original GGUF; do not rebuild/install while model live.

Added default-OFF nested DSP CMake option GGML_HEXAGON_Q8_F32_REFERENCE. Candidate branch computes per32-element F32 max/scale/inverse/roundf, stores FP16 scale, then reuses original tiled packing and integer-dot kernels. Original FP16 quantizer remains default; Q8_1/offset quantizer unchanged. This is a scalar diagnostic/reference path, NOT proposed final fast implementation and not yet compiled/tested. Source diff checks pass. New build-q8-reference.cmd ON/OFF configures v81 option then invokes existing rebuild/sign/install flow. AFTER handle48107 is terminal, build ON, verify hash/signature, run activation_precision oracle both replays first, then DMA/RMS/MMbias regressions before model tests. Preserved alignment-preflush-candidate artifacts provide signed rollback; remember nested v81 cache option stays ON until explicitly resetOFF. No promotion or speed claim.

## Corpus math/code pass; final-layer masked-flag lead ruled out

Handle54867 still live: math and code each pass32 CPU-selected tokens with complete traces; prose CPU now running, password not yet run. Full-vocabulary floats differ from CPU, gates allow those differences but not selected-token mismatch. Source audit confirms qwen35moe has an early-versus-late inp_out_ids branch controlled by embeddings_nextn_masked. However context defaults masked=false, and MTP explicitly sets target masked=false (draft masked=true). The branch alone does NOT distinguish plain and MTP target here; last-layer crop hypothesis is ruled out as stated. Do not change it or remove hidden rows needed by MTP catchup. n_rs_seq/state-history kernel policy and output-lifetime/layout remain actual candidate discriminators; require first-divergent runtime evidence.

## Exact activation controls pass; frozen four-prompt gate live

Build22839 terminal exit0/testexe only. activation-exact-replay1-replay-1 and activation-exact-replay0-replay-0 each6/8 cases: all4 F32 and both exact Q4_0 cases pass32 comparisons each; random Q4_0 cases fail first same NMSE .000005454/.000005483.384 passing comparisons across both arms,4 random-case failures. Supports precision-sensitive activation quantization lead, not whole-model attribution. No production DSP change, no tolerance relaxation.

Live handle54867 runs run-decode-correctness.ps1 -Prefix alignment-corpus-v1 -Tokens32: hash model/runtime artifacts, serial CPU/NPU cases math/code/prose/password, strict complete selected-token comparison. GGUF SHA256 A4D2D76A1E12E4E945F610C649B8BEC72BFD0C6956299611C51F6266414B3827. Math CPU completes32 traced tokens in1.266sec decode, so tracing alone is not sufficient to explain earlier raw-Hello CPU slowdown; do not assert cause from correlation. Poll same handle; current runner stops at first failing case, so do not claim all four covered if it stops. Next precision experiment should add opt-in CPU-compatible F32 activation quantization retaining tiled layout, prove identity oracle first, then full-model CPU tokens/repeatability and paired latency. MTP prompt divergence remains separate open audit. Previous live93624/22839 statements superseded.

## Corrected-kernel MTP still drifts; activation precision oracle reproduces gap

Handle93624 terminal exit0. alignment-chat-plain64 vs alignment-chat-mtp64 fails complete target-token gate at step15 (4602/264); initial prompt logits already differ NMSE .00173994209/max .809248924. Same expanded GGUF/routerHVX/allfusions/replay. Draft27/36 accepted. Diagnostic stage sums draft230.793ms/36 calls, two-token catchup84.491ms/36; target-counter delta2167.368ms includes one nearzero sample, not precise per-verify ownership. No MTP promotion or new speed claim.

Added existing-harness test_mm_activation_precision: exact identity weights F32/Q4_0, k128/256,32 changing F32 inputs. Build38826 terminal exit0/testexe only copied; installed DSP unchanged. activation-identity-replay1-replay-1 and activation-identity-replay0-replay-0: F32 cases pass64 inputs per replay; Q4_0 cases fail first NMSE .000005454/.000005483 at stock1e-7 with fusion0 and replay1/0. Thus reduced graph exposes CPU/NPU precision gap without fusion/replay. Not attribution of model step39 gap.

Added exact-input controls: each32-value block includes maximum127/64 and all entries integer/64, so CPU/NPU scale/inverse/values are exactly representable. Test-only rebuild launched; poll its live handle, then run both replay arms to contrast random versus exact inputs. No DSP implementation change or tolerance relaxation. Comparator selected-pair diagnostics remain unchanged gates. Broader corpus runner remains pending.

## Fusion and replay preserve128 logits; corrected-kernel MTP retest live

Handle10569 terminal exit0. alignment-unfused128 and alignment-replayoff128 each pass strict128-row bitidentity/all selected tokens against normal alignment-npu128. CPU step39 gap persists unfused with identical scores. No longer fusion/replay divergence observed for this prompt; no broad CPU equivalence claim. Installed EFC17600... hash reverified; no lingering inference before next job.

Live handle93624 runs alignment-chat-plain64 then alignment-chat-mtp64, same losslessly expanded F32-MTP GGUF, routerHVX, fusion/replay1,64 tokens/logits. MTP run includes stage timers (diagnostic only); KV-only and fused-MTP-verifier remainoff. Compare complete target selected sequences/logits and prompt row before attributing verification/rollback. Broad four-prompt gate still pending; goal incomplete.

## Longer CPU gate fails at narrow-margin token40; NPU controls live

Handle22616 terminal exit0. alignment-cpu128 and alignment-npu128 complete128 rows/selected logs. Strict CPU token gate FAILS zero-based step39: CPU279 versus NPU1608. Reference pair logits22.2513428/22.2421513 (margin .00919151306); candidate22.2472/22.2552643 (reversed margin .00806427002). Same-prefix row NMSE .00108732384/max .358676672. Narrow margin is evidence of a close decision, not proof of expected rounding or absence of bugs. Earlier16-token pass remains valid but is not longer acceptance. CPU traced eval431.441sec/128, sampling dump overhead not assumed to explain CPU evaluation slowdown.

Live handle10569 runs alignment-unfused128 then alignment-replayoff128,128-token full logits on installed EFC17600... . Compare both to alignment-npu128 strictly to distinguish fusion/replay dependence from general CPU/NPU numerical differences. No competing model or rebuild. Comparator now reports selected-pair scores/margins on divergence without changing strict gate. Corpus gate still unrun; MTP still unresolved.

## Longer reference verified live; frozen multiprompt runner prepared

Previous goal turn was progress (alignment oracle/full-model gates and profile). Handle22616 polled live this continuation; CPU128 reference is generating rows, not hung. Its first16 rows reproduce CPU-reference selected tokens and pass candidate prefix comparison, but128 gate remains incomplete. About20GB working set and20GB physical memory free observed; no memory-exhaustion evidence. Leave same job running; it automatically runs alignment-npu128 afterward. Do not restart or run a competing model benchmark.

Added run-decode-correctness.ps1: frozen four raw prompts (math/code/prose/password), default32 tokens, CPU then NPU serial full logits with complete selected-token gate, explicit replay/fusion/router policy, model and installed DLL/EXE/DSP hashes, refusal to overwrite existing labels. PowerShell parse and diff checks pass; suite NOT yet executed. Next run after current128 gate, using unique prefix. It is a focused four-case gate, not Steve's complete broader suite.

Read-only fusion audit: try_fuse_mul_mat_add rejects multirow HVX fusion unless n_hmx>0. Therefore generic hvx_mm_2d residual-alias overwrite hypothesis is not a demonstrated active multirow fusion bug; do not patch it speculatively. Actual MTP retest on corrected kernels remains required.

## Corrected full-fusion profile and untraced latency recorded

alignment-normal128 exit0:28.24tok/s,35.41ms/token,126 graph reuses. Single unpaired run, no speed improvement claim. alignment-profile16 completes output: strict16-row identity versus unprofiled raw-alignment-after-a passes. Last batch1381ops/sum32.990ms: MUL_MAT10.389ms (output head2.958ms), MUL_MAT_ID_NX4.637ms, MUL_MAT+ADD2.968ms/70 calls, MUL_MAT_ID2.669ms, SSM_CONV+WIN1.818ms, GDN+CPY1.786ms, MUL_MAT_NX1.486ms, GET_ROWS1.447ms. These are perop diagnostic sums, not untraced latency decomposition. All fusions remain active; shorter observed correctness repaired. Handle22616 continues CPU128 full-logit reference then matching NPU128; poll same handle. No promotion yet, broader prompt and MTP gates still required.

## Full-fusion short decode corrected; longer gate and profile live

Handle58760 terminal exit0. raw-alignment-after-a/b all16 full-vocabulary rows BIT-identical, selected sequences complete/equal. Candidate versus CPU raw-cpu-logits passes16 selected tokens, floating logits differ as expected. Candidate versus preflush unfused raw-preflush-fusion0-replay1 passes strict16-row bitidentity. Previously full fusion diverged CPU at step1; dispatcher change resolves that observed divergence without disabling fusions. Installed signed catalog membership verified for EFC17600... . Oracle logs confirm MUL_MAT+ADD actually fused.

Live handle22616 sequential job: alignment-normal128 (untraced), alignment-profile16 (PROFILE1/logits), alignment-cpu128 (CPU/logits), alignment-npu128 (normal NPU/logits). Longer CPU trace is diagnostic and slow; do not use traced/profile throughput as benchmark. No promotion until longer gates and more prompts; MTP still unresolved. Signed candidate artifacts preserved in alignment-preflush-candidate; preflush-only and original rollback artifacts retained.

## Alignment-aware fused matvec passes focused oracles; model gate running

Build97754 terminal exit0. Installed v81 SHA256 EFC17600F16F21B612DF2A43EC53B2F46967F95492642FA7D9A963C4682FCD46 combines destination preflush with alignment-aware residual add. Both matvec callsites in matmul-ops.c now use existing hvx_add_f32 dispatcher instead of hvx_add_f32_uaa; aligned inputs retain the aligned path.

New mm_bias oracle: k32/2048, n32/64/256, disjoint/inplace,32 changing inputs each. Unfused before patch passes768 comparisons across replay1/0. Fused before patch fails n32/64 in both alias arms (NMSE .288578213/.149568974 for k32 and .002702658/.002193338 for k2048); n256 controls pass. Small perthread residual slices violate128-byte source alignment. After patch fusion8 passes all768 comparisons across both replay modes. DMA-dead-reuse256 and RMS-reuse512 additional comparisons pass. git nested diff --check passes (line-ending warnings only).

Live handle58760 runs raw-alignment-after-a then b, normal all-fusion/replay1,16-token full logits, unchanged original GGUF/raw Hello. Poll same handle, compare CPU selected tokens and repeat bitidentity before normal128 speed/profile. Candidate NOT promoted; no new speed claim. Earlier preflush-only installed/live statements superseded.

## Residual-add matmul fusion isolated as failing family

Handle22967 terminal exit0. mask8 alone fails CPU sequence step1 (271/353), same .0150917313NMSE/max1.61351246. mask50 pairedMM/allreduce passes strict16-row bitidentity to unfused; therefore also CPU16 tokens. Other families individually passed. Source/installed preflush6D48E2F... unchanged, no inference live, no promotion.

Concrete next audit lead: hvx_mv_2d and repacked matvec add residual using hvx_add_f32_uaa with vtcm_src2+src0_start_row. F32 ssm_alpha has32 output rows distributed over8 threads; slices may start4float(16byte) apart, while helper requires aligned source2. Repacked quantum32 slices aligned. Need confirm helper alignment and precomputed start/rowdiv, add small F32 matmul+bias oracle (width32) with changing inputs, then test unaligned-safe helper or aligned perthread bias scratch. Do not infer fix from bisection alone.

## RMS-multiply-only preserves unfused output

mask58 completed exit0, fails CPU sequence at step1 with same first-decode .0150917313NMSE/max1.61351246. Handle77198 terminal. Live handle22967 now mask8(MMadd-only) then50(MMNx/MMIDNx/allreduce),16tokens/PROFILE1/replay1. Poll same handle; no binaries changed.

mask4 complete exit0:16 CPU tokens agree; strict16-row bitidentity to unfused passes. Profile2114ops/39.550ms includes121 RMS_NORM+MUL/697us, confirming active fusion. Rules out this family alone for observed prompt failure. Live handle77198 now only mask58 matmul/allreduce group; poll same handle and compare CPU prefix. If failing, split8(MMadd) versus50(MMNx/MMIDNx/allreduce). Installed/source preflush6D48E2F... unchanged; no promotion or benchmark claim.

## GDN-copy isolated pass; RMS/matmul split next

Handle22248 complete exit0. mask64 GDNcpy-only passes16 CPU selected tokens and strict bitidentity to unfused16 logits. Profile2205ops/39.996ms, GDN+CPY30/1.814ms confirms invocation. mask62 fails CPU token at step1 (271/353), same first-decode NMSE .0150917313/max1.61351246 as all1. Thus GDNcpy alone not culprit here; remaining family group reproduces failure.

Live handle77198 runs raw-preflush-mask4(RMSmul-only) then58(allreduce/MMadd/MMNx/MMIDNx),16tokens PROFILE1 replay1 routerdefault. Installed/source6D48E2F... unchanged, no promotion. Compare CPU tokens/strict unfused identities and actual opcode counts; continue family bisection toward repair.

## Complementary fusion group exact; GDN-copy split running

Handle51662 complete exit0. mask2944 CPU token gate passes16 and strict logits match unfused16/16; last profile1692ops/sum34.436ms, active SSM_CONV+WIN30,ADD+NW40,ADD+SG40,MOE+WNORM39,MOE+TOPK1. This group's fusions execute and preserve these outputs. mask3070 strict16-row comparison to normal all1 passes, validates equivalent active-family mask under profile. First group126 fails at step1.

Live handle22248 runs mask64 (GDNcpy only) then62 (RMSmul/MMadd/MMNx/MMIDNx/allreduce),16tokens/PROFILE1/replay1/routerdefault. Poll same handle; inspect logits/CPU tokens and actual fused counts. Installed/source preflush6D48E2F... unchanged, no promotion. Earlier live51662 superseded.

## Matrix complete; fusion-family bisection live

mask126 complete exit0, fails CPU token sequence at step1 (271/353), same first-decode NMSE .0150917313/max1.61351246 as all1. Final profile1924 ops/sum37.924ms:121 RMS_NORM+MUL,70 MUL_MAT+ADD,50 MUL_MAT_NX,40 MUL_MAT_ID_NX,30 GATED_DELTA_NET+CPY. Confirms active families in failing first group. Handle51662 now mask2944 then3070. Next narrow126 into64(GDNcpy) versus62(otherfirstgroup), unless complementary results contradict.

Handle13906 complete exit0. fusion0/replay0 strict comparison to fusion0/replay1 passes16/16 logits bit-identical, matching CPU tokens. fusion1/replay0 equals fusion1/replay1 bitwise but both differ CPU at step1. This isolates fusion-enabled execution as discriminator for this rawprompt on preflush candidate; not yet particular fuse/cause.

Live handle51662 sequentially runs raw-preflush-mask126, mask2944, mask3070,16 tokens each, replay1/routerdefault, full logits and perop PROFILE1. Compare CPU selected prefixes, mask3070 versus all1 identity and last-batch profile actual families/timings. Do not benchmark traced wall time. Installed/source6D48E2F... candidate unchanged, NOT promoted. Poll same handle.

## Fusion-off raw candidate passes CPU tokens with replay enabled

fusion1/replay0 completed and reproduces CPU token failure at step1 (271/353); its entire16-row logits are bit-identical to fusion1/replay1 normal candidate. Fusion-on difference therefore does not require replay. Handle13906 now only final fusion0/replay0 control. Next fusion-family masks126/2944 on same candidate, no binary change needed.

Handle31983 terminal: isolated preflush passes16 selected CPU tokens. First matrix run raw-preflush-fusion0-replay1 passes CPU16 tokens and strict bit-identity against raw-preflush-isolated16 rows. This changes replay and router selection relative to isolated while fusion remainsoff; those changes do not explain observed gap here. First matrix run exits0,2235-op replay. Live handle13906 now runs fusion1/replay0 then fusion0/replay0, router defaultoff. Poll same handle. Candidate6D48E2F... source/binary unchanged, not promoted.

Decode runner Fusion validation expanded to existing backend mask range0..8190 for diagnostic family bisection; default1 unchanged. Active fusion masks include126 (allreduce/RMSmul/MMadd/MMNx/MMIDNx/GDNcpy) and2944 (SSMwindow/MOEadd/sharedgate/routerweightnorm). Need confirm fusion-on/replay-off result before attributing and bisecting; no accepted performance claim from traces.

## Reuse defect also reproduces without replay; raw candidate capture live

dma-dead-mm-old replay0 exits1: disjoint rows1/9 pass, alias fail NMSE1.993436722/1.999657181. Replay not necessary for narrow defect. Added optional ReplaySettings selector to oracle runner, defaultboth unchanged. Logs prove same dead/concat addresses in alias arms and different addresses in controls; producer HVX-tiled at2 rows, HMX-tiled at18. HMX transfers converted output into direct DDR pointer. Candidate all256 comparisons passed earlier.

Reinstalled signed candidate6D48E2F... from preserved artifacts; signtool verifies. Live handle31983 runs raw-preflush-default-logits then raw-preflush-isolated-logits16 tokens, same original model/rawprompt/trace. Poll same handle. Source candidate, no promotion. After results isolate fusion/replay individually while holding router setting fixed and compare CPU selected tokens, profile accepted path only.

## Focused matmul-to-DMA reuse reproducer fails old, passes preflush

Build68720 complete exit0. dma-dead-mm-old replay1: disjoint rows1/9 pass32 inputs each; alias rows1/9 FAIL first comparison NMSE1.993436722/1.999957541 against stock1e-7. Runner stops after failed replay1; old replay0 not yet run. Swapped signed preflush candidate so/catalog with no inference live, signtool verifies. dma-dead-mm-preflush replay1/0 all256 comparisons pass. Installed DSP now6D48E2F..., source preflush candidate. No full-model promotion. Earlier installed-old/build-live entries superseded.

Next verify producer/direct-store and alias traces, old replay0 failure, then raw preflush default/isolated token comparisons and separate replay/fusion/router controls. This is a demonstrated narrow cache-handoff correction, not proof all model drift resolved.

## Raw isolated old-DSP matches CPU tokens; focused oracle strengthened

Handle78041 terminal exit0. CPU versus raw-old-default16-token comparison fails selected-token gate at step8 (CPU1510/old2614), prompt logits already differ. CPU versus raw-old-isolated passes all16 selected tokens; floats differ. Isolated simultaneously disables fusion/replay and selects routerHVX, so cannot attribute difference to one switch yet.

Added test_dma_dead_reuse to existing test harness (whole graph,32 changing inputs, rows1/9, disjoint/forcedalias), appended dead producer before unrelated concat DMA writer and RMS consumer. Initial SCALE producer version passes256 CPU comparisons on oldDSP; logs verify alias addresses equal and3-op replay capture/hit. SCALE itself uses DMA output, so this is not proof against dirty CPU-store reuse. Revised producer to F32 MUL_MAT; build handle68720 live, only testexe installed by script. Next poll build then run dma_dead_reuse pattern and verify actual producer/kernel/alias/no fallback. Installed oldDSP52A4... unchanged, source preflush candidate, no model live.

## Matched raw logits captures running

Extended run-decode.ps1 with default-off LogitsTrace, optional Replay/Fusion (defaults1), RouterHvx (defaultoff), retaining all normal defaults. Trace mode writes full logits/selected-token detail logs and clears tensor probes. Live handle78041 sequentially runs raw-cpu-logits, raw-old-default-logits, raw-old-isolated-logits,16 tokens each, original GGUF/raw Hello/c256. Isolated old disables fusion/replay and selects routerHVX. CPU trace noticeably slower than untraced rawCPU; diagnostic timing not a benchmark. Poll same handle; no restart or DSP swap while model live. Installed old52A4..., source preflush candidate, no promotion.

## Raw-prompt CPU reference completed: neither NPU transcript matches

Handle92024 completed exit0,16 raw tokens; log verifies0/42 GPU-offloaded layers and CPU_REPACK weights. CPU output starts Hello / I am trying to use the `get` function..., whereas prior normal oldDSP128 output says following code/PHP and preflush says AWS CDK/VPC. Thus isolated chat token agreement does not generalize to raw normal trajectory. Need matched-length token/logit captures and first-divergence localization; do not promote either transcript as CPU-equivalent. Raw CPU timing30.74tok/s is one short16-run diagnostic, not validated performance comparison. No inference remains. Installed oldDSP52A4..., source preflush candidate.

## CPU reference favors preflush at isolated token16

CPU handle16121 completes exit0:16 rows, zero GPU layers. All16 selected tokens match preflush candidate. Both rollback old-DSP runs differ from CPU at step15: CPU/candidate4602 versus old264. Cross-backend logits differ from prompt (NMSE .0280212/max1.975); do not require cross-backend float identity or claim all numerical errors solved. Comparator --require-tokens now gates complete matching selected sequences while allowing float differences; candidate exits0, old-b exits1. Candidate still NOT promoted.

Live handle92024 runs CPU raw-prompt16-token reference with original GGUF and normal completion settings, label dst-preflush-cpu-raw-reference. Added run-decode.ps1 -Device none/ngl0, defaultHTP0/ngl99 unchanged. This checks separate raw-prompt trajectory behind differing128-token normal transcripts. Installed DSP remains old52A4..., source candidate; candidate signed artifacts preserved. Poll same handle, no concurrent model benchmark.

## Rollback reproduces variation; CPU reference running

Handle84426 complete exit0. Old-DSP rollback-long-a/b16-row strict comparison exits1: prompt exact, first decode NMSE .00139511853/max .705013514; all16 selected tokens match each other. Candidate short/long repeats were bit-identical; returning variation after restoring old DSP strengthens preflush repeatability evidence, not correctness proof. Candidate versus old diverged selected token at step15.

Added run-mtp-stage.ps1 -Device none (defaultHTP0 unchanged), with device none/ngl0 for CPU reference. Live handle16121 runs dst-preflush-cpu-reference16 tokens with matching GGUF, prompt, seed/temp, context/batch and no MTP. Runtime log verifies0/42 layers offloaded, CPU/CPU_REPACK model buffers. CPU prompt first selected token9419 agrees, floating logits differ as expected across backend. Continue same handle; CPU decode is materially slower. Installed DSP remains old52A4...; source preflush candidate. No promotion.

## Destination preflush fails normal transcript gate; rollback challenge live

Handle32488 terminal exit0: normal128-token decode 28.06 tok/s /35.64 ms,126 graph reuses,1381-op replay. Trimmed transcript DOES NOT match established topk-alias-control: candidate produces repetitive AWS CDK/VPC answer versus baseline PHP PDO answer. Numerical repeatability under isolated settings is not correctness proof. No promotion or speed claim.

Saved signed candidate6D48E2F... so/catalog in dst-preflush-candidate. With no live inference restored prior52A4D490... so/catalog; signtool membership verification passes, host hashes unchanged. Source remains preflush candidate, installed DSP is OLD for challenge. Live handle84426 runs dst-preflush-rollback-long-a/b,16 tokens and same isolated settings/model as candidate pair. Poll same handle, compare old repeat logits and selected tokens. Need independent CPU/reference and production-config discrimination before interpreting changed trajectory.

## Destination preflush longer repeat passes strict identity gate

dst-preflush-long-a/b both exit0:16/16 full-vocabulary logits bit-identical and all selected tokens match. Short pair also3/3 exact. Added opt-in compare_logits_traces.py --require-identical: rejects missing/incomplete selected-token alignment, unequal row counts or differing logits/tokens; reports total compared rows. Positive short/long captures exit0, earlier cache-syncht varying pair exits1. Diagnostic trace timing is not a speed benchmark. git diff --check passes.

Live handle32488 now runs only final dst-preflush-default-regression; do not restart. Candidate DSP6D48E2F... still installed, NOT promoted. After terminal result compare text/time, preserve candidate signed artifacts and restore prior signed DSP for matching16-token challenge; variation returning would strengthen evidence. Source remains candidate until decision.

## Destination preflush: first unretained pair bit-identical

Handle83377 completed exit0. dst-preflush-plain-a/b: all three sampled full-vocabulary logits bit-identical, finite, selected-token alignment available and matching. No callback/retention/fill, fusion/replay0. First promising observation for destination preflush; one short pair is not a correctness acceptance or cause proof. Existing RMS oracle512 comparisons also passes. Installed candidate6D48E2F... remains experimental.

Live handle32488 runs dst-preflush-long-a/b (16 tokens, same isolated settings/model), then dst-preflush-default-regression (128 normal decode tokens, original GGUF). Poll same handle; compare longer logits/selected tokens and normal transcript/performance. Need rollback challenge and dedicated unrelated-buffer-reuse oracle before promotion. Earlier live83377 statement superseded.

## Destination preflush candidate installed; full-model repeat running

Added htp_tensor_flush_all for assigned destinations immediately before dirty marking in proc_op_req. Uses existing tracked ranges, no tensor-lifetime changes or new trace callbacks. Conservative cache-handoff experiment, not a proven fix. HTP outputs4 fit helper arrays sized inputs10. Built all DSP targets, installed v81 only; build75714 exit0, signed catalog membership verifies. Installed DSP SHA2566D48E2F15B0EDEAEC3AF07F3F58D2FF3B2B5E90B9C2D72183512527D8E18B930. Original signed rollback remains pre-cache-syncht.

dst-preflush-rms replay1/0: all512 changing-input CPU comparisons pass at original tolerance. Live handle83377 now runs dst-preflush-plain-a/b, F32 MTP model in non-MTP mode, three tokens, routerHVX, fusion/replay0, logits trace only, no allocation fill/retention/callback. Poll same handle, compare logits/selected tokens, then default performance. Candidate NOT promoted; earlier prior-installed state superseded.

## Cache-completion barrier rejected; signed DSP restored

Completed cache-syncht plain repeats still differ: first decode logits NMSE .00118588821 / max .625419617, second .000501184986 / max .362013459. Prompt logits bitwise equal, all three selected tokens match. Barrier does not fix variation. Candidate default regression 28.68 tok/s / 34.86 ms, 128 runs / 126 reuses, 1381-op replay; trimmed transcript equals topk-alias-control. No promotion.

Source barrier reverted; candidate so/catalog preserved in cache-syncht-candidate. Restored prior DSP so/catalog from pre-cache-syncht; signtool verifies restored catalog membership. Installed DSP SHA256 52A4D490D48524236D300F53C2A8DFDB4DD81886ED04F71BB89C9901C2453F52. Host unchanged. Rollback regression handle87737 completed exit0: 28.34 tok/s / 35.29 ms, 128 runs / 126 reuses, 1381-op replay. Single sequential regressions do not establish speed difference. Earlier candidate-installed/live18117 entries superseded.

Next hypothesis: output address reuse leaves old dirty cache contents before DMA overwrite. proc_op_req flushes sources but only marks destinations dirty. Explicit inplace oracle does not cover unrelated dead-buffer reuse. Destination preflush is a prospective conservative experiment, not a demonstrated fix.

## Cache-completion candidate installed - end-to-end gates running

Build93529 completed exit0, signed installed v81so verified with signtool catalog check. Candidate DSP SHA25644BEB26F78AF070E62B249562C22298304A23F4BA8A4AE8F81FB3969F555D869; rollback pre-cache-syncht retains original52A4D490... so/catalog and host files. RMS reuse oracle passes512 CPU comparisons at unchanged tolerance, replay settings1/0. No end-to-end fix claim.

Live handle18117 runs cache-syncht-plain-a/b without callback/retention/fill, fusion/replay0, then cache-syncht-default-regression. Continue observing this handle; do not restart. Candidate is installed for experiment, NOT promoted. Compare complete logits/selected-token alignment, default transcript and timing before deciding retention or rollback. Earlier build-pending status below is superseded.

## Latest continuation - cache-completion barrier candidate building

Selective final-attention retention: attention result exact; final-layer unused prefill rows vary, 243/246 tensor records exact. Decode outputs/logits exact under the probe; cannot claim production localization.

Found a concrete cache-handoff hypothesis: direct clean/invalidate helper lacks trailing syncht, whereas Qualcomm memory-ordering documentation and local fence read helper use it. Added trailing transaction barrier candidate, preserved signed rollback artifacts in pre-cache-syncht. Build handle93529 live, v81 linked, packaging/install pending. No inference concurrent. Next verify signed installed artifacts and run reuse oracle, unretained plain repeat, default regression. No fix/speed promotion until gates pass. Earlier binary hashes below describe the pre-candidate state.

## Latest continuation - final-block retention suppresses prefill variation

Completed retained-finalblock-a/b (previous handle 24587): all 498 tensor records and all sampled logits bitwise equal. This wider output-retention probe suppresses previously observed prefill variation too, so it does not localize the untouched failure. Retaining only final Q/gate projection plus all layer outputs (retained-final-qg-a/b, completed handle 86365) gives 245/246 exact records: Qcur_full-39 and every decode boundary exact, with only l_out-39 four-token prefill row 0 differing (max .035435915). All sampled logits/selected tokens match. One repeat pair does not establish that Q/gate lifetime causes the nine-token issue; allocation layout changes remain a confounder.

No runtime/source changes this continuation; both pairs exit zero, diff check passes, no inference remains. Current default remains 28.93 tok/s regression, not a new benchmark pair; MTP/correctness unresolved. Next retain the final attention output to test whether the remaining prefill variation survives, or implement an allocation-preserving DSP trace with explicit coherence/probe-equivalence checks. Earlier live-handle statements below are historical and superseded.

## Latest continuation - layer-38 gap and initial-buffer fill discrimination

`retained-gap38-{a,b}` retains 0-37 and 39, leaves only 38 unprotected: 231/234 tensor records equal, all captured earlier boundaries exact, final output varies in nine-token prefill row 1 and both decode steps. First decode hidden NMSE .000152, logits .0000874 / max abs .153. Observed range is 38-39 under changed allocations, not unperturbed production localization.

Added default-off GGML_HEXAGON_ALLOC_FILL / runner -AllocFill byte diagnostic. Fills newly allocated RPC memory BEFORE mapping/loading, without changing graph lifetimes or adding synchronization. This affects initial bytes, including guard/padding/weight buffers, and startup traffic; it is not a production fix. `alloc-zero-{a,b}` proves byte-zero fill executed, but first decode logits still differ NMSE .00131853 / max .643049. Prompt logits and all three selected tokens match. Initial allocation contents are not a sufficient explanation.

Post-build default regression: 28.93 tok/s, 34.57 ms/token, 128 runs, 126 graph reuses, 1381-op replay, exact baseline transcript match, zero fill markers. Single regression, no speed promotion. Host DLL SHA256 now EDB9FF528857E8D0BB0622343549891FA576BD529485D6A362D2176D0072DC0D; relinked llama.dll 981E8D385E324BB3474E632A68CC0D03F3B65DC035067DEB663042EBD7009BAB. DSP unchanged (52A4D490...). Live process modules verified pkg-wos host/llama/OpenCL paths. Default runners clear fill flag. Next final-block retained intermediates probe is running; session 24587, labels retained-finalblock-a/b. Do not restart while live.

## Latest continuation - observed range 37-39; small reuse oracle passes

All-layer retention repeat `retained-thirtynine-{a,b}` again gives bitwise-equal decode outputs/logits, with two differing unused prefill rows (238/240 records equal). Despite the label, its regex retains ALL layers 0-39; it is a replication, not a narrower final-block localization. `retained-thirtyseven-{a,b}` retains 0-36 plus 39, leaves 37/38 unprotected, and preserves final decode variation: 224/228 records equal, first hidden NMSE .000134 and logits .0000685. Earlier retained boundaries are exact; observed range is 37-39 under changed allocations. All selected tokens match.

Added existing-harness test_rms_reuse_chain and run-rms-reuse-oracle.ps1: three RMS -> broadcast multiply -> residual add stages, disjoint/inplace arms, F32 widths 2048 and 256x16, token widths 1/9, 32 deterministic changing inputs. All 512 CPU comparisons pass at stock 1e-7 NMSE, replay setting on/off, fusion off. Logs prove inplace output shares the initial work tensor address; width-one graphs capture/hit 10-op replay. Width-nine arms are repeated tests but not replay-capture proof. This excludes this small explicit alias chain as a reproducer, not general allocator/KV correctness.

Only test executable rebuilt/copied; installed host/DSP/llama runtime unchanged. Model probes and oracle exit zero; script parse and diff checks pass; no inference/test remains. Next narrow retained gap to layer 38 and construct a late-block/state test or allocation-preserving snapshot. No speed or correctness promotion.

## Latest continuation - observed late-layer range 35-39; attention HMX not necessary

`retained-thirtyfive-{a,b}` protects layers 0-34 plus 39: all protected earlier boundaries match bitwise, while both final-layer decode outputs differ. 213/216 tensor records match; first decode hidden NMSE .0000587 and logits .0000339. Observed variation now lies in layers 35-39 UNDER THIS ALLOCATION-CHANGING PROBE, not a production-equivalent localization.

Independent `attention-hvx-{a,b}` pair uses no tensor retention/callback, disables fusion/replay, and forces only full-attention off HMX. All profiled attention operations report HVX. First decode logits still vary (NMSE .000578, max abs .502); prompt logits and three selected tokens match. Attention HMX is not necessary for the observed variation. Final decode: 2235 ops, op sums 40.675/40.531 ms; attention totals .414/.408 ms, head 2.929 ms each. Heavy diagnostic logging makes wall-clock rates unsuitable as benchmarks. No kernel fix or speed promotion.

Runner now exposes -FaSelect/-GdnSelect with explicit defaults 2; default benchmark resets both to 2 so diagnostics cannot leak. Runtime binaries unchanged. Four runs exited zero, PowerShell parse/diff checks pass, no inference remains. Next: retain through layer 38 and isolate the final block/reuse chain, then allocation-preserving correctness tests.

## Latest continuation - variation survives retention through layer 29

Repeated three-token plain probes retain layers 0-9, 0-19, and 0-29 plus layer 39, with replay/fusion disabled and router HVX on the expanded model. Every retained early boundary is bitwise repeatable in each pair; final-layer decode outputs and logits still vary. The 0-29 pair has 183/186 matching tensor records, first decode hidden NMSE .000248 and logits .000126 (max abs .185). Prompt sampled logits and all three selected tokens match. This narrows the observed variation to layers 30-39 UNDER THE PROBE, not untouched production: retention changes allocations and may hide earlier defects.

Read the full DSP dirty-range implementation and cache-line flush helper. No demonstrated bookkeeping defect: interior clean ranges remain conservatively dirty and flushes round to cache lines. Destination reuse before DMA output remains an untested hypothesis, not a fix. No runtime code/binary changes this continuation; all six runs exited zero, diff checks pass, no inference remains. Next: late-layer retention discrimination and an allocation-preserving kernel/reuse oracle. MTP and numerical correctness remain unresolved; no speed claim.

## Latest continuation - whole-batch retained probe, allocation sensitivity

Added default-off retained-output tracing: selected tensors stay live until the graph completes, then are read once after a graph synchronization. No per-layer callbacks/cuts or extra copy ops. Device profile confirms one complete HTP submission per microbatch. It still changes allocator reuse and must not be treated as equivalent to untouched execution.

All-layer retention produces bit-repeatable decode outputs/logits in the repeated three-token pair, while some non-final prefill rows remain unstable. Retaining ONLY l_out-39 preserves variation (first decode output NMSE .00335; logits .00121). This rules out interpreting the all-layer stable pair as a final-normalization fix and strengthens lifetime/allocation sensitivity as a diagnostic lead. Next narrow retained ranges or use an allocation-preserving DSP snapshot. All probe settings remain opt-in; no correctness fix or speed improvement claimed.

Installed llama.dll SHA256 `342ED94B44FBE60546D7B5D2B1AA14EF8D7ABC3BEC956048AF7874FB5D0C3956`; Hexagon host and DSP unchanged. Default runner clears retained/callback/logit probes. See FINDINGS.md for complete evidence and limitations.

Post-rebuild default regression passes: 28.77 tok/s, 1381-op replay, exact earlier baseline transcript match, 128 runs. Not a new record pair. All diagnostics exited zero, checks pass, and no inference process remains.

## Latest continuation - selective intermediate probe narrows the variation

Default-off filtered tensor trace now captures full logical intermediates with type/dimensions/strides. Early-layer repeated probe: all 648 records in layers 0/1 match bitwise, while first decode logits still vary (NMSE .000736). Layer-boundary probe: decode outputs/logits become bit-identical, indicating synchronization/batching sensitivity; one prefill row in l_out-39 differs (.0354 max abs), with its final row exact. These probes perturb scheduling and are diagnostic evidence, not a production fix or proof that a particular layer/kernel is at fault.

Tools: run-mtp-stage.ps1 -TensorTrace -TensorFilter <regex> -LogitsTrace -Fusion 0 -Replay 0; compare_tensor_traces.py [--row-stats]. Next probe is final hidden/head only, then a preserved-batch or DSP-side capture to avoid suppressing the bug. Plain transcript gates alone remain insufficient. Hexagon/DSP binaries are unchanged; common library contains the default-off callback. See FINDINGS.md for exact captures and limitations.

Head-only probe reproduces differing hidden input before the LM head (first decode NMSE .00466), with device output logits also differing. Nine-token prefill has an unstable non-final hidden row (max abs 4.61), while its final row stays exact. Next priority: an in-batch retained-copy or DSP snapshot that avoids callback synchronization cuts, with explicit probe-equivalence validation. All runs exited zero and no inference process remains. No numerical fix or performance improvement is claimed from these probes.

## Latest continuation - numerical variation also occurs in plain decode

Added opt-in full logit-row capture at the existing sampling synchronization point, without per-op graph splitting. Plain/MTP differences are present in prompt logits before acceptance (NMSE 0.00174), but the plain repeat also differs from itself on the first decode (NMSE 0.00111), despite identical complete text. Replay-off short repeats reproduce the first-decode variation (NMSE 0.00130) with bitwise identical prompt logits and matching selected-token prefixes. Therefore neither MTP rollback nor descriptor replay alone explains the observed drift.

Tools: run-mtp-stage.ps1 -LogitsTrace [-Replay 0] [-Fusion 0], compare_logits_traces.py. Raw argmax may be masked by samplers and must not be mistaken for selected-token divergence; newer traces log selected tokens and use them for prefix alignment. Trace rates are diagnostics only. The default trace is off, main decode resets all diagnostic flags. Runtime Hexagon/DSP hashes unchanged; rebuilt common library contains the trace. Next work is fusion-off repeatability and locating the first divergent state/tensor while preserving the relevant production fuse.

Fusion-off + replay-off repeats also vary on first decode (NMSE 0.00122, max abs 0.759), while prompt logits and all eight selected tokens match. This rules out either mechanism as the sole cause. Next isolate recurrent-state inputs and underlying one-token kernels. All trace runs exited zero; no inference process remains. No performance promotion follows from traced timings.

## Latest continuation - exact router IDs fail on HMX prefill

The stronger dot -> softmax -> ordered top-8 -> norm oracle finds an expert-ID mismatch at token width 9, seed 853810: element 47 is HTP 98 versus CPU 234. Widths 1/2 pass. Failure reproduces with replay off, and the kernel trace selects HMX with FP16 input conversion. This contradicts treating standalone F32 matmul tolerance passes as proof of router correctness.

New default-off `GGML_HEXAGON_ROUTER_HVX=1` keeps only F32 router logits matmuls off HMX. All 192 chain comparisons pass with targeted selection, replay on/off; tolerances unchanged. Layer-40 nine-token router profile is 56 us HVX versus 40 us HMX, while one/two-token routers remain ~20 us. MTP rate pair on the expanded model is 26.62 control / 26.44 candidate, with differing text. Do not enable by default or claim MTP correct. Reproduce using run-mtp-router-chain.ps1 (-RouterHvx for candidate, -MmSelect 1 for global-HVX diagnostic).

Installed host SHA256 is now `2823FD01D22F7EDDF095DBFE4D1DF8B2E14EBC45A0DD16C04562A2AEFE37E654`; DSP remains `52A4D490...`, OpenCL stub retained. This supersedes earlier host hashes. Next priority remains first-divergent target execution and explicit speculative accepted/rejected-prefix cache checks; the synthetic router failure has been isolated and an opt-in remedy implemented, not an end-to-end MTP fix.

Default decode regression passed: 28.72 tok/s, 1381-op replay, exact trimmed-text match to earlier topk-alias-control, 128 tokens. Existing direct-logit alias oracle scope remains three cases and all 96 changing-input comparisons pass. No inference/test process remains.

## Latest continuation - MTP gate placement improvement, still experimental

Created separate `models/gguf/qwen36-35b-a3b-q4_0/Qwen_Qwen3.6-35B-A3B-Q4_0-mtp-f32.gguf` (20,837,295,744 bytes). Only the layer-40 router and shared-expert gate are expanded BF16 -> F32, with exact value preservation. The current original model is untouched. Independent readback verifies 20,824,199,680 unchanged payload bytes and all 526,336 expanded values. Tools: build-expand-mtp.cmd, expand-mtp-bf16.cpp, verify-mtp-expansion.cpp. The decode-stage runner accepts -ModelPath to select the separate experiment.

The candidate removes both CPU boundaries: draft scheduler splits drop from five to one, allowing single-submission graph replay. Full draft device work remains ~5.3 ms, catch-up ~1.5 ms, so the gain comes mainly from placement/round-trip overhead. Clean serialized BF16 control / F32 candidate: 19.59 / 26.39 tok/s, both 27/36 drafts accepted. Earlier candidate repeats at 26.46 with exact matching candidate text. But candidate/control and candidate/plain transcripts differ; no MTP correctness or accepted performance claim. Keep the original model as default, KV-only and fused verify off.

Gate arithmetic tests cover router/shared dot shapes at token widths 1, 2, 9 with BF16-representable F32 weights and 32 changing inputs per shape; 384 CPU comparisons passed across replay on/off at stock NMSE 5e-4. Whole-graph logs prove replay for widths 1/2; width 9 does not capture. Host-stage diagnostic medians: draft 6.045 ms, two-token catch-up 2.223 ms, target 56.731 ms. This is not exact-ID or KV ownership validation. Next work: dot-to-router ordered-ID oracle, first-divergent target tensors for plain versus MTP (two-token verification remains ~54 ms on DSP), and explicit accepted/rejected-prefix cache tests. See FINDINGS.md for evidence and limitations. No inference/test process remains.

## Latest continuation - MTP stage profiling, correctness still open

Added opt-in `LLAMA_MTP_PROFILE` host-stage timers and `run-mtp-stage.ps1`, `parse_mtp_stages.py`, and `compare_cli_transcripts.py`. Profile runs synchronize and are diagnostics, not accepted throughput measurements. The timer requires `--perf` and includes both prompt/eval counters; the first profiling log had zero target times and is invalid for that stage. Startup counter deltas are not steady-state target measurements.

With independent draft KV and all experimental fuses off, plain CLI decode reached 28.34 tok/s; standard MTP reached 19.46 tok/s but differed from the plain greedy transcript at character 53. MTP is not target-equivalence validated. Shared-cell fused verify remains unset.

The experimental, default-off `LLAMA_MTP_KV_ONLY=1` retains attention/KV writes in zero-output catch-up but omits the downstream attention projection/residual/FFN/hidden output. Same-binary 64-token control/candidate rates were 20.43 / 21.96 tok/s, but transcripts diverged at character 164. Do not promote this speed result or enable the flag by default. A KV tensor oracle and accepted/rejected-prefix tests remain required.

Synchronized 32-token diagnostics: two-token catch-up median 15.214 ms control versus 1.220 ms candidate; target verification medians 56.142 / 55.738 ms. Draft decode medians were 13.171 / 20.472 ms, so reduced catch-up cost partially moves into later work. Separate eight-token DSP traces show normal catch-up as three HTP submissions (~1.6 ms combined), versus one (~0.6 ms) in the candidate. Catch-up already avoids the LM head. The full draft has a ~2.94 ms head and ~0.99 ms TOP_K. Two-token target verification costs ~54 ms on DSP.

The loader reports two BF16 layer-40 weights, including `blk.40.ffn_gate_inp.weight`, on CPU. Draft reservation has five scheduler splits (three HTP segments), whereas target reservation has one. Investigate these CPU boundaries before assuming descriptor/buffer-capacity flushes. BF16-to-F32 value-preserving expansion is a possible next experiment, not yet implemented or validated. Current rebuilt llama/common profiling code leaves timers and KV-only behavior disabled by default; DSP and Hexagon host hashes remain as recorded below.

## Latest continuation - top-k alias executes, still opt-in

The installed v81 skel is now 848432 bytes, SHA256 `52A4D490D48524236D300F53C2A8DFDB4DD81886ED04F71BB89C9901C2453F52`; `signtool verify /pa /c` verified its installed catalog. Host `ggml-hexagon.dll` SHA256 is `58035981BB3828486D718361F186A80DE43AE862D77989556CB05C0BAC664648`. The previous 844176-byte PHP skel and catalog remain in the campaign for rollback. The OpenCL stub remains installed. This supersedes the installed-file description below.

`GGML_HEXAGON_MOE_TOPK_ALIAS=1` permits logits/output aliasing in the router fuse. Default/unset keeps the previous overlap check. The DSP copies logits into scratch before stores. The live output/output overlap check remains unconditional. The 40-layer candidate runs without NO-SUPPORT and uses 1264 steady ops versus 1381 with the flag off. SHEXP_EP, GDN-state, RMS-scale, and fused MTP verify remain off.

Same frozen host/skel, seed 1234, temp 0, Hello, 128 generated tokens:

| Run | tok/s | ms/token | Transcript |
| --- | ---: | ---: | --- |
| `topk-alias-control` | 28.70 | 34.85 | PHP through "You have an" |
| `topk-alias-candidate2` | 28.95 | 34.55 | Exact trimmed-text control match |
| `topk-alias-candidate3` | 29.00 | 34.48 | Exact trimmed-text control match |
| `topk-alias-control2` | 28.96 | 34.53 | Exact trimmed-text control match |

The first all-layer trial (`htp-qwen-topk-alias`, before adding the environment gate) also reached 29.00 but inserted `password` into the PDO constructor and differs from its 28.59 control. Do not call all-layer top-k correctness proven. The later matching pair does not erase that mismatch. The candidate/control speed bands overlap. The conservative matched flag-off pair is now 28.70 tok/s, superseding 28.42 as a measured control pair, not evidence of an alias-fusion speed win.

The router oracle now covers disjoint storage, logits/weight overlap, and logits/index overlap. Each case uses 32 deterministic changing inputs (mt19937 seeds `0x51A17 + iteration`). Expert IDs must match CPU exactly and weights use the existing CPU NMSE tolerance in their original order. The test graph gets a stable unique nonzero ID so Hexagon actually captures/replays it. All 288 comparisons passed across fused+replay, overlap-fuse-off+replay, and fused+no-replay. Pointer logs verify the physical overlaps, and graph logs show 2-op fused capture versus 5-op control capture for aliased cases. Reproduce with `build-router-test.cmd` then `run-router-oracle.ps1`. Logs: `topk-oracle-{fused-replay,control-replay,fused-no-replay}.log[.err]`.

The first broader frozen-binary gate passed exact trimmed-text SHA256 comparisons on code, arithmetic, and prose. Each requests 64 generated tokens and measures 63 single-token decode evaluations (the first token comes from prefill). `run-router-gate.ps1 -Prefix <new-prefix>` reproduces the pairs. Rates are not 128-token Hello record claims.

| Prompt | Control tok/s | Candidate tok/s | Control/candidate ms/token |
| --- | ---: | ---: | --- |
| Python Fibonacci function | 28.66 | 29.20 | 34.89 / 34.25 |
| 17 times 23 | 28.78 | 29.26 | 34.75 / 34.18 |
| Why the sky is blue | 28.72 | 29.00 | 34.81 / 34.48 |

All six runs exited 0, reused 62 graphs, and logged 1381/1264-op control/candidate replay. These broaden the evidence for the candidate but do not explain the original password mismatch. Keep the flag opt-in pending longer/deeper-context gates and first-divergent-tensor evidence if the mismatch recurs. MTP correctness and speed remain open.

Warm profile pair: `topk-alias-control-profile` versus `topk-alias-profile`, four generated tokens, PROFILE=1. The last DSP batches are 32.881 versus 32.389 ms; op sums are 32.017 versus 31.588 ms; batch-minus-op-sum is 0.864 versus 0.801 ms. Candidate MOE+TOPK is 40 ops / 610 us. These debug runs are not throughput measurements. See FINDINGS.md for the breakdown and next work. `run-decode.ps1` reproduces a fixed-seed control; `-TopkAlias` selects the candidate, `-Tokens 4 -Profile 1` selects profiling. No inference process was left running after this continuation.

## Earlier campaign record

Updated: 2026-09-26 19:55 local. The Qwen Q4_0 GGUF on disk has had its 60 Q5_0 shared-expert tensors requantized to Q5_K in place. Restore from `npu_campaign/q5_0_shexp_backup.bin` with `requant-shexp.exe --restore` before treating that file as the original quant. Best measured decode is 28.42 tok/s over 128 tokens: 28.46 and 28.42 (`htp-qwen-topk-revert.log`, `htp-qwen-topk-revert2.log`), 35.14 and 35.19 ms, both PHP through "You have an". That pair replaces 28.38 and 28.40 (`htp-qwen-replay.log`, `htp-qwen-replay2.log`). Same 1381-op graph. Not a new fuse. The GDN state-gather fuse did run (1391 ops, 28.59 tok/s, `htp-qwen-baseline2.log`) and printed the aws-cdk fragment, so the matcher is not called. Do not enable it. The expert-weight mul is now folded into `ADD+NW` and the text is still PHP (`htp-qwen-addnw.log`, 28.37 tok/s, 35.24 ms, 1381 ops). That does not beat the 28.42 pair. `SHEXP+EP` (SwiGLU + shared-expert down + the f32 gate dot folded into the sigmoid epilogue) printed a different INSERT (`htp-qwen-shexpep.log`, email column, 28.93 tok/s, 34.57 ms, 1261 ops) and is not called. The installed skel is the restored PHP file, `pkg-wos/lib/libggml-htp-v81.so`, 844176 bytes, with the catalog that verifies it (`npu_campaign/libggml-htp-php-844176.cat`). Confirm run `htp-qwen-shexpep-off.log` is PHP through "You have an", 28.62 tok/s, 34.94 ms, 1381 ops. One run, not a pair. 28.4 stands. The v81 prefix so is the unused 848432-byte `SHEXP+EP` link. The previous PHP skel is still `npu_campaign/libggml-htp-v81-php-839952.so` (839952 bytes). Fuse-off rerun `htp-qwen-gdn-off.log` is PHP at 27.76 tok/s and does not replace 28.4. With f16 KV and a prefix already cached, tg64 is 28.34 at depth 0, 26.41 at 2048, and 25.11 at 8192 (`bench-depth.log`). Graph replay (`GGML_HEXAGON_REPLAY`, default on) submits the captured single-flush command again. The previous 27.6 pair was 36.21 and 36.36 ms; this pair is 35.23 and 35.21 ms. Router top-k is now fused (`MOE+TOPK`) and the 32-token text is still the PHP PDO fragment: 27.23 tok/s, `htp-qwen-topk.log`. Two 128-token repeats are 27.71 and 27.57 (`htp-qwen-topk-n128.log`, `htp-qwen-topk-n128b.log`). They overlap the 27.6 band, so they are not a new best. The 26.92 tok/s figure is the best 32-token average from before the router-norm fuse and includes two cold graphs. The router-norm fuse is on and still prints PHP: 26.60, 27.86, and 26.37 tok/s (`htp-qwen-wnorm.log`, `htp-qwen-wnorm2.log`, `htp-qwen-wnorm3.log`). 27.86 did not repeat. The RMS matcher stays off. The repro pair with that matcher off was 26.88 and 26.68 (`htp-qwen-repro2.log`, `htp-qwen-repro.log`). The ADD+N number was 26.41, log `htp-qwen-addn2.log`. The conv-window number was 25.79, log `htp-qwen-convwin-final.log`. The embedding is on HTP0 (`splits = 1`). MTP verify is slower than plain decode: 18.94 tok/s over 64 tokens, acceptance 0.75, log `htp-qwen-mtp3.log.err`. `LLAMA_FUSE_MTP_VERIFY=1` appended blk.40 attention to the target graph and aliased that KV into the draft context. Two temp-0 Hello runs disagreed: `htp-qwen-mtp-fuse.log` keeps "a question" at 22.2 tok/s, `htp-qwen-mtp-fuse2.log` says "questions" at 21.3 tok/s. Leave the variable unset. Flag-off PHP is `htp-qwen-fuse-off.log`, "You have an", 28.56 tok/s, 35.01 ms, 1381 ops. One run. 28.4 stands.

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
- Router norm and the expert-weight mul are fused. Router top-k fuses layer 39 only; fusing the other 39 aborts on this skel (Step 16). The shared-expert down epilogue (`SHEXP+EP`) is faster and changes the INSERT. Do not call it. Still open from the 17:40 token: the q/k `RMS_NORM`+`SCALE` pairs (634 µs, matcher stays off) and the 2.96 ms output projection. Do not requant `output.weight`. The sub-8 µs remainder is about 4 ms and is scattered.
- MTP accepts drafts (0.75, mean length 1.75) but a 2-token verify costs about two single-token forwards. Plain decode is faster. Log `htp-qwen-mtp3.log.err`, 18.94 tok/s. `LLAMA_FUSE_MTP_VERIFY=1` drifted (see Step 15). Leave it unset.

## Step 15 — fused MTP verify tail, off

- State: measured, off. `LLAMA_FUSE_MTP_VERIFY=1` appends blk.40 attention to the target graph and aliases that layer's KV into the draft context. The catch-up `llama_decode` is skipped because `ctx_other` is set. The MoE and the MTP LM head stay on `draft()`.
- `htp-qwen-mtp-fuse.log`: temp-0 Hello keeps "a question". Banner Generation 22.2 t/s.
- `htp-qwen-mtp-fuse2.log`: the same command says "questions". Banner Generation 21.3 t/s.
- The pair does not agree. Do not set the variable. `run-mtp-fuse.cmd` is the reproducer.
- Flag off, log `htp-qwen-fuse-off.log`: PHP through "You have an", replay hit 1381 ops, 28.56 tok/s, 35.01 ms. One run. It does not replace the 28.42 pair. Installed skel remains 844176 bytes. `absorb_shexp_down` is not called.
- The token embedding is on HTP0. `splits = 1`. One sched copy remains (logits).
- DMA skel open returns `512` (`AEE_ERPC`) once the library path is visible. No DSP GB/s yet.

## Step 16 — router top-k is one layer, alias fuse aborted

- State: measured, not shipped. The steady graph's `MOE+TOPK` is layer 39 only. Layers 0–38 stay softmax + argsort + get_rows + `MOE+WNORM`. The skip is `overlap`: the 256-wide logits row shares its buffer with an output.
- `op_moe_topk` copies logits to scratch before either store, so that alias is safe, but the installed 844176 skel returns `NO-SUPPORT` when `src` equals either destination. Fusing all 40 rows aborted. Log `htp-qwen-topk-all.log.err`: count 40, then `dsp-error NO-SUPPORT`. No completion text.
- The host overlap check is back. Do not drop it while this skel is installed. `argsort-ops.c` in the tree now allows a logits alias. The skel rebuilt from that source is still 848432 bytes (sha256 `52A4D490D48524236D300F53C2A8DFDB4DD81886ED04F71BB89C9901C2453F52`, not the older `FD3AE079…` SHEXP link) and was not installed.
- Confirm pair on the restored host and the 844176 skel: PHP through "You have an", no email column, replay hit 1381 ops.
  - `htp-qwen-topk-revert.log`: **28.46 tok/s**, 35.14 ms/token. Seed not required. Graphs reused 126/128.
  - `htp-qwen-topk-revert2.log`: **28.42 tok/s**, 35.19 ms/token. Graphs reused 126/128.
- 35.19 and 35.14 both beat 35.21. **28.42 stands.** `absorb_shexp_down` is not called. `pkg-wos/bin/ggml-opencl.dll` is the stub, 102400 bytes. `HTP_OP_MAX_BUFS` is 64. The Q5_0 backup was not restored.
