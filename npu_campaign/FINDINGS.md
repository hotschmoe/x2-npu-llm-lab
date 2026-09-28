# NPU campaign findings

## Harness format choice, speed context, and promising revisits (research index)

The detailed custom Hexagon harness research/to-do, including symmetric INT4, W4A8/W4A16, Intel AutoRound INT4, Q4_K/IQ4_NL, MXFP4/NVFP4, speculative packing/kernel ideas, and a gated MTP revisit, is in [HEXAGON_HARNESS_RESEARCH.md](HEXAGON_HARNESS_RESEARCH.md). Its next research tracks are (1) Qualcomm's documented versus actually executed operand formats, widths, and layouts, (2) our own quantization versus lossless DSP-native repacking, and (3) a checkpoint-pinned Qwen/Gemma/Nemotron MoE shape and traffic comparison. This is a future research lane; the testing goal is paused and no format conversion or harness implementation has begun.

Important correction to the earlier conversational summary: the DSP **Q4_0 packed** 32x32 tile is576bytes, equal to GGUF Q4_0's4.5bits/weight; its **aligned destination** is640bytes. Q4_1 packed is640bytes. There is no demonstrated 11% Q4_0 model-storage saving from merely replacing its present tile with generic INT4. The real options are lower unpack/scaling cost, better actual transfer/alignment behavior, or a *different quantization* with separately measured accuracy. AutoRound is offline rounding/scale optimization, not a Hexagon hardware auto-round instruction.

Speed numbers should retain their test context. The older fast path produced reproducible~28.4tok/s full128 decode and later~28.7-29.0tok/s candidate/control bands, but full-logit repeat/cross-backend coherence was not universally qualified. A17-token code prompt reported72.04prefill tok/s; one-repetition `llama-bench` pp4 gave51.90tok/s, neither a long-prompt prefill record. The newer strict arithmetic-diagnostic path gives a measured exact-output Q8_0 conversion win, 13.02/13.09 ->14.22/14.23tok/s in no-logit-trace A-B-B-A, while its diagnostic13-token prefill still takes~131seconds; these two backend paths are not apples-to-apples. CPU-vs-NPU first greedy divergence remains step93 on the strict branch, and MTP target equivalence remains open.

Promising but unaccepted leads to revisit with matched exact-logit/first-divergent-tensor gates: GDN state-gather fusion (~30 state copies), shared-expert SHEXP+EP fusion (~120 ops), router top-k aliasing (117 ops), and MTP lossless BF16-to-F32 draft gate expansion (19.59 ->26.39tok/s but still slower than plain and text-different). Early transcript differences do not prove the respective fusions are faulty because later plain repeated runs also showed allocation-sensitive numerical variation with fusion/replay OFF. CPU-style Q8_1 fixed a sampled expert-down tensor but slowed decode and shortened the CPU greedy prefix93 ->60, making it a correctness-factorial control rather than a speed candidate. The first approximate Q8_0 rounding dead end was already revisited and made exact, producing the current measured strict-branch decode win.

## Fast Q8_0 survives a second prompt and untraced A-B-B-A

A separate 64-token eclipse prompt gives all64 candidate logits/tokens bit-identical to scalar control and13.80 versus12.71tok/s. No-logit-trace 128-token `Hello` A-B-B-A decode gives scalar76.37/76.81ms per token and fast70.26/70.32ms; the mean latency reduction is~8.23% and displayed throughput rises13.055 ->14.225tok/s (~8.96%). Each arm is a fresh process and all four generated transcripts match, but prompt/model startup (~131seconds) is a separate, still unresolved bottleneck and is excluded from these decode rates. Installed baseline was restored after each arm; signed fast archive remains available. This strengthens the candidate as the current decode speed lead without claiming universal exactness, MTP readiness, or production promotion.

## Exact-bit HVX Q8_0 activation conversion is the new speed lead

The complete v81 thread trace put F32-to-Q8_0 conversion in one-token fused MoE gate/up at about40-42us per worker, with one shared K2048 activation row and eight quantization workers. An OFF-default HVX conversion preserves the existing per-block F32 max, scale, and reciprocal but vectorizes code rounding and packing. Direct HVX conversion truncates and is wrong (1009/2048 codes); signed-half bias still misrounds F32 values immediately below +/-0.5 (11/2048 synthetic codes). Integral/fraction compare with explicit half-away rounding survives both the frozen model rows and the constructed boundary fixture, byte-for-byte through the full replicated tile.

On signed native runs, 25 matched layer-0 retained frames, two-token logits, four-token profiled logits, and two independent full128 logit/token streams are bit-identical to the scalar-hoist control. Profiled complete one-token DSP batches drop75,896us ->70,461us mean (~7.16%); traced full128 decode improves12.74/12.86 ->13.84/13.78tok/s. These are distinct evidence types: simulator cycles locate the quantizer opportunity, native profile measures DSP batch impact, and long-run traces establish output identity on this workload. Not yet a production promotion: independent workloads and untraced A/B rates remain, and the original CPU-vs-NPU selected-token divergence remains unchanged by exact-bit speed work. Signed archive and flag-off source remain reproducible.

## CPU-style Q8_1 expert-down is a verified local correction but full-model dead end

An OFF-default F32 Q8_1 tiled quantizer compiled for v81 matches the installed CPU Q8_1 packed bytes exactly on both frozen layer-0 one-token SwiGLU captures: 0/4096 code differences and 0/512 scale/sum-byte differences. In the actual signed NPU run, all 30 retained layer-0 frames align with the scalar-hoist control and only the four small-token expert-down frames change; the T9 prompt down frame and every upstream frame stay bit-identical. At T1, CPU-vs-NPU expert-down NMSE improves from1.4026357e-5 to4.64301155e-14/maxabs2.60770321e-8; native CPU Q4_1/Q8_1 frozen oracle independently agrees, and rotated expert IDs fail unchanged threshold (NMSE1.434296).

But the full model gets worse by the campaign's correctness criterion: two complete128-token NPU runs are bit-identical to each other, yet each first differs from the independent CPU reference at step60, versus step93 for the prior control. First logit bits differ atstep0; the local down correction does not make complete outputs identical. Instrumented replay0 complete one-token DSP batches rise from75,896us to77,133us mean; expert-down summed across40layers rises6,378us to6,827us. Traced full128 rates are12.64/12.72tok/s versus prior12.74/12.86, not a decode gain. This OFF-default diagnostic is rejected for promotion; its value is proving a local arithmetic cause and ruling out a naive CPU-contract substitution as the route to better full-model correctness. Baseline three installed files and OFF build configuration were verified restored; archive remains for reproducibility.

## Exact v81 Q8_1 quantizer explains sampled one-token expert-down gap

Layer-0 paired CPU/NPU `Hello` captures have bit-identical MoE expert IDs in all five captured occurrences. At one-token decode, the preceding SwiGLU tensor is nearly identical across backends (NMSE4.33887726e-13), but Q4_1 expert-down differs (NMSE1.4026357e-5). The installed CPU Q8_1 quantizer plus Q4_1 dot1 reproduces CPU expert-down exactly; the same primitive on NPU's own inputs misses by NMSE1.40309456e-5.

A standalone v81 simulator harness calls the existing `quantize_row_f32_q8_1_tiled` source on the NPU's exact frozen eight K512 SwiGLU rows. The current quantizer header and `matmul-ops.c` are SHA-identical to the signed scalar-hoist capture archive. The simulator's 163840-byte tiled output repeats bit-exact under separate builds, including the candidate's vectorization/LTO/xqf/O2 flags. Strict unpacking verifies every replicated code, scale, and affine-sum lane. Feeding these simulator-produced Q8_1 blocks through the installed CPU Q4_1 dot1 yields NMSE4.70744871e-14/maxabs2.70083547e-8 against the NPU's own 8x2048 captured expert-down output, versus NMSE1.39024114e-5 against the CPU's own output using its frozen inputs. On the NPU input, simulator Q8_1 differs from CPU Q8_1 in42/4096 activation codes and80/512 scale/sum bytes. Rotating expert IDs with packed activations unchanged fails the fixed1e-7 gate (NMSE1.43423959). This strongly localizes the sampled decode difference to the quantizer arithmetic contract, not selected expert IDs; it does not physically prove runtime packed bytes or isolate the later first-token divergence at step93.

This is a correctness lead, not a speed gain. Existing matched instrumented replay0 decode has40 expert-gate `MUL_MAT_ID_NX` operations summing~7.03ms and40 expert-down `MUL_MAT_ID` operations summing~6.38ms of a~75.9ms batch. Next controlled option is an OFF-default Q8_1 CPU-contract diagnostic for Q4_1 expert-down, tested with frozen tensors, negative map, full128 exact same-backend bits, CPU first-divergence index, and measured decode cost. Do not promote solely because it narrows a local tensor difference.

## Vector RMS efficiency signal repeats with all128 output bits preserved

B090fullpairbothnative0, all128logits/tokens identicalbetweenarmsand69F2control. Traceddecode12.83/12.89tok/s versuspreceding10.95/10.98, matchingprofiledirection. BothCPUchecks remainfirstdifference93, so optimizationdoesnotextendCPUagreement. Baseline3files independentlyrestored. This is reproducible diagnostic-efficiency evidence, notindependentABBA/untracedproductionqualification. Huge~337secondprefill remainsnext efficiency target.

## Vector RMS arm A preserves128-row output with a traced decode gain

B090armAall128logitbitsandtokens identicalto69F2. Traceddecode12.83tok/s versuspreceding10.95/10.98, consistentwithmeasuredRMSprofilegain. CPUfirstselecteddifference remains93, no numericalqualificationextension. RepeatarmB and independentbenchmarkpending; huge337.35secondprefill unchanged. This is usefulefficiency evidence withoutpromotion.

## Vector RMS output preserves sampled logits and materially reduces measured category cost

B090four-rowprofilealllogitbits/tokens identicalto69F2. 2048-widefusedRMS summedcost21.997->8.568ms (~61%lower), completeDSPdecodebatch88.434->74.857ms (~15%lower) inoneorderedprofilepair. Headcost~3.066msunchanged. Thisisameasuredruntimegainwithsampledoutputpreservation, notmerelyanassemblyprediction; full128equivalence/repetition andindependentrepeatedbenchmark stillpending. Huge337.69secondprefillunchanged. NoCPUcorrectnessorproductionpromotionclaim.

## Register-barrier experiment closes with complete output equivalence

69F2 both full128 runs are bit-identical to each other and archived8DC1. CPU first disagreement remains93. Fused RMS reuse explicitly executes RMS_NORM+MUL and passes16cases replayon/off, supplementing16priorunfusedcases. Thus replacing volatile stack barriers with tiedregister barriers preserves these tested outputs; it is not universal arithmetic proof. Traced10.95/10.98tok/s and smallprofilegain remainunqualifiedspeed signals. Baseline independently restored. LargeRMSdecodecost remains, motivating nativeSFoutputvectorization ratherthan claimingstacktrafficwassolebottleneck.

## RMS register barrier preserves complete arm A output

69F2 arm A is bit-identical to archived8DC1 across all128 logits and selected tokens. CPU first disagreement remains93, so this arithmetic-equivalent efficiency experiment does not extend correctness. ArmBrepeat pending. Traced10.95tok/s versus prior10.85/10.86 is not independent replicated speedup evidence. Four-row RMS profile signal remains small; no promotion.

## Register rounding barriers preserve sampled logits but yield only a small timing signal

69F2 four-row fused model profile is bit-identical to 8DC1. Width-2048 RMS summed cost falls 22.725 -> 21.997 ms (~3.21%) in one ordered profile pair; total steady DSP batch mean 88.724 -> 88.434 ms (~0.33%) with overlapping ranges. Stack-round-trip removal alone does not explain or solve the large remaining diagnostic decode cost. No independent repeated speedup claim. Full128 equivalence/repetition and CPU gates pending; defaults remain OFF and no promotion.

## Width-specific normalization profile narrows decode efficiency target

Of 23.72 ms fused RMS cost per steady diagnostic decode batch, 22.73 ms belongs to 81 width-2048 operators (~95.8%); width-128 and width-256 fused operators sum 0.74 and 0.26 ms. Archived assembly uses native F64 conversion/add, with volatile F32 rounding barriers generating stack traffic. A software-double-helper explanation is unsupported. This supports focused arithmetic/barrier optimization with exact-output controls, not a threshold change or claim that normalization explains remaining CPU token disagreement.

## Current diagnostic costs measured without changing sampled output

8DC1 basic profile preserves four logits/tokens bit-for-bit versus unprofiled candidate; bounded CPU token check passes four rows only. Three complete steady decode batches average 88.72 ms DSP wall. Fused RMS normalization sums 23.72 ms, exceeding head 2.98 ms and each individual matrix category. Plain matrix categories dominate 337.31-second prefill (278.03 s ordinary, 37.88 s ADD-fused, 20.35 s NX), while gathered up/gate/down sum about 0.95 s. This narrows efficiency priorities for these diagnostic controls, not production bottleneck claims or internal HMX-stage attribution. Profiling disables replay and adds logging overhead. Legacy cycle-gap arithmetic wraps and is not used as evidence.

## Q8 operand correction closes as a stable local win, not whole-model qualification

8DC1 full128 pair 94826 completes normally twice, with all logit bits and tokens identical across arms. BOTH CPU comparisons first disagree at step 93, unchanged from F32 activation-only and Q8 output-only controls. Correcting the sampled plain-Q8 operand errors is insufficient to extend this tested prefix. No claim that all remaining error belongs to another single component follows. Prefill 337.50/337.48 seconds and traced decode 10.85/10.86 tok/s exclude performance promotion. Baseline three files independently hash-verified restored; no inference remains live.

## Q8 operand arm A: sampled precision gains are insufficient for token qualification

8DC1 arm A completes normally but first CPU token disagreement remains step 93, despite correcting the four sampled plain-Q8 projection operand errors. This rejects the claim that those local errors alone explain the remaining full-model disagreement. It does not rule out unmeasured layers, gathered experts, output-head arithmetic, or normalization/activation differences. Same-backend repetition remains pending arm B. Prefill 337.50 seconds and traced decode 10.85 tok/s are diagnostic measurements, not a qualified throughput improvement.

## Actual plain Q8 operand intervention removes sampled local HF operand error

8DC1 capture native0: T9originalQ8_0 ssm_out1.75e-12/shared-up1.31e-14/gate1.07e-14/down5.10e-13, versusCE15~4.22e-8/3.24e-8/2.08e-8/5.80e-8. T1alloriginalgatesremain~1e-14. Required8upstreamtensors bitidentical; wrongweightmappingcontrolsreject~2NMSE. Thisisactualruntimeinterventionandlocalprecisionimprovement, notjustidealHFpairprediction. Wholemodelagreementremainsunproven. Prefill337.67seconds versus229.11previous capture exposesadditionaldiagnosticcost andexcludespromotion; noisolatedpreparationcyclemeasurementyet.

## Q8 output-only experiment closes: stable but insufficient

CE15 fullpair63487 bothnative0/full128logitbitsandtokensidentical. BOTH CPUgates stillfirstdiverge93, same0027. Therefore targetedpairedoutputexport fixesmeasuredlocalprecision butdoesnotextend testedwholemodelagreement. No performance/correctnesspromotion: traced10.95/10.89tok/s, prefill~229seconds. Baseline3files restoredandindependentlyhashverified. RemainingQ8operandprecision remainsmeasuredandtestable, not provenwholemodelcause.

## Q8 output-only local improvement does not extend arm A CPU prefix

CE15 full128 armA native0 butCPU firstselecteddifference remains93 (CPU60239/NPU3812), samepreceding0027. Reference margin0.0516681671/candidateopposite0.157621384. LocalQ8output residualcorrection isreal butinsufficient toextendthismodelprefix. FullrepeatabilitypendingarmB. Prefill229.29seconds/traceddecode10.95tok/s excludespromotion; no causalclaimthatallremainingerror isQ8operandrounding.

## Plain Q8_0 paired output export works locally; operand rounding remains

CE15 capture native0: all4Q8projectionsatT9/T1 passunchangedoriginal1e-7. ssm_out T9originalerror6.47e-8 ->4.22e-8; measuredoutput-to-HFoperandideal5.80e-13, so targetedexportfactor iscorrected whileoperandprecision remains. Sharedup/gate/down output-to-HFideal~9.27e-15/8.15e-15/1.71e-13. Required8upstreamT9 tensors bitidentical to0027 provecontrolledsameinputs forinitialoutputprojection; laterFFNinputs legitimatelychange downstream. Wrongweightcontrolsreject~2.0NMSE. Full128pair63487pending; no token/performancepromotion. Diagnostic~229secondprefill remainsunacceptable.

## F32 activation-low extends CPU prefix reproducibly, still unqualified

0027 controlledpair44754 closesnative0twice. All128 logitrows/selectedtokens arebit-identical betweenarms. BOTH CPUgates firstdiverge93, extendingprecedingcombined5C39 divergence60. This reproduceswholemodel tokenprefixbenefit ofmeasuredF32activationcorrection, notfull128 correctness. Traceddecode10.89/10.90tok/s and~230.5secondprefill excludeperformancepromotion. Baselineall3 restoredandindependentlyhashverified. RemainingplainQ8_0projectionoutput/operandfactors measuredlocally justifyseparateablation; gatheredexperts/coherence/MTP remainunqualified.

## F32 activation-low arm A extends fullmodel CPU prefix but does not qualify

Actual0027 full128 armA native0; CPUfirstselecteddivergence93 versus precedingcombined5C39 divergence60. CPU60239 versusNPU3812, reference margin0.0516681671/candidateopposite0.089805603. Stopcomparison atchangedprefix. Thus the measuredlocalF32 correction also extends testedmodel tokenagreement, but remaining128 gate fails and repeatability remainspendingarmB. HistoricalQ6-only also reached93; this is not a newqualifiedbestdecode. Prefill230.55seconds/traceddecode10.89tok/s excludesperformancepromotion.

## After upstream fixes, sampled plain Q8_0 output projection residual dominates

0027 T9 layer0ssm_out onowninputs passes originalQ8_0 NMSE6.47e-8. PairedCPU now has only1 changed referenceinputcode/0scaleblocks; predictedinputeffect1.85e-11 versusmeasured6.47e-8, residualfraction0.999972. This contrasts with earlier upstream-dominated drift and isolates a remaining local projection-side precision factor, not wholemodel token causality. CurrentplainQ4/F32 correctionflags leaveQ8_0HMX untouched. Frozenfactorials measureactivationHF1.66e-8/weightHF2.09e-8/both4.22e-8; currentoutputHFexport adds~2.73e-8. Threeproducts+pairedexport predicts~1.74e-12 CPUcontract onlyoffline. A future owned-buffer plainQ8_0 ablation is justified after currentfullpair, not modelpromotion or gate relaxation.

## Actual activation-low intervention corrects sampled F32 alpha/beta errors

Signed00270743 capture native0 on unchanged T9 input: alpha/beta originaldense errors fall from3.57e-9/4.68e-9 to1.93e-14/9.91e-15. Wrongweightrow controls reject1.446/1.566. QKV/z retain~9.2e-15 originalQ8_0 gates. GatednormCPUcross falls1.09e-10 ->9.75e-13, near its ownlocal error8.85e-13. This is measured runtime local/downstream evidence for F32 activation operand precision, not frozen prediction alone; no claim that fullmodel token divergence is solved. Prefill230.4seconds excludes performance promotion. Pipelined T33 diagnostic operations remain unsupported and bothprevious/new candidates fail the broad reuse suite there.

## Combined fullmodel control closes with exact repetition but failed CPU agreement

5C39 full128 pair99685 completes native0 twice. Strict comparison verifies all128 logit rows and selected tokens bit-identical across runs. Both independent CPU gates fail at selected token60 (CPU1061/NPU353). This is stable numerical disagreement, not run-to-run randomness in this controlled pair. The successful sampled layer0 corrections are insufficient for model qualification. Traced decode10.98/10.91tok/s with ~228.7second prefill is not promoted. Baseline host/DSP/catalog independently verified restored; no inference live.

## Strong layer0 local improvement does not extend the combined arm A CPU prefix

Combined5C39 arm A finishes native0/128 tokens but first CPU selected-token divergence remains60 (1061 versus353). Thus fixing both initial normalization quantization boundaries and sampled plain-Q4 operand errors is insufficient for fullmodel token agreement. Full samebackend repeat gate remains pending arm B. Prefill228.65seconds/traced decode10.98tok/s excludes performance promotion.

Frozen combined alpha/beta factorials locate remaining local dense error overwhelmingly in activation HF rounding, not weight HF rounding. Ideal activation high+low predicts ~8.9e-15/~3.3e-16 CPU-contract NMSE; paired export is also needed to retain this precision. This motivates a distinct runtime F32 operand ablation, but neither frozen math nor the token failure establishes that alpha/beta causes the first wholemodel divergence. Original gates remain unchanged.

## Combined RMS and plain-Q4 correction removes both measured local factors

5C39 combinedcapture native0: sampledfullT9QKV/z originalCPUowninputerrors~9.2e-15; pairedactualCPUerrors~2.2e-14 with0referenceinputcode/scalechanges. GatednormCPUcrossdrops Q4operand-only3.131e-7 ->combined1.093e-10; frozencounterfactual residual nowrecurrentinputside (1.076e-10), zside~2.7e-14. Normlocalkernelstillpasses~9.3e-13. This interaction explainswhyseparatelocalfixes eachleftdifferentfactor; notwholemodelagreement proof.

F32alpha/beta stillmatchHFoperandideal~1e-14 butorigdense~3.57e-9/4.68e-9, a remainingmeasuredprecisionfactor. Causalitytoallremainingstate/modeldriftunproven. ExistingRMSreusecases pass16of16 overreplayon/off,inplace/outofplace,tokens1/9,32changinginputs; untouched256widepathcoveredalongside2048reference. Fullcombined128pair99685running; scalarprefill~229seconds excludesperformancepromotion.

## RMS arithmetic alone fixes a local boundary but not fullmodel token agreement

13CA01B8 controlledfull128pair bothnative0 andalllogitbits/tokensrepeatidentical. BothCPUfirstselecteddifference60. Traceddecode11.01/11.03tok/s, prefill~2.07seconds; noqualifiedthroughput improvement. InitialQ8_0boundarycorrection is real/local, notfullmodelpromotion. Remainingprojectionoperanderror measuredseparately motivatescombinedfactorcontrol; laterexpert/F32projectionfactors remainunisolated. Baseline restored/nojobs live.

## Actual scalar RMS arithmetic ablation removes the initial input quantization-code discrepancy

Signed13CA01B8 RMSreferenceON/Q4operandOFF capture native0. Bitidentical originalembeddinginputs: newT9normalizedoutput exactlymatchesofflineCPUarithmeticoracle (0error/0bitsdifferent), butcompiledCPU remains1.9613e-15different. ReferenceQ8_0codes allmatchCPU now, versus2changesbefore. Wrongweightnegative rejects0.0165394689; original1e-7gateunchanged. This supportsinitialnorm arithmetic as a separateprecisionfactor, not alteringquantizerties.

T9gatecrosserror remains1.1483071e-8 withpredictedinputeffect0; outputmatchesHFoperandideal~7.5e-15, sooperandroundingpersistswhenitscorrectionOFF. Normandprojectioncorrections eachfix theirmeasuredfactor butfullmodelcombination/tokenqualification remainsunproven. Scalar2048RMSdiagnostic/defaultOFF only; capture ratesnotbenchmarks. ControlledRMS-only128pair61591 running.

## Bit-identical embedding inputs isolate initial RMS arithmetic-path differences

PairedCPU/NPU retained model.input_embed T9inputs hashcc3f16d5... bitidentical. Actual initialattn_norm outputs differNMSE1.6755037e-14; frozeninputeffect0. This directly excludesdifferentembeddinginputs for thisselecteddifference. NumPyCPUarithmeticreconstruction isnotcompiledCPUbitexact (CPU3528bitsdiffer/NMSE1.9613e-15), butitsreferenceQ8_0codes allmatchCPU anddifferinexactlytwofromNPU. Supportscontrolled RMSarithmetic ablation, not broadclaimallmodeldrift isRMS orchangingquantizerties.

Strictlocaloriginal1e-7 gatespassCPU/NPU; wrongweightnegativeNMSE0.0165394663 rejects; selfpairalldriftzero/missinginputsreject. Bothactualcapturesnative0, baseline restored. No RMSkernelreplacementyet.

## Q4 operand correction is a reproducible fullmodel dead end in this configuration

Controlled1152136C full128pair native0/botharms; samebackendfullbitsandselectedtokens identical/native0. BothmatchingCPUtokencomparisonsreject firstat60. Thus stablelocalprecisionimprovement didnotextendCPUgreedyprefix here; no improvementover prior65/historical93, nonequalified. Prefill228.56/228.54seconds is not viableperformance. Keepmechanismandrawartifacts, notpromotion. Baselinehost/DSPrestoredverified. Next initialRMS pairedinputcapture/oracle willtestupstreamboundarycontract withoutassumingcausalityfromoutputdifferencesalone.

## Local Q4 CPU-contract correction does not extend the first fullmodel arm's CPU token prefix

1152136C full128armA native0, butmatchingCPUfirstselecteddifference60 (CPU1061/NPU353), earlierF32residual65/historicalQ6-only93 remainlonger. Step0logitNMSE0.00589067697. LocalQKV/z~9e-15 accuracy is useful mechanismevidence, notfullmodelqualification; correctingmanyprefillprojections changesdownstreamhistory/nonlinearboundaries andnotnecessarilymonotonicgreedyprefix. Candidate's228560.59msprefill also unacceptable. Secondarmstilllive; no repeatidentityclaimyet. Gatesunchanged/no promotion.

## Two residual Q8_0 code differences sit exactly on a half-integer boundary

Frozen layer0 T9 pairedinputs: column1842 in tokenrows0/6 scales to EXACT32.5 onNPUderivedinputs versusoneF32stepbelow32.5 onCPUderivedinputs. Referencecodes33vs32 followunchangedroundawayfromzero; storedHFscalesmatch. This sharpens upstreamprecision investigation, not supportsalteringtierule. Optional boundedboundarydiagnostic/selfpairzero/compile checks pass; originalCPUgateunchanged, runtimequantbytesnotcaptured.

CPU RMS uses F32squares/F64sum/F32mean/sqrtf/scale; HTP qf32vectoraccumulation/reduction/approximatersqrt differs. This is source-level contract evidence, not proof which stage caused these inputdifferences without pairedprenormcaptures. Full128pair remainslive; no promotion.

## Actual runtime Q4 operand correction reaches frozen CPU contract; scalar preparation is too slow

1152136C matchedcandidate capture1510 native0. Full layer0 T9 QKV8192outputs/z4096outputs on sameactualinputSHA887f30af... pass unchanged originalQ8_0 CPU oracle at9.26978496e-15/9.2480069e-15, previously7.4645e-9/1.14785e-8. Actual tail-owned lowparts/rawtileddecoding/retainedHH+HL+LH integration works in this capture. Originalfullmodel/token qualification remains pending. Prefill228709.12ms/13tokens is unacceptable; scalarsoftwarehalf/rounding/repeatedprep and32chunks are diagnostic costs, notoptimizedlatency. Do notship asperformancewin.

Remaining gate crossbackend drift7.25619691e-9 is reproduced by paired reference input quantization:2of18432Q8_0codes change,2of576blocks,0scaleblocks; predicted7.25664291e-9, residual2.11472806e-14. Dense input differenceeffect only1.03728946e-14. These are Pythonreferencequantizer codes, not capturedruntimequantizationbytes. Near1.7e-14 initialnormalizationinputdifference can flip quantizationthresholds and propagate. Do not blame alreadyCPUmatchingprojection or weaken gate. GatednormT9crossimproves5.53e-7 ->3.13e-7 butstill dominatedbyzinputdifferences; localgatednorm~9.3e-13 passes.

Controlled full128 samebackend/CPUtoken pair launched59544; noresultclaimwhilelive. After modelgate evidence, precision/performance path needs hoisted activationlowprep and vectorized preparation in owned HVX-worker context; separately owned VTCMtailmustremain distinct from overlappingGroupC.

## Activation conversion scratch cannot persist across HMX weight compute

Source ownership audit finds Group C act_f32 deliberately overlaps Group B weight/output/dequant regions in the layout builder. Reusing it for persistent lowparts during compute would corrupt live buffers even if a byte-size check passed. Raw weight scratch is also consumed by residual export and cannot supply later lowpart decode once overwritten. Runtime correction needs separately owned storage or adjusted chunks; no inference layout change yet.

Independent host-layout fixture validation: original GGUF Q4 codes/scales packed exactly as host tiled repacker, scalar DSP reconstructed values checked against independent original F32 arrays. Two signed88990AC8 HMX runs native0/normalbackendtestpasses,18fixture576-event frames identical; packed and direct model fixtures have exact same numerical metrics (corrected5.67837589e-15, wrongidentity1.81683218). Independent Python tile decoder checks all generated bytes/F32hashes/zero padding; swappednibble negative NMSE2.01356208 rejected. This establishes sampled packing-to-correction mechanism, not live DMA input provenance or model/token/speed proof. DefaultOFF and baseline restored.

## Actual frozen model reference operands survive three-product correction on HMX

Two signedB8001F89 HMX-routed tests add9capturedprefill activationrows/17dispersed original layer0 gateweightrows, K2048. Original declaredQ8_0 reference contract (not runtimequantbytes): rawHFerror4.92180021e-8 ->HH+HL+LH retained-export/F32sum5.67837589e-15; wrongidentity1.81683218 rejected. Bothnative0/normalbackendtestpasses;544metric-events/run identical. Existing32syntheticchecks unchanged/pass. Generatedexacthex F32 header independently roundtrip hashverified and archived with compilerdefinitions. DefaultOFF/sameowned scratch/baseline restored.

This supports a runtime precision diagnostic, not claims about fullprojection coverage, runtimepacking, tokenagreement or speed. Separate lowoperand workspace and correctCPUquantizationcontract remain integration requirements; three HMX products plus residual extraction have unmeasured cost. No local/token gate relaxed.

## Three-product operand correction works on HMX synthetic F32-origin fixtures

Signed4B1CCA79 candidate, two actual HMX-routed runs: eight new fixtures cover padded/full shapes and K32/256/1024/2048. HH+HL+LH each high/low residual-exported and F32 summed against original F32 operands give NMSE1.57e-15..1.16e-14; all shifted-identity negatives~1.09..1.175 reject. Both runs native0, normal Q4 backend test passes,512metric-events/run bit-identical, old retain fixtures unchanged. Nonzero cross-products and HH residual enforced. Scratch remains owned existing pre-activation workspace with272640byte capacityguard; defaultOFF.

First run failed a fixture assertion (last small-product residual was exactly zero for dyadic inputs), not numerical accuracy. Corrected condition verifies the relevant HH residual and cross-product usage without relaxing accuracy. Immutable failing artifacts kept; decoder independently recomputes numeric gate. This is actual hardware mechanism evidence, not full model/provenance/performance qualification. Frozen model operands still need a direct fixture and owned low-part layout before runtime promotion.

## Operand correction needs accurate partial exports, not just additional HMX products

Controlled F32 residual EBBA repeat is bit/token identical across128 rows, yet diverges from matching CPU at token65 in both arms. Local alpha/beta/state improvements do not establish full decode correctness; historical intended-placement prefix93 remains unqualified.

Frozen layer0 T9 z/QKV operands separate comparable weight and activation HF rounding errors (z6.6201e-9/5.0671e-9; QKV3.6795e-9/3.7936e-9). Ideal HF high+low representations recover the selected Q8_0 CPU contract exactly for these inputs. HH+HL+LH without LL has ideal errors~1e-16, suggesting three products as a focused future mechanism experiment. However ordinary HF partial exports give~4.1e-8..4.4e-8, worse than the current corrected output's operand error. Ideal two-half exports bring three-product errors~1.7e-15..1.8e-15. This screens out naive extra-product/HF-output correction, not proves any hardware implementation or latency gain. HMX partial accumulation, packing, tile ownership and F32 summation still require actual tests; original1e-7/model/token gates unchanged.

Paired gated-norm counterfactual T9 drift is reproduced almost entirely by z-only input change; T1 drift is recurrent-input history instead. Self-pair exactly zero and wrong-head negative rejected after oracle refactor. Frozen input swaps identify promising diagnostics, not measured runtime interventions or whole-model causality.

## Retained residual mechanism survives multi-tile K2048 accumulation in focused fixtures

SignedACB04E9C twoHMXroutedruns, K32/256/1024/2048 timesfull/paddedtiles timescorrect/shiftedidentity =32checks. Strict256-eventmetricdecoder eachrun andcrossrunframeidentity exit0. CorrectedNMSE0..1.08507148e-14, K2048~2.23..2.29e-15; allshiftedidentitynegatives~1.09..1.175 rejected. Nonzeroresidualsinallpositivefixtures. K2048 uses2deep32tileloads beforeHFretainexport, testsaccumulationacrosshardwareloadchunks. Owned268544byteexistingVTCMscratch capacityguard, nohostformat/growth.

Stilldyadicfixturemechanism, notactualmodelprojection/corpus/performance. Realintegrationneedsownedidentity/high/lowplanes andmatchinghost/DSPlayoutcost accounting, F32sumthenbiasonce, pipeline/gatheraliasaudits. Addsidentitymultiply/secondstore/HVXsum; measuredaccuracydoesnotprovelatencywin. Baseline/configsrestored; no gate relaxation.

## HMX retained-accumulator residual extraction works in focused dyadic tile fixtures

Probe biasfix0x3c00(scale1/bias0) instead0x3c003c00(scale1/bias1), signed931AD6F8 candidate. TwoHMXroutedexistingbackendtest runsnative0. Full/padded32x32K32fixture rawroundingNMSE7.49146736e-8/7.69357271e-8 ->corrected0 viaHFhighretainstore, HMXnegativeidentitysubtract, HFlowexport andF32sum. Nonzeroresiduals836/114 proveexercisedprecision. ShiftedidentitynegativeNMSE1.17524505/1.09103572 correctlyfailsmechanismaccuracy. Decoderstrictbothpass/metricsidentical. Priorprobeerrorwasfixturebiasinitialization, notproofhardwareunsupported.

This is narrowhardwaremechanismevidence, notfullQwenprojection/token/speedproof. NoF32storeintrinsicneeded. Modelintegrationmustownbuffers/revalidateVTCM/layout/pipeline/fusedbias/longK/replay andoriginalCPUgate; extraidentityHMXproduct/outputstore/HVXsum costsnotmeasured. Baseline restored/configsOFF. No thresholdrelaxation/promotion.

## Retained-residual hardware extraction is not yet validated

OFFdefaultVTCMownedprobe compiled/signed4A2D0617. Firstfocusedf16casepasses wereHVXrouted andthereforegiveNOretentionevidence. Quantizedq4_0case m16n8k4096 explicitlyhmx-tiled routesprobe, thenDSP INTERNAL-ERROR/hostnative-1073740791. Internalfixtures coverpadded/fulltiles/zero/signedoperands andshiftedidentitynegative; numericFARFmetricsnotvisible so failurecauseunresolved. Couldberetainsemantics/layout/fixturearithmetic, notproofhardwareunsupported. Baseline restored, do notusehypothesizedresidual inmodel orclaims. Routingverificationadded torunner topreventfalselycountingHVXpassasHMXproof. NoCPUgate/performancequalification.

## Split-K rounding does not rescue tested up accuracy; old/new up inputs are byte-identical

Offline fullQ8_K256quantization thenKchunk32..2048 idealFP16partialexports+F32sum: newNPUupallfailoriginalCPU1e-7 (1.12011394e-7..1.22052614e-7). IndependentCPUupallfailtoo (1.07537929e-7..1.14839559e-7). Downpasses, insufficienttojustifynewhardwarepath. Addedstores64..1 withouttestedupgatebenefit; rejectascurrentfix, not universalclaimaboutsplitK.

Old9A66/new1A0DupcanonicalF32inputSHA256 identicalbc7009372593cef50ea66977fb5a53da34beef594ef7c9ba18a07b27385c33f6, sameGGUFweight. Therefore measured upimprovementnotcausedbydifferentinputbytes; stillcombinesF32D/M andqf16bypass. Ideal two-half output highHF(dot)+lowHF(dot-high), F32sum, originalCPUupNMSE7.60070296e-8 passes. This testsrepresentationalcapacityonly, notsupportedaccumulatorextraction/runtimecorrection. :retain.hf residualspike needs focusedsemantic/layout/negativecontrols beforeuse. Explicitroundedoutputoracle exits0 up/down; originalCPUgate unchangedupfail. No speed/qualificationclaim.

## Actual F32-weight HMX output matches ideal FP16 projection; final rounding explains measured up failure

Matched2226host/1A0DDSP capture44256/CLI0, layer5T9: originalCPUQ8_KupNMSE1.22052029e-7 exits1, down8.1984324e-8 exits0. Unchanged1e-7. IdealGGUFweights/inputFP16 thenfloat64dot/finalFP16output matchesup exactly(NMSE0/max0), down2.73738399e-16/max5.96046448e-8. This narrowcapture shows intendedoperandrepresentation/outputcontract; notgeneric hardwareaccumulatorproof. Withoutfinaloutputrounding oracleerrors4.53253323e-8/4.428068e-8. OldF16effectiveD/Moraclefails5.91854657e-7/3.44809712e-7. TestedupidealcontractdiffersCPU1.22052614e-7, accountingforitsfailure. No thresholdrelaxation/fullmodelqualification. Prototype combinesF32effectiveD/M andqf16bypass; factorcausalitynotisolated. Baselineall3restored/hashverified/configsrestored.

## SDK audit narrows accumulator-export investigation; qf16 is not ordinary IEEE intermediate arithmetic

Installed SDK19.0.07 targethmx_hexagon_protos.h exposes HF before/after/retain accumulatorstore andcvt.hf; noSF accumulatorstoreintrinsic found. Headerabsence doesnotprove hardwareabsence. LocalSDK qfloat_tool.c documents VonNeumannfractionalrounding/implicitone. Our simple IEEEweightoracle omits thisqf16chain, explaining why no exactrepresentationclaim is warranted. New OFFdefaultHMXQ5F32diagnostic bypassesqf16dequant whileconsumingmatchedappendedF32D/M; requirespairedhost1024weightformat andseparateactualprojectiongates. No install/runtime/speedclaimyet.

## Q5 HMX effective-weight representation and final output rounding are separate accuracy constraints

Same9A66 frozeninputs/T9, new repackedFP16D/M+finalFP16weight+Q8_KFP16input float64oracle is closer toruntime(up3.26255159e-7/down2.04726165e-7) than originalweightFP16ideal, but stillfails1e-7. It differsfrom originalCPUcontract5.64974897e-7/3.23633203e-7 evenwithout hardware accumulation. OptionalfinalFP16output doesnotresolve runtimegap. Qf16intermediates/actualaccumulator are notmodeled; no runtimeweightbytesproof.

Ideal originalGGUFweightsFP16+Q8_KinputFP16 on sameinputs differsfromCPU7.60071426e-8/4.43354105e-8; finaloutputFP16 changes1.22052614e-7/8.24191408e-8 (upfails). IndependentCPUupfrozeninput:7.15077127e-8 ->1.14839559e-7 confirms this tested representation issue. Not an unavoidablelowerbound orproof of no cancellation; do not claim F32 effectiveweightsaloneguarantee localgate. OriginalCPUQ8_Kregressioncontrol stillpasses1.87267724e-14, thresholdunchanged. Actualsource stores HMXacc with :after.hf thenexpandsF32; investigate supportedprecisionexport separately fromweightrepresentation.

## Complete256 activation blocks are necessary but insufficient for measured HMX Q5 prefill accuracy

Signed9A66FD8C capture86308/CLI0, actual layer5up/down occurrence2/T9: q8_k_fp16_math NMSE8.02458706e-7/4.57709997e-7, both fail unchanged1e-7; originalQ8_K8.19285623e-7/4.76661285e-7 also outsidegate. Wrong32blockQ8_0 references0.000297467578/7.83709568e-5 anddense0.00030096855/6.92270078e-5 are much farther. Complete256-block staging changed intended activation contract, no runtimebyte/fullCPUproof. OriginalweightFP16ideal does not account for effectiveD/M repackrounding/HMXdequant/accumulator/outputpacking. Investigate these on sameinput before fullmodeltest; candidate not locallyqualified, no speedclaim. Baseline restored/hashverified. NoGGUFmutation.

## HMX Q8_0 local improvement does not improve full matching token prefix

Candidate588D8F4F pair64398 bothCLI0 and128 all repeat logits/selected bit-identical. CPU128 strict token comparison exits1 at60(1061/353), CPUmargin0.141433716/NPUopposite0.0226650238; step0NMSE0.0127697778/max0.895251065. Q6-only prefix93 remains longer, candidate rejected for promotion. Diagnostic14.32/14.19tok/s not benchmark win. Baseline restored, build configurations OFF/certificate restored. Full-model gate is independent of locally passing arithmetic and cannot be replaced by average error or repeatbits.

HMX eligibility M > HTP_MM_HMX_MIN_NROWS explains why small chunks may use HVX. K256 activation emulation must preserve full-block ownership: current column DMA partitions32blocks among threads; expert helper loads32F32 independently. Neither can substitute a256 quantizer without restructuring reads/partition and validating scratch bounds. First implement audited complete256-block paths and retain originalCPU oracle/negative controls, not rounded32 approximation or threshold relaxation.

## HMX Q8_0 activation emulation closes measured first-layer prefill contract gap

Candidate588D8F4F capture14128/CLI0, block0 Q4_0 qkv occurrence2/T9: original CPU Q8_0 oracle NMSE4.88149871e-8 and FP16-math oracle4.0975371e-8 both pass fixed1e-7, versus dense2.59892576e-6. Old HMX capture was dense-close/Q8_0-failing. Occurrence3/T4 originalQ8_0 1.25008026e-14, FP16math1.82038511e-8; both pass. T4 exactness requires routing inspection, not an unsupported HMX attribution. Oracles omit actual accumulator/output packing; own input captures are not cross-backend frozen-input causal comparison. No full-model token/performance qualification and no K256 HMX coverage. Baseline DSP/catalog restored/hash verified; candidate compatible with baseline host/weight formats.

## Closing local weight-scale gap does not qualify fullmodel; HMX activation contract remains distinct

MatchedF32effectiveQ5weight fullpair58138 all128repeatbits/selectedexact butCPUfirstdiff15(4602/264); step0NMSE0.00486678944lowerdoesnottranslateintoqualifiedgreedysequence. Baselineall3filesrestored, diagnostic14.11/14.16tok/s notwin. Root/nesteddiagnosticconfigsOFF/certnormal. Hostrebuiltfromdirtysource means tokenchangesnotcleanweight-onlycausality. LocalCPUprojectiongatespassremainsvaluableevidence, notfullmodelproof.

SameCPUfrozenprefillinputs, idealCPUquantizedactivation+originaldequantizedweights bothroundedFP16 thenfloat64dot: firstlayerQ4Q8_0NMSE7.46310225e-9, layer5Q5Q8_K7.1506022e-8, bothfixed1e-7pass. HMXcurrentdenseactivationcontractsarefartherfromCPU. ReferenceomitsHMXaccumulation/outputpacking andF16effectiveD/Mrepackloss, so runtimeactivationemulationstillneedsseparatemath/token gates. Preserveweight-type gating and256blockcoverageacrossDMAchunkboundaries; originalF32routerweightsmustnotgetQ8quantization. NoGGUFrewrites/thresholdrelaxation.

## Preserving Q5_K effective weight scales closes measured CPU-local decode gap

Matchedhost2226402B/DSPA9F5894A firstdecodeactualQ5layer5 up/down,32611CLI0: originalCPUQ8_Koracle nowpasses1e-7 at3.67594073e-8/5.8261387e-8 versuspreviousF16weight/F32sum5.27571371e-7/6.44089737e-7fail. IntendedF32D/M+F16activationD+F32sumoracle2.97831386e-14/1.33250698e-14; oldstoragecontractfails4.9937619e-7/5.29243945e-7. Owncapturedinputs, localprojectiononly, notsameinputfullmodelbisection orbitsidentity. Newpairedhostrebuildfromcurrentdirtysourcealso means fullmodeldifferenceisnotyetcleanweight-onlycausality; retainarchivedbaseline/newsource provenance. Reportedmodelallocation+15MiB matchesderivedtilegrowth; no speedclaim. HMXprefillstillusesF16prefix, noGGUFrewrites. Baselineall3restored/hashverified; normalbuildconfigrestoredOFF+certificate withoutbuild. Full128pairlaunched; modeltoken/repeat/performance/MTPgatespending.

## F32 Q5 activation sums recover prefix15->60, still below Q6-only93

B725AA fullpair83709 completeall128repeatbits/selectedpass; CPUfirstdiff60(1061/353), notqualified. CPU/NPUoppositemargins0.141433716/0.114160538; step0NMSE0.00891350656. BaselineD2Drestored, diagnostic14.03/14.20tok/s notperformancewin. Confirms sumprecision influences modelsequence butdoesnotclosefullcontract; retainQ6-onlyprefix93 asbestcurrentintended-placementdiagnostic, notcorrectproductiondecode.

Weightformat audit: currentQ5tile768 hasno sparevector fororiginalF32effectiveD/M; append256bytes ->1024(+33%tiletraffic). FourHVXkernelinstantiations/twoHVXdotstrides plussharedtilemacros andhostrepackneedmatchedchanges. HMXdequantusesalignedstridebutstillF16prefix; appendedF32 alonechangesGEMVnotHMX arithmetic. CurrentRPCinterfacehasnoformatcapabilityhandshake, somatchedhost/DSPpackageprovenance/restorationrequired; no silentmixedformat. Derived60Q5sharedweightsoverhead15MiB ignoringpadding/workspace, notspeedmeasurement. No modelweightrewrite planned.

## Q5_K offset-sum precision isolation deployed, remaining weight precision gap persists

B725AA0845B7621D9DCBDD07BEDCDBA3E9B331941885F405A3AA98F70224DD85 F32sum-onlycontrol83679CLI0, actuallayer5T1. Up/down intendedF32sumoracle NMSE2.60046469e-14/1.02513295e-14 versus F16sumalternate4.6664833e-8/2.08421336e-8 (also within1e-7). OriginalCPUQ8_K remainsoutsidegate5.27571371e-7/6.44089737e-7. Preserve thresholds and do not claim fixedgate distinguishesbothformats alone. Runtime sum storage/consumer paths reviewed, arithmeticresidualcloselymatches intendedcontract; no modeltoken/performance qualification. Baseline restored aftercapture; fullunretained128pair launched. EffectiveweightD/MremainF16 and knownofflineprecisionfloornotremoved.

## Better local activation contract regressed model tokens; Q5_K precision sweep identifies distinct losses

Q5+Q6 reference6FF0937 fullpair84214 all128repeatbits/selectedpass but CPUfirsttokenmismatch15 vsQ6-only93, so rejectpromotion. Step0NMSEimproves0.01713->0.01002 whilegreedyprefixshortens: do not rankcorrectness bylogitdistancealone. D2Drestored,14.12/14.06tok/s traceddiagnosticnotwin.

Same frozenCPUinput/integers/weights up/downQ5_K T1 oracle8precision combinations: allF32~4.26e-14/2.18e-14; roundingweightD/Malone~4.93e-7/5.28e-7 (failsboth), activationDalone~2.54e-8/5.15e-8 (passesboth), sumalone~2.08e-7/7.24e-8 (failsup), allhalf~7.75e-7/6.65e-7 (failsboth). OriginalCPUoutputs remainreference, thresholdunchanged. Offlinefactorseparation isolates representation loss withoutdifferentupstreaminputs; not proof of whichroundingcausestoken15. WeightrepackeffectiveD/Mprecision mustbe addressed separately from activationintegercontract/sumprecision. No GGUFweightschanged, no broadqualification.

## Q5_K reference prototype matches tiled math but not original CPU contract

OFF-default Q5_K activation reference signed6FF0937 stores shared256signedscale/int8 in existing32group tiles plus FP16(F32scale*sum32). CPU sameinput precision emulation up7.75026904e-7 versus originalCPU proves packing/offsetrounding can exceed fixedlocalgate before runtime. Deployed67931 T1 up/down matches explicit tiledoracle2.53306955e-14/8.34684425e-15NMSE; originalGGUF+Q8_K math stillfails5.53934304e-7/1.61946852e-6. Both weightrepack effectiveD/M andactivationD/sum rounding are accounted for by newreference, not a proof of each contribution in isolation. No claim of CPUequivalence; neworacle diagnoses storagecontract only, originalselectedtoken/fullmodel gates unchanged. Baseline restoredaftercapture; full128pair84214launched. No modelrewrites.

## Q5_K decode confirms local CPU Q8_K contract; original NPU references still fail

Actual first decode occurrence4/T1 layer5 shared up/down, session87082 bothCLI0. CPU upQ8_K3.80178643e-14NMSE/down1.74026544e-14; FP16-scale-only emulations1.41239921e-8/2.04310522e-8 also passfixed1e-7. NPU closest tested referenceQ8_0 up9.04594154e-6/down1.36427688e-5, both fail. Multi-token HMX prefill explanation no longer sufficient for these T1 observations. Need explicit Q5_K shared256scale/int8 contract plus offset sum32, with FP16scale/sum precision measured, rather than assume original Q8_1 is exact F32Q8_0. Packedweight scale/offset rounding remains possible contribution; own input differsupstream, no claim of solecauseof token93failure. Baseline restored and goal incomplete.

## Confirmed Q5_K shared-up prefill separates CPU activation math from NPU path

Actual layer5 up Q5_K, occurrence2/T9, CPU uses Q8_K reference1.87267724e-14NMSE; rounded FP16 scale remains within gate3.65846444e-8. NPU/mm2 closest dense8.11425801e-7 but fails fixed1e-7; Q8_05.96679247e-5 andQ8_K0.000297911741 fail. Multi-token HMX prefill is not a clean HVX Q8_1 quantizer test. Next single-token decode capture required before attributing residual drift or changing Q5_K activation math. BothCLI0/session21620terminal0, baseline restored. Own captured inputs used, not same-input CPU/NPU bisection. No threshold relaxation.

## Layer0 shared weights are Q8_0, not the intended Q5_K probe

GGUF metadata corrected the planned Q5_K capture selection: blk.0 shared up/gate/down are Q8_0, layer5 contains actual Q5_K. Preserve completed11229 captures as Q8_0 controls, never label them Q5_K evidence. Layer0 up occurrence2/T9 CPU matches Q8_0 math1.85794092e-14NMSE; NPU/mm2 matches dense7.49492825e-8, while Q8_0 fails2.85641926e-5. Shows another local prefill activation-contract difference; independent inputs differ upstream, so not a same-input causal bisection. Runtime/shape/type gates prevent broad claims from incorrectly selected weights. Next layer5 capture explicitly targets confirmed Q5_K weights.

## Q6_K head contract extends intended-backend CPU prefix, not complete correctness

Full unretained128 mm2 pair65884 bothCLI0: all128 complete logits bits and selectedtokens repeat exactly. CPU first mismatch moved60 (F32 Q8-only control) ->93 (Q6_K Q8_K-style head), CPU60239/NPU3812; margins0.0516681671/opposite0.246034622. Still fails unchanged token gate. Step0 NMSE0.0171299684 is worse than F32 Q8-only0.00610490297 despite longer selectedtoken prefix: numerical distance and greedy prefix are distinct evidence, neither alone qualifies the model.14.45/14.49tok/s traced/costly whole-cache diagnostic, not performance promotion. Baseline restored. Q5_K activation offset path uses original FP16 intermediates; F32_Q8_REFERENCE does not control that path. Next measure frozen shared-expert projections before assuming it is the residual token cause.

## Opt-in Q6_K activation contract verified on actual NPU head samples

Candidate243C6F45 (full hash in STATUS), whole-cache repeat control plus F32 Q8 and Q6_K activation reference, head-math-prefill-q6k capture59007 CLI0. Same actual Q6_K head weight, occurrence3/T1/eight sampled rows. Q8_K with tiled FP16 scales matches NMSE1.56846704e-9/max0.000643871641; Q8_K F32-scale also within local gate at2.76802276e-8/max0.00325390497. Q8_0 now fails1.80682729e-5/max0.0802623889. This verifies the deployed diagnostic changes the local activation contract as intended, without a model weight rewrite. No full-vocabulary, token, performance or MTP qualification follows. Baseline restored after capture. Full unretained128 pair launched session65884, first arm verified live; wrapper restores baseline after both arms terminal. Main source whole-cache patch removed; candidate source archived separately, nested diagnostic build flags ON/ON.

## Shared 256-element activation scale remains feasible in existing tiled precision

Frozen CPU head, eight previously selected vocabulary rows: Q8_K integers with scale rounded through FP16 produce NMSE 1.84103435e-8/max_abs 0.00189209663, passing the unchanged local 1e-7 gate. The corresponding existing NPU head capture rejects this alternative (3.55743478e-5/max_abs 0.0795360577), a useful negative control. This is emulation only, not deployed kernel evidence. FP16 scales lose precision relative to CPU F32; no claim of bit identity or full-model token correctness. The block-parallel tiled worker boundary is 128 elements while CPU Q8_K is 256: each worker must read the whole input pair but emit only its assigned tile. Q5_K has a distinct offset/sum path; begin narrowly with Q6_K rather than altering all K quantizations at once.

## Actual Q6_K head uses different activation block contracts on CPU and NPU

Completed26214 CPU/NPU single-token chat headcapture, NPUwholecache+F32reference/mm2. Frozenresult_norm/result_output occurrence3 oneoutputvector, actualGGUFQ6_K output.weight, sampled8vocabularyrows9419/0/11/4602/264/60239/3812/99731. CPUmatches scalarQ8_K256element/F32scale oracle4.6172844e-15NMSE; NPUmatches Q8_032element/F16scale1.49551345e-9NMSE. Alternatives faildeclared1e-7: CPUQ8_02.22190851e-5/dense1.5941992e-5, NPUQ8_K3.58102057e-5/dense6.46726899e-6. This identifies a localactivationcontract difference, notbadQ6weightdecode orprovenonlycauseof60/93tokenfailures. Eachmathreference uses its owncapturedinput; modelinputs alreadydiffer.

Newsampled-row option slices quantizedweightbytes beforedequant toavoid whole-vocabulary allocation; bounds/duplicates/shape/finite checks and1e-7unchanged. Q8_K scalarreference followslocalggml-quants.c nearest_intmagic-add/signedmax/scale; doesnot prove runtimebytes. T4selection rejected becauseheadoutputvectorT1, corrected; nohidden thresholdrelaxation. BaselineD2D restored. Next selectively emulateCPUQ8_K activation contract forKweights, keepingQ4unchanged and auditingtiledF16scale loss; still requirefullmodeltokens/repeat/performance/MTP.

## Completed HMX-enabled precision control repeats, but token gate still fails

4770 terminal0/bothCLI0,23E6/mm2 wholecache+F32reference128pair strictbits/selectedrepeatpasses all128rows. CPUselectedtokenfirstdiff60 remains. Botharms HTP19127MiB/Host1MiB/noCPU_REPACK, diagnostic14.32/14.23tok/s, not uninstrumented speedpromotion. D2D/catalog restored, sourcebaseline/referenceOFF/no liveinference. Next inspect remaining actualquantization contracts (CPUQ8_K versus internalQ8_0 for K-quant/head and HMXprefill versusCPUQ8 activations), without treating a precision control as fullcorrectness or performance success.

## HMX-enabled F32 reference improves CPU prefix without large CPU fallback

First4770 arm wholecache-q8f32-mm2-repeat128-a CLI0/complete128rows. MatchingCPUfirstselecteddifference60 CPU1061/NPU353,61commonprefixrows, margins0.141433716/0.0886878967; improved overreferenceOFF/MM2 firstdiff15, notcorrectnesspromotion. InitiallogitsNMSE0.00610490297/max0.828184128. Placementreports HTP19127MiB/Host1MiB/noCPU_REPACK vsMM1 hybrid16GiB CPUrepack.14.32tok/s traceddiagnostic not a pairedbenchmark. Secondarm pendingrepeatbits; originalbaseline restored byfinallyafterbothterminal. This shows usefulquantizerprecision on intendedbackend path, while HMXprefill/otherquantization/reference differences remain. CPU/NPUmargin is not especially tiny here; do not dismiss remainingfailure as harmlessrounding.

## Real-input quantizer improvement extends complete CPU prefix, but full gate still fails

23E6 wholecache+F32reference/mm1 complete128pair81678 bothCLI0 repeatbits/selectedstrictpass. MatchingCPUfirstselecteddifference93 (60239/3812), versus15 withreferenceOFF;94commonprefixrows, margins0.0516681671/0.155864716. Localprojection improvement supports this precision change, but model nottokenqualified. Do not attribute all remaining drift to one stage fromtokenmargin alone. HybridCPU_REPACK16560MiB and3.64/3.34diagnosticrates are not intendedNPUperformance; baselineD2D/catalog restored. Nextmm2 HMXenabled pair running same23E6, testingplacement and accuracy rather than acceptingCPUfallback.

## Existing F32 activation reference matches actual frozen projection

Under samewholecache/MM1 first-layer probe, enabling existing F32referenceQ8quantizer changes realQ4 projection vsQ8_0 activationmath NMSE6.42224989e-7 ->1.32239585e-14 (max8.33359081e-6); threshold1e-7 unchanged. Denseoracle2.53194076e-6 reflects intentionalactivationquantization. Supports localFP16/intermediatequantizer rounding distinction from F32reference, not weightdecodebug or fullmodelqualification. Signedcombinedcandidate23E6 archived; configOFF/mainsource reverted and baselinepackage restored aftercapture. Live128repeat/CPUtoken gates required, with CPUfallback placement explicitly retained aslimit. Scalarreference not proposed as speedoptimization.

## HVX-preferred projection still misses CPU activation oracle; combined128 CPU gate fails

mm_select1 means HVX->CPUfallback, not allHVX. Captured first-layer Q4projection vsreferenceQ8_0:6.42224989e-7NMSE/max0.0275074409, faildeclared1e-7; dense2.15424129e-6. CPUprompt4 referenceQ8 agrees1.2086363e-14. Wholecache/mm1 pair72468 CLI0 both,128complete bits/selectedrepeatidentical. Crossbackend matchingCPU128 gate firstselected15 CPU4602/NPU264,16sharedprefixrows, oppositeNPUmargin0.0619049072; stablebutnotCPUtokenqualified. RuntimeCPU_REPACK16560MiB invalidatespureNPU/allHVXinterpretation oflabel; unpaired4.91/2.76tok/s underotherhostmath work not speed evidence. BaselineD2D/catalog restored/no livejob.

Next use actualfrozeninput case to evaluate existing F32reference quantizer or weight/output-stage math before attempting optimized quantizer. Generic synthetic passes are insufficient on these model activations. Shapeguard rejected mistakenoccurrence2/T4 selection, corrected3; no thresholdrelaxation and no test failure suppressed.

## Actual first-layer projection has two different activation-math references

Captured samechat prompt9/4 CPUandNPU E956/mm2 block0 inputembedding, norm, qkv/z projections and output. Embeddings exact, normsnear1e-14NMSE. Prompt9 qkvprojectionCPU/NPU differs2.56288407e-6NMSE/max0.0465145111. FrozenGGUF blk0.attn_qkv.weight Q4_0 exactdequant/F64dot oracle: CPUcloselymatchesreferenceQ8_0activationquantization1.29274106e-14NMSE, notdense2.49906236e-6. NPUcloselymatchesdense5.13357858e-8NMSE, notQ8reference2.59709846e-6. All declaredthresholds1e-7 unchanged. GGUF PythonQ8reference not proof ofruntimeactivationbytes; norminputs are closelyequal, notbitidentical, so nonlinearrounding can adddifferences. Both locallyreasonablemath paths do not qualify same selectedtokenstream.

Evidence explains an earlyaccuracy difference independent of cache repeatability, rather than proving bad Q4weightdecode. Next samewholecache/mm_select1 (allHVX) frozen projection and128repeat/CPUtoken gates, live72468; earlierallHVX instability tests withoutwholecache do not answer this combinedcontrol. No promotion; declaredCPUtoken gate remains required despite dense-like NPUprojection havinglowerdenseerror.

## Matching CPU128 reference terminal confirms token15 failure

7498 terminal0/CLI0,128complete rows/selectedtokens. Strict require-tokens comparison against wholecache128-a exit1, firstdiff15 CPU4602/NPU264 withsame16prefixrows and same margins/metrics reported below. Earlier partial evidence now backed by completed alignedlogs. Repeatbits fixed in this test does not qualify numerical/tokenaccuracy. CPUdiagnostic0.53tok/s undertracing, not ratebaseline. BaselineD2D installed/no liveinference at audit; reference retained for subsequent candidates using same CLI/chat/model/settings.

## Stable whole-cache chat output fails matching CPU token prefix

While fresh same-stage-runner CPU128reference7498 remains live, partial21-row comparison already shows selectedtoken15 CPU4602/NPU264 against E956 complete128run, with16 shared-prefix rows. CPUmargin0.00659942627 vsNPU opposite-margin0.0288944244; step0logitNMSE0.0369790405/max2.03249264. Partial run is sufficient for this counterexample, not for a complete128gate or any later changed-prefix attribution. Let CPUreference finish. Wholecache repeatbits pass is not selected-token/accuracy qualification; do not promote slower diagnostic. No threshold relaxation on narrow margin.

Cache coverage audit identifies unified memalign4096 ctx+main/work/HMX stack+DMA descriptor rings (ctx->footprint), and separate512KiB DDR scratch outside tensor src/dst ranges. Existing perbatch wholecache flush/prefetch does not make perop activationflush equivalent to wholecache. This identifies further controls, not a proven metadata coherency bug. Source primary main.c init_ctx and dma-queue.c ring layout; device-level cache attributes/coherence unresolved. Frozen actual matrix math/activation quantization oracle remains relevant to independent accuracy failure.

## Activation-only range control does not recover whole-cache success

Temporary3324 candidate flushes every nonweight/nonfence nonempty src/dst activation range independent of dirty tracker, followed by syncht(memoryclobber), tracker maintained. Built35644/signed; exactmain/so/catalog archived. Session96723 bothCLI0 unretained128 pair fails strictbits/token: firstbits0 NMSE0.00569143334/max1.09128118, firstselected15,16 commonprefixrows. Diagnostic25.30/25.03tok/s rejected, not throughputwin. BaselineD2D/catalog restored, temporarysource patch removed. Compared with E956 wholecache128bitpass, changesinclude rangecoverage/cacheAPI/trackerreset/timing/layout; no proven weight-cache cause from this control. Next fresh same-chat-runner CPU128 reference to qualify E956 selectedtokens before optimizing cache scope further.

## Full128 repeat-bit identity with whole-cache per-op diagnostic, without retained probes

Session23800 bothCLI0/restoration terminal0. perop-wholecache-unretained128-a/b strictbits/selected complete128rows allidentical with E956, diagnosticFA1/GDN1/routerHVX/replay1/fusion1. No retained/snapshot probes. Rates17.09/16.74tok/s instrumented, not promotion. This extends prior matched32retained pass and provides stronger cache/control evidence; cache API barrier/timing/layout confounds remain, no universal rootcause claim. BaselineD2D restored/hashverified. Next activation-only independent-of-tracker rangeflush plus syncht candidate3324 built35644 and live96723 tests same128workload.

Crossbackend reference provenance matters: historical alignment-cpu128 is llama-completion raw -no-cnv/c256/Hello1prompttoken, whereas new stage runner uses llama-cli chat singleturn/c512/Hello13prompttokens. Attempted cachedCPU compare differedstep0 and was excluded as mismatched inputs, not a modelaccuracy failure. FutureCPU gate must use the same stage runner/model/prompt formatting; raw workload's historical39difference remains a separate result.

## Whole-cache per-operation control restores repeat bits in matched retained layout

Baseline early-block-boundaries-repeat32-a/b, session11897 CLI0 both, retained embedding and l_out0..9, no snapshots: all363 nonwarmup records exact, but32-row logits strict exit1 firstbits6 NMSE0.000118477632/max0.206396103; all selected tokens same. Temporary diagnosticDSP E9560F297EFB470BC70A61615BE6130F0E35CD6366FDE52A84F2C2D588FC5160 adds QURT whole-data-cache flush/invalidate and clears dirty tracker before everyop input prep; no arithmetic changes, original batch-boundary cache calls remain. Matched perop-wholecache-boundaries32-a/b session89083 bothCLI0, same host/model/flags/probe: all32logits bit-identical/selected-identical, all363early records finite/exact. Rates16.43/16.50 vs22.47/22.49 instrumented; not performance promotion.

This is control evidence, not root-cause proof: SDK qurt_memory.h says cache API also performs syncht onv60+, and whole-cache calls change timing and code layout. Tracker reset/invalidate is not a standalone barrier control. Earlier standalone syncht rejection does not make cache causality proven. Candidate exact source/so/catalog archived backup-perop-wholecache-candidate; prechangeD2D/catalog/source backed up. Temporary source reverted (git diff --no-index --ignore-space-at-eol against backup exit0), installedD2D restored/hash verified after firstpair. Unretained128 pair now required before attributing general repeatability benefit.

## Independent recurrence validates actual captured attention and state carry

check_frozen_gdn_math.py implements float64 decay/delta/outer-update/query recurrence from frozen F32 model inputs and compares recorded attention plus next graph's frozen input state. Layer10 prompt4 botharms matches attention6.66431474e-14/state1.22510262e-13NMSE; layer10 firstdecode1 matches1.97280864e-14/1.43833466e-14. Wrong Q/K mapping h//2 instead of modulo16 fails strongly3.01318991/0.874687496NMSE. Layer18 prompt9/prompt4/decode1 in botharms allmatch attention<=4.71086639e-13/state<=2.19796319e-12 (threshold1e-7 unchanged). This adds a math/carry oracle beyond CPU/NPU agreement; it does not require float64/F32 bits to match. Requires ordinary consecutive graph carry, no intervening reset/rollback.

Deeper snapshot18 layout pair44284 completed32rows/CLI0 both, selectedtoken firstdiff15 and bitsdiff0. Prompt9 already differs at retained l_out9, while initialstate18 stillidentical; laterinputs vary, so layer18 outputdifferences are not operator-local repeat failures. Prompt9/4 comparison1equal27different across28 records. Moving probes changed upstream layout, limiting prior snapshot10 downstream-localization claim. Both arms' different actual inputs produce independently correct GDN attention/carry. Pursue actual matrix-input/reuse and earlier ownership seams; no arithmetic/backend/MTP promotion.

## Real nonzero recurrent state now included in isolated accuracy and prefix tests

Opt-in layer10 immutable DUP outputs capture conv window plus Q/K/V/gate/beta/state before GDN; source/consumer tensors unchanged. DUP not allocator-inplace eligible; output flags preserve snapshot storage. Actual snapshot pair32 selectedtokens first differs19 while all210 retained/snapshot records from prompt9 through that common-prefix step remain bit-identical, including l_out9 and l_out10. Logits already differstep0. This observed retained layout points downstream of10, not universal localization: probes alter allocation/fusion and cannot establish uninstrumented root cause.

GDNFX002 state-aware fixture2229272bytes SHAB6127A737CFB5FD9C480253CCE5EB53369CD3968D38A7095287923936273C6D0 freezes all five production inputs and actual nonzero initialstate (523776elements nonzero,maxabs1.05241084). Both captures identical. Existing001 format preserved. Actual-state T4 K1/K2, contiguous/permuted and independent accepted3/verify4 partial-prefix gates:6/6 bothreplay,32repeats each=384 comparisons. Wrongslot deliberately uses finalslot0 instead of partialslot1: bothpartialcasesfail CPU/NPUstateindependently, ordinary4pass; expectedbackendexit1. V1 legacyfixture regression12/12 passes. No arithmetic threshold relaxation or synthetic-state substitution. State-aware fixtures rejected by old chain path until explicitly designed for actual carry.

Exporter groups snapshots by explicit graph-index at conv-window marker, validates7 exactshape finiteF32 records including state and all payload framing, refuses overwrite; wrongtokennegative produces nooutput. Snapshot gate actualshape[1,32,T,1], not older pre-reshape gate[32,T,1,1]. An initial shape mismatch rejected export and was corrected, not counted as a passing fixture. Next compare standalone output against frozen model attn_output or probe downstream boundaries with same-prefix constraints; full-model repeatability and MTP remain unqualified.

## Actual fused decode does not expose the optimized generic concat stage

concat-dim0dma-stage16-a CLI0, FA1/GDN1/routerHVX/replay1/fusion1, device profile1/verbose/logits, no retained tensor probes. Parser validates15 complete1384-op one-token batches with exactly one2048x248320 output head. Mean DSP batch33970.933us; grouped operator attribution: plain MUL_MAT7539.067us, expert gate/up MUL_MAT_ID_NX4671.400us, MUL_MAT+ADD2982.733us, output head2969.467us, expert down2707.333us, GDN+CPY1728.867us, SSM_CONV+WIN1493.467us. No standalone CONCAT in selected steady batches. Warmup/prefill CONCAT source qkv strides32768:4 are transposed and outside row-DMA eligibility. This explains why generic concat microbench gains cannot be assigned to actual fused decode.

Profile samples are within one instrumented run, not independent performance trials; verbose wall throughput1.34tok/s is intentionally unusable as an uninstrumented rate. New campaign parse_stage_profile.py checks complete selected batches/head shape and groups all operator times, not just head. Summed operator times are attribution, not additive host latency. Second matched arm b completedCLI0/wrapper99202 terminal0,15 complete1384-op batches, mean33850.533us, head2968.467us, SSM_CONV+WIN1460.067us; broad costs repeat. Strict logits comparison exit1:16 complete rows, selected tokens allmatch, bits differstep0 NMSE0.000559164819/max0.300038815. No128-token qualification from this16-token instrumented pair. First outer wrapper falsely rejected completed arm because inherited LASTEXITCODE was stale; actual CLI0/logs complete, second arm uses script success/throw rather than stale native exit code. Comparator selected markers now parsed directly from raw log bytes to avoid unrelated invalid UTF-8, preserving complete-row and binary gates.

Next correctness probe design: immutable graph copy of gathered pre-GDN state and selected conv history before writes, separate opt-in name/filter, exported with dimensions/strides and all five activation fixtures. Existing postgraph backing views are not immutable pre-operation evidence. Extra copies alter allocation/order and may suppress instability, so require paired layout controls and no root-cause inference from a passing copied-state arm alone. No new production probe implemented yet.

## Generic row-DMA concat has paired local performance evidence

Completed concat-model-shape-abba-v2 A/B/B/A, DSP27 vs D2D, unchanged host/test executable, four12/12 correctness arms and four12-case performance arms. Packed F32 dim0 [3,8192,1,1] plus1/4/9 rows: baseline means98.31/178.285/292.125us versus candidate78.545/60.85/65.535us, reductions20.10/65.87/77.57%. Both candidate timings lower than both baseline timings for every one of12 shapes. Candidate restored/hash verified after reverse-order baseline arm.

Evidence scope: batched amortized operator cost with replay1/fusion0, calibrated run counts, not per-token latency or model speedup. Source row-stride variants are not production transposed-qkv strides. Full-model repeatability still fails at selectedtoken15; no inference/MTP correctness promotion. Initial empty perf output rejected and excluded; registration added to existing separate perf factory before complete v2 series.

## Focused concat fix is insufficient for full-model repeat/token identity

After approved fresh-session recovery and focused/default/negative gates, normal Q4_0 Hello128 pair with candidateDSPD2D/hostC9, diagnostic FA1/GDN1/routerHVX/replay1/fusion1/UB128, no retained tensors, logits tracing: bothCLI0/session78828 terminal0.128rows each, strict comparison exit1: logits bits differstep0 NMSE0.00307839269/max0.902320385; selectedtoken difference15 (4602/264), margins0.12786293/0.121154785. Only16 rows share generated prefix; later divergences not operator-local evidence.27.17/27.06tok/s diagnostic rates, no uninstrumented paired speedup claim.

Dim0 row-DMA candidate addresses focused generic direct persistent-cache failure, not full-model root cause. Production-like transposed conv path already passed baseline; do not conflate these findings or claim MTP fixed. Next paired stage cost/placement audit and actual model ownership seams remain required.

## Fresh session and DMA concat regression gates pass after approved recovery

Explicit user approval used only for PID231784 termination and one fresh session, no reboot/driver reset. Interrupted old replay0 never executed graph; wrapper exit1/backend-1 is recovery bookkeeping, not accuracy evidence. Fresh fusion0/replay0 all18/18 pass32 repeats; earlier replay1 also18/18. Total1152 focused comparisons (384 bypass cases, not convolution ops).

Existing default CONCAT suite passes48/48 supported cases for replay1and0,96 total; unsupported types not qualified. DMA candidate wrong-tail control both replay settings0/18pass, expectedexit1 with CPU/NPU independent mismatch. HostC9/DSPD2D unchanged. Focused discrepancy addressed; full-model repeat/CPU tokens/MTP and latency profiling still required before promotion.

## Row-DMA concat address mapping checked independently offline

While10553 remains live at session open, independent host-side scalar tensor-index oracle compared with two-descriptor row-copy mapping:512 deterministic randomized dim0/dim1 cases,646620 addressed bytes, all pass. Seed0xD0C0 LCG; element widths2/4, variable source widths/row counts/outer planes, dim0 row and plane padding, dim1 contiguous layout. Checks destination offsets, source selection, row/plane strides and segment offset. Source descriptor limits audited against dma-queue.h (24-bit widths/strides and16-bit heights onv81). This is address-formula validation only, not actual DMA/cache/completion/performance qualification. No competing run or production mutation.

## Pending second process waits in session open, not graph execution

Non-invasive nonsuspending ARM64 cdb stack inspection of replay0 PID231784 (session10553), saved dim0dma-replay0-init-stack-v2.log, terminal0/detached. Main stack waits through Qualcomm libcdsprpc remote_handle64_open/invoke and WaitForSingleObjectEx from ggml_hexagon constructor. Current wait precedes graph execution. No claim that concat candidate cannot affect prior DSP cleanup/state; no causal driver diagnosis from public symbols alone. Process preserved, no reset or concurrent NPU run. First candidate replay1 suite passed18/18; replay0 remains unqualified.

## Row-DMA dim0 concat avoids focused scalar-path discrepancy in first arm

Signed DSP D2D98EDEA1BDFD8652EE0766C4B7707AE1209E335164E6115478091C304B68CD candidate built54715 terminal0, hostC9 unchanged; baseline DSP27/source/catalog backed up. Existing dim1 DMA helper generalized to dim0 row-contiguous F32/F16/I32, matching outer dimensions and valid descriptor limits/strides, single-device only. Two DMA copies instead of generic scalar element loops; transposed-input path unaffected by nb0 eligibility.

Fusion0/replay1 focused18/18 pass32 repeats, including two former generic direct-cache failures, independent full/split and repeat byte gates plus CPU/NPU components. Replay0 still live10553 at session allocation, not qualified. This shows an effective focused candidate, not established full-model root cause or measured latency win. Negative controls/default CONCAT regressions/stage benchmarks/full-model token and repeat gates pending.

## Removing convolution arithmetic suppresses focused failure

Six opt-in bypass_conv1 controls use oldest-window-view transpose+CONT in place of convolution, same9/4/1 concat/tail-copy/cache policies and full14 delayed-prefix semantic reference. Build52393 terminal0/testexe only,18 cases. Fusion0/replay0 and fusion1/replay1 each16/18 pass: all bypass variants pass32 repeats, including generic direct persistent-cache reads; only convolution-containing generic direct persistent raw/SILU fail NPU-only. Group op_desc remains SSM_CONV solely for test selection; bypass graphs have no convolution op.

This does not isolate convolution arithmetic as cause: removing it changes DMA traffic, allocation and ordering. Simple strided-copy/scalar-read path alone passes. Inspect actual convolution-containing dependency/alias order and DMA/cache seams next; do not adopt snapshot workaround or infer full-model cause. No production mutation retained.

## Copying committed cache to immutable read state suppresses focused failure

Added snapshot_read1 to existing opt-in conv chain: keep reused cache commits, but CPY each committed state to a fresh immutable buffer consumed by next chunk. Four added cases raw/SILU x generic/transposed,12 total. Fusion0/replay0 and fusion1/replay1 each10/12 pass; only generic persistent direct reads fail NPU-only, first attention token13 plus state mismatch. All immutable reads pass32 repeats, same CPU/NPU component and independent semantic/repeat bit gates. Extra copy changes cache transitions, lifetime and timing; not proven production fix.

Build87448 terminal0/testexe only also rebuilt restored cpy source, removing rejected preflush binary from build output. Installed hostC9/DSP27 unchanged/hashverified. Host descriptors deduplicate data+shape/strides; sampled flags0, no weight exemption found. Minimal copy/read reuse without convolution is next, rather than assuming arithmetic failure or adopting expensive snapshots.

## Production-like transposed convolution source passes; unconditional pre-copy flush is insufficient

Expanded frozen conv chain with logical-byte-preserving transposed chunk layout. Baseline fusion1/replay1 and restored fusion0/replay0 each6/8 pass: transposed-source persistent/distinct raw/SILU all pass32 repeats; generic distinct passes, generic persistent fails NPU-only at final token13/state. Layout conversion also changes graph lifetime/timing; not proof production cannot hit related ownership errors.

Dirty tracking does not explicitly record clean lines filled by scalar reads. A narrow diagnostic unconditional destination clean/invalidate before same-shape DMA CPY was built/signed/installed (84452 terminal0), but same6/8 outcome persists. Rejected and source reverted; original DSP27 and signed catalog restored/hashverified, no production change retained. Rejected binary remains in build output pending rebuild; installed package is baseline. Next isolate generic concat/cache read and snapshot ordering instead of assuming this cache hypothesis proved.

## Frozen convolution chain exposes reused-cache discrepancy on NPU

Existing test opt-in conv_state_chain compares full14-token SSM_CONV against9/4/1 split, kernel4/channels8192, declared synthetic frozen inputs/history, raw/SILU and distinct/persistent caches. Bytewise full/split equality independently per backend, crossbackend component NMSE1e-7 and repeat bits. Generic CONCAT input layout; production transposed-source layout remains untested.

Distinct-cache variants pass32 repeats in fusion1/replay1, fusion1/replay0, fusion0/replay1 and worker1/fusion0/replay1. Persistent variants fail NPU-only; CPU remains full/split identical. Detailed first attention differences occur token13 (last1-token chunk), with finalstate mismatch; repeat onset and channels vary across runs. Replayoff, fusionoff and worker1 are not sufficient fixes. HostC9/DSP27 unchanged. This is a focused discrepancy, not established full-model root cause or safe production fix.

Wrong-tail control shifts each state source one token earlier: all4 variants fail on both CPU/NPU at repeat0, first attention difference token9, state mismatch, replay1/0 expected exit1. Oracle catches incorrect semantics even with CPU/NPU agreement. Next audit graph alias/ordering and cache completion contracts, reproduce production transposed-source CONCAT, isolate strided copy/read seam.

Corrected GDN fusion0 session88099 terminal0/replay1 and0 each32/32 pass; corrected fused/unfused total4096 comparisons, without restoring withdrawn historical K2 claims.

## Corrected forward carry passes focused chain; no matching production indexing bug found

15321 terminal0, fusion1/replay1 and0 each32/32 pass (2048 comparisons) after both chain carry views were corrected to newest slot0. Component NMSE, frozen repeated bits and whole-chunk restore/replay gates pass. This restores focused K2 forward-carry evidence, not the withdrawn historical claims or full-model correctness.

Production source audit finds direct newest-first snapshot copy, build_rs row gathering and s_copy's idx*mem->size+src0 selection/reset; normal index0 and rollback-distance groups are consistent with slot semantics. No analogous production written-1 selection found in that inspected path. Actual state ownership, conv/KV rollback and model-level repeat instability remain open.

## Harness correction found by independent prefix oracle: snapshots newest-first

CPU10955/11037 andDSP885 mapping targetslot=T-1-t shows slot0final/slot1onetokenback. Earlierchain K2 selected written-1, so consumedolderstate; allpriorK2forward-carry qualificationclaims withdrawn. TheirCPU/NPUagreement andsamehistoryrestoreconsistency stillobserved, but notcorrectforwardcarry. K1unaffected. Fixednormal/replayedcarry tooffset0,test-only build40475terminal0. No model/backend arithmeticchange.

New T4K2partialtest compares slot1 to independent T3K1prefix fromsameinitialstate, andfirst3attentionoutputs byteidenticallyperbackend; eachcomponentagainstCPUunchanged1e-7/repeatbits. Normal30435terminal0/replay1and0 each12/12pass (8ordinary+4partial)=768comparisons incl256partial. Deliberatewrongslot0 all4partialcasesfail while8ordinarycontrols pass; expectedexit1, validateswrongslotrejection evenCPU/NPUagree. Correctedslot0chainmatrix rerunlaunched; no fullMTPqualification/promotion.

## Wrong restore target is rejected even on both numerically agreeing backends

85621terminal0 normalfusion1/replay1and0 each32/32pass2048 comparisons. Added test-only wrong_restore flag (defaultoff, runnerexplicitlyclears) replacingpre4snapshotwithpost4snapshotduringrollback. Build57162terminal0/testexeonly. Negative67577terminal1 asintended:16wrongrestorevariants fail,16normalcontrols pass. Attentionandstate original-vs-replay checks report bitdifferenceindependentlyonbothbackends; confirmsoracle rejects semanticrollbackerror ratherthanacceptingCPU/NPUagreementalone. DoesnotclaimfullmodelMTProllbackcorrectness.

Faultflagcleared, normalfusion0matrix launched38130confirmedlive. HostC9/DSP27hashesunchanged/reverified. Remaining partialsnapshotselection,convstate/KVrestore andactualfullmodelownership semanticsnotcovered bywholechunkfrozenactivationtest.

## Whole-chunk rollback oracle restores prior state and demands byte-identical replay

Build89249terminal0/testexeonly. Existingfixturechain now snapshotsstateafter9promptchunk, processes4, restorespre4state, replayssame4, thenprocesses1decode. ExplicitCPYdependenciescaptureimmutablepre/poststates before persistentoverwrite; checks original/replayedattention/state bytes independentlyonCPU/NPU, componentsagainstCPU1e-7 andunchanged-inputrepeatbits. Persistent/distinctcache,K1/2,layouts,zero/seededinitialstate =>16rollbackvariants plus16normalcontrols32repeats. Fusion1/replay1all32/32pass1024comparisons; session85621replay0confirmedlive. Wholechunk/syntheticinitialstate coverage, notpartialMTPacceptance orconv/KVrollback. Next wrongrestoretargetnegativecontrol mustfail tovalidateconsistencygate ratherthanmerelyCPU/NPUagreement.

## Persistent cache overwrite did not reproduce frozen-chain instability

Added persistent_cache1 alongside distinctbuffercontrol to opt-in chain. Samecachepool overwritten aftereachchunk9/4/1; nextGDN takesCPY-producedlastvalidsnapshot. K2T1 writesonlyslot0, avoidsreadingstaleunusedslot1. Runtime confirmed CPY0/1/2 sharebacking(exampleK1data000002CDB04C2000), notmerelynamedreuse. Sixteen cases(layout,state,K,cachepolicy)32repeats each. Build3022terminal0/testexeonly; installedhostC9/DSP27 unchanged.

Fusion1session24367/fusion0session40056 terminal0, eachreplay1/0 16/16pass; total2048 strictCPU/NPU comparisons withperchunkattention/finalstate gate andperbackendrepeatbyteidentity. SimpleGDN+CPYpersistentreuse withrealactivationsanddeclaredsyntheticinitialstatepasses; doesnotqualifymodelstateinitialization,allocatorcross-opreuse,convstate,orMTPsnapshotrestore/rollback. Thoseownershipseamsremainnexttargets, no productionpromotion.

Carried-state fusion0 control30907 terminal0/replay1and0 each8/8pass. Combinedfusion1/fusion0 total1024 strictchaincomparisons; no livejob orfullmodelpromotion.

## Frozen9->4->1 activations with evolving copied state pass HVX

Added opt-in existingtest gdn_chain=1 using ordered capturedfixtures9/4/1 and zero/seededsyntheticinitialstate. EachnextGDN depends on precedingCPY output lastvalidsnapshot; eachattentionchunk andfinalstate separatelygated, repeatbyteidentityrequiredperbackend. Distinctcachebuffers perchunk, notactualmodelinplaceallocation orMTProllback. Eightlayout/state/chunkK variants32repeats each; initialstatereset onlybetweenchains. Build91259/13262terminal0/testexeonly, installedhostC9/DSP27unchanged.

Fusion1/replay1and0 each8/8pass=512 comparisons, session26024terminal0. Runtime firstchunk GATED_DELTA_NET+CPY actuallyfused, subsequentCPYs separate; don'tinferall3fusedfromglobalFusion1. Fusion0control replay1 also8/8pass, replay0polledforcompletion. Passingchain supplements isolated1536passes butdoesnotexplainfullmodelvariation/excludecachealias/lifetime issues. Next broadenownershipcoverage via persistentinplacecache oractualprestate snapshots, keepingstrictgates.

## Aggregate NMSE hid seeded-state HMX attention failure

Fixtureoracle now gates attention and each stateslot separately using unchanged1e-7limit, and verifies repeatbyteidentity independently on bothbackends for unchangedinputs. Build75905terminal0/testexeonly. Strict T9HVX512pass plus actualT4prompt512pass andactualT1decode512pass=1536 CPU/NPU comparisons; bothlayouts,K1/2,zero/seededstate,replay1/0. Perbackendrepeatbits stable in these passingcases, notcrossbackendbitidentity.

StrictT9auto/HMX replay1and0 each4/8pass vs oldaggregate6/8: seededK1attention1.62984832e-7 fails while state1.30817204e-8passes; large low-error state had diluted aggregate. ZeroK1attention2.46165239e-7/state3.04427295e-7fail. Bothlayouts same, K2HVXpasses. Failedaccuracycases stopfirstrepeat, so no samebackendrepeatqualification for those failures. No thresholdloosening; doesnotexplainGDN1fullmodelvariation.

T4fixture records104..108 SHA B19E14F3C42F5ACD9E3D2185D64FD83CB7926B0916C5B7F67C0B5E828404F76B/132120bytes; T1firstdecode135..139 SHA DBEE0916B4CBB4F9213278EB4761804BC9B749F615D1B38A2DA89AB27DA879FF/33048bytes. Actualactivations butsyntheticstate. Next chain9->4->1 with evolvingstate andGDN+CPY/cacheownership boundaries, to test interactioncurrentlymissingfromisolatedfixtures. No promotion.

## Captured-model GDN activations: HVX512 passes; HMX zero-state K1 accuracy failure reproduced

Added opt-in GDNFX001 fixture reader to existing test_gated_delta_net; checks shape/payload length/finiteness, consumes Q/K already normalized, packs logicalinputs into contiguous or permutedleaves. Explicit zero or fixed-gridseeded synthetic state; not actual recurrentstate replay. Eight variants32 repeats each, K1/2; runner clears fixtureenv outsidefixturemode and prints dataSHA. Builds67375/5701 terminal0/testexeonly; installedhostC9/DSP27 unchanged.

Real layer10 prompt9fixture SHA5DE02A32, GDN1/Fusion1/replay1and0 each8/8passes=512 CPU/NPU comparisons. GDN2 replay1and0 independently each6/8passes: zero-stateK1 bothlayouts fail aggregateNMSE~3.04e-7 againstunchanged1e-7gate; attentionNMSE2.46165239e-7,state3.04427295e-7. Runtime K1hmx-chunked-scalar VTCM7690240; K2hvx-recurrent-scalar VTCM1048576. SeededstateK1aggregatepasses do not implybitidentity or allcomponents belowthreshold. Realactivationnumericalfailure supplements earlier randomHMXfixtures; no proofrounding-only or explanationforGDN1fullmodelrepeatfailure. Need frozenactualstate/ownership-boundary tests beforefullmodelcausalclaim. No promotion.

## Internal GDN captures remain stable; actual frozen activations exported for operator harness

29702 terminal0/bothCLI0 Hello12831requirednames131each/4061records. Firstdiff89=L19(prompt9) NMSE4.05889625e-05/max0.166457951. Prompt9 records62..88 all27 finite/identical0 including earlyboundaries, all12 internal block11 stages andL14. Logitsbits0 NMSE0.0058011769/max0.947532177; selected15(264/4602), margins0.10185051/0.000352859497. Internalblock11 captured differences occur only aftertokensdiverge here; newretention did not reproduce B/C's earlier localvariation. L14 varies later beforetoken15, so promptstabilityisnotwholeprefixstability. No promotion.

Exporter export_gdn_fixture.py produces compact GDNFX001 header(D128/Hq16/Hv32/T9) followedby logicalF32 Q/K/V/loggate/beta. From A/B firstmatchingprompt9 records73..77, both297240bytefixtures SHA5DE02A320F09AB42BD29CB4607FFC44040B316A381D5EA356C82D2A1E3977569. Five activations byte-identical; state intentionally absent. Exactframing/type/shape/finite checks, missingcaptureexit1/nooutput, existingoutputexit1/nooverwrite verified. Next existingtest-backend-ops opt-in fixturecases should consume already-normalizedQ/K directly, explicitly initialize zero/seededstate, compareCPU/NPU K1/2/layouts withrepeats. Not claiming actualmodelstate replay; frozenoperatorinputs separate kernels from fullgraph ownership/lifetime hazards.

## Same layout B/C isolates first observed variation to block11 attentionpath/state

72409 terminal0/CLI0 exact19-capture armc. AC firsttensor43=L7(prompt9), logitsbits0 NMSE0.004786738/max0.757945538, selected60(1061/353), margins0.00156021118/0.0250968933. BC firsttensor45 attn_residual-10(prompt9) NMSE2.24305207e-06/max0.00836181641; sameframe embedding+L0/L1/L3/L5/L7/L9 records38..44 finite/identical0, preceding records also equal. BC logitsbits0 NMSE0.00347167167/max0.708541393, selected15(264/4602), margins0.0288944244/0.151729584. Thus not merely changing probe layout: fixedlayout varies at different earliest ranges across repeated runs. BC specifically supports block11 attentionpath or its storedstate varying with same captured currentinput/history, not a unique GDNkernel fault. Next normalizedinput/projection/conv/predelta/gate/output boundaries at layer10, preserveexisting19. A retained state_predelta view is not immutable preopstate: later persistentstate CPY can modify backing before postgraph read, so use real snapshots before claiming frozen-state input identity. No promotion.

## Attention/FFN captures move first observed variation upstream; no culprit in blocks11/12

15800 terminal0/bothCLI0 Hello12819requiredcaptures131each. Firstdifference43 L7(afterblock8,prompt9) NMSE0.000114174124/max0.0111532509; embedding+L0/L1/L3/L5 same-frame38..42 finite/identical0,L7 strict1. Downstreamattn_residual/ffn_out layers10/11 also differ but inputs already vary, so no localstagefault established. Logitsbits0 NMSE0.00762000924/max0.976641655; selected15(4602/264), margins0.00382423401/0.0288944244. Retaining stages shifted first variation from block12 to block8; changing retention does not monotonically narrow rootcause. Next repeat exact19-capture layout armc and comparetoeachpriorarm rather than add more tensors immediately. No promotion.

## Prompt-time variation narrowed to blocks11..12 under retained layout

51935 terminal0/bothCLI0 Hello12814required captures131each. Firstdiff35 L11(afterblock12,prompt9) NMSE1.43940033e-06/max0.00520630553; same-frame embedding+earlythroughL9 records28..34 strictfinite/identical0, L11 strict1. L12/L14/later vary. Logitsbits0 NMSE0.000792570272/max0.428418756, selected15(4602/264), margins0.0829963684/0.0277080536. Afterblock10 throughblock12 or associated states varies before changed sampled inputs, not identified faulty operator/unprobed proof. Next preserve14/addL10 and attn_residual/ffn_out layers10/11 for attention vs MoE localization. No arithmetic/build change or promotion.

## Block15 midpoint narrows observed prompt variation to blocks11..15

12764 terminal0/bothCLI0 Hello12812captures allrequired present131each. Firstdifference31 L14(prompt9) NMSE7.30027696e-06/max0.0151357651. Same-frame embedding and earlyoutputs throughL9 records24..30 finite/identical0; L14 strict1. Downstream variations include real headinput47(prompt4) NMSE0.00519365353/max1.10332632. Logitsbits0 NMSE0.00557900575/max0.829092979; selected15(4602/264), margins0.0225811005/0.0288944244. Retained plan shows range afterblock10 throughblock15/states varying before any sampled input changes, not unique faulty kernel/unprobed rootcause. Preserve12/add outputsafterblocks12/13 next, no promotion.

## Identical block10 prompt boundary feeds varying block20 output

2645 terminal0/bothCLI0, Hello12811 retained boundaries allpresent131 each/1441 records. First L19 diff29(prompt9) NMSE3.40025016e-05/max0.139535546; same-frame embedding+L0/L1/L3/L5/L7/L9 records22..28 finite/identical0 and L19 strict1. L29/L39 differ too; result_norm prompt9 is empty, actual prompt4 record43 differs NMSE0.00366111492/max0.670762539. Logitsbits0 NMSE0.0138678706/max1.29138541; selected15(4602/264), margins0.0387077332/0.236562729. Scope22/max187 prompt+decode1..15 isolates shared inputs, not later changed prefixes. Variation already upstream of head; afterblock10 throughblock20/states under this layout. Retention caveat remains; next preserve11/add block15 midpoint. No promotion; installed hostC9E4D8A/DSP27DC648C hashes reverified.

## Actual embedding and early boundaries identical through token104

46752 terminal0/bothCLI0 corrected model.input_embed+L0/L1/L3/L5/L7/L9.917 records,7 per frame; warmup0..13,prompt9 14..20,prompt4 21..27,step1 28..34. Logits bits0 NMSE0.00205683904/max0.457926899, selected104(9764/47), margins0.363044739/0.0863609314. First tensor differences756..762=step105, after changed sampled inputs. Scoped skip14/max742 actual prompt+steps1..104 all742 finite/bit-identical0, exact required names verified. Thus all observed early boundaries throughblock10 remain identical during logits variation under this layout, not universal stability of unretained graph. Next preserve7 and add downstream L19/L29/L39/result_norm. Optional comparator --require-tensors now fails missing expected captures within scope; tested present0/missing1/outside-scope1, default unchanged. No promotion.

## Early-plus-mid pair: blocks1/2 equal while block6 varies during prompt

26053 terminal0/bothCLI0 Hello128. Intended embedding filter did not match actual qwen35moe model.input_embed (qwen3next uses model.embed_tokens); do not claim embedding was observed. Five retained outputs L0/L1/L5/L7/L9,655 records; warmup0..9,prompt9 10..14,prompt4 15..19. First diff12 L5(prompt9) NMSE0.000140893768/max0.0148127079; L0/L1 records10/11 strictfinite/bit0, L5 record12 strict1. Logitsbits0 NMSE0.00337936964/max0.72514087, selected19(1438/12622), margins0.0701026917/0.106679916. L0/L1 later differences follow changed tokens; no firstblock fault evidence in this plan. Suspect range afterblock2 throughblock6/states, under retention caveat. Correct embedding name and add L3 afterblock4 next; no arithmetic/build change or promotion.

## Block8 retention changes failure pattern; no unconditional range narrowing

12796 terminal0/bothCLI0 Hello128 retained L1/L5/L7/L9. Logits bits0 NMSE0.00167951521/max0.396555126, selected112(0/65915), margins0.0625495911/0.0116291046.524 records order4 per frame, warmup0..7,prompt9 8..11,prompt4 12..15. First differences56..59=step11 all4 outputs; L1 NMSE0.0045278368/max0.00885372609. Scoped prompt+steps1..10 skip8/max48 all48 finite/identical0; first varying frame56/max4 strict1/all4 differ. Token inputs still common until selected112. Earlier block2 now varies, whereas previous plan stayed stable through token15 divergence and block10 varied in prompt. Retention materially changes the observed pattern; neither uniquely faulty block nor universal stable prefix established. Initial logits0 variation occurs while these boundaries are still equal. Next include embedding/L0 to inspect early-boundary relation within one plan, not claim probe stability is a production fix.

## Block6 identical, block10 varying already during prompt

20466 terminal0/bothCLI0, Hello128 L1/L5/L9 retention. Logits firstbits0 NMSE0.00964649462/max0.995898724; selected15(4602/264), margins0.054895401/0.0100212097. 393 records, warmup0..5/prompt9 6..8/prompt4 9..11/step1 12..14. First L9 diff8(prompt9) NMSE4.31836267e-05/max0.0101788305, same-frame L1/L5 strictfinite/identical0 and L9 strict1. Prefix scope skip6/max51 includes prompt and steps1..15: all34 L1/L5 records identical, all17 L9 records different. L1/L5 firstdiff57/58 step16 follows token15 divergence. Thus blocks7..10 or their associated state vary with identical block6 boundary under this plan, starting during prompt rather than only decode. Retention changes lifetimes; do not stitch differing probe plans into unconditional root-cause proof. Next add L7(block8) while retaining current boundaries. No speed/correctness promotion.

## Identical layer2 boundary feeds varying layer10 output at step8

67647terminal0/bothCLI0 Hello128L1/L9retention. Logitsbits0NMSE0.001300401/max0.351103626, selected111(0/65915), margins0.0973625183/0.107374191,112commonprefixrows.262records/37equal225different overall; laterprefixchangesnotrootcauseevidence. Actualtwo-boundaryorderconfirmed: warmup0..3,prompt9records4/5,prompt4records6/7, step1records8/9. FirstL9diffrecord23=step8NMSE0.0031238361/max0.0115880258; L1record22sameframebitidentical/finite exit0, L9record23exit1. L1remainsidenticaluntilfirstrecord52=step23NMSE0.0023416319/max0.00292944908. Bothdifferencesprecedeselected111, so notjustchangedtokeninputs.

Atstep8 variationliesbetweenoutputafterblock2andafterblock10orassociatedstoredstatesunderprobe; currentinputboundaryanditspriorhistoryareidentical. Initiallogitvariationat0predatesL9first8, so thisdoesnotexplainallvariationasanexclusivefaultinthisrange. RetentiondifferentfromearlierL0/L1pair prevents universalearlyblockstabilityclaim. Next3compactboundariesL1/L5/L9 (after2/6/10),128tokens, midpointnarrowingwithoutmultiGBstateexports. No promotion.

## Two-boundary pair shows both earliest blocks stable before selected divergence

30679terminal0/bothCLI0 Hello128RetainL0/L1/logits/headcap0, hashesC9E4D8A/27DC648Creverified. Logitsbits0NMSE0.0059607497/max0.752601147; selected56(271/198), margins0.0210227966/0.249969482,57commonprefixrows.262records orderL0/L1 pairs:0..3warmup,4/5prompt9,6/7prompt4,8+decode.120equal/142different, first120/121 mapstep57 AFTERselected56divergence. Scope skip4/max116 coversactualpromptandsteps1..56, all116equal/finite strict0. Thusbothfirsttwoblockoutputsremainidenticalwhileinitiallogitsvary; theirlatechangesfollowdifferenttokensanddon'tidentifyfaultyblock. Unobservedstoredstates/lifetimes andretentioneffectsremaincaveats. Diagnostic26.04/26.14tok/s notpromotion.

NextsimultaneousL1/L9boundaryprobe128 tocheckoutputafterblock2enteringrangeblocks3..10andoutputafterblock10withinsameplan. Do notcombine stable boundariesfromdifferentprobes asproofsameunprobedexecutionstable.

## First block output is stable before token divergence under one-boundary probe

12013terminal0/bothHello128CLI0, l_out-0onlyretention/headcap0/currenthostDSP. Logitsfirstbits0NMSE0.00141180521/max0.368807673, firstrawargmax95, selected104(9764/47), margins0.0684356689/0.0611820221; compared105sharedprefixrows.131firstlayerrecords:0/1warmups2rows,2prompt9,3prompt4,4..130decodeonecolumn.108equal/23different firstrecord108=step105, AFTERtoken104divergence. Strictscope skip2/max106 promptandsteps1..104 all106byteidentical/finite (exit0). Thereforelaterfirstlayerdiffsarecompatiblewithchangedtokeninputsandnotfirstblockdefectevidence. Currentfirstlayerhiddenidenticalwhileinitiallogitsvary suggestsvariationafterfirstblockorotherlayerstatesunderprobe, notproofeveryunprobedfirstblockoperationcorrect.

Source l_out-0 followsattentionresidual/MoEFFNresidual/cvec, notGDNstate-onlyboundary. Next simultaneouslyretainl_out0/1,128tokens tocompareadjacentinputs/outputswithinsameprobeplan. Differentretentionplanscanalterlifetimes, allpromotiongatesremainunpassed.

## Midpoint probe separates earliest observed logit and hidden-state differences

97615terminal0/bothCLI0 Hello64mid20onlyretention/logits/currenthostDSP/headcap0. Logitsfirstbits13 NMSE0.000112541827/max0.166591644, all64selectedtokenssame. Tensor67records:0/1warmup2048x2,2prompt2048x9,3prompt2048x4,4..66decode2048x1.17equal/50different, firstrecord17 NMSE0.000244394218/max0.00946378708. Record3lastpromptcolumnfeedsstep0, thenrecord4step1;record16step13 bitidenticalandfinite strict0, record17step14 differs strict1. Atfirstlogitdifference midpointoutputstillidentical; post-midpointcomputation/storedstatevariesunderprobe. Followingstepmidpointalsovaries. Doesnotexcludeearlierunobservedrecurrentstatesoridentifyfirstbadop; differentretentionplanthanhead-inputprobe preventsnaivebisectionofunprobedgraph.

Tensorcomparator --require-finite optional(defaultoffsoattention-infmasktracesremainvalid). Positiveactualself67finitepass/identicalNaNnegativecheckrejects. Next128tokenfirstlayer-onlyprobe improvesintermittencycoverage, no callback/all-layerretention. No promotion.

## Vocabulary-head input varies before any generated-token divergence

98885terminal0aggregate, bothHello64CLI0/headcap0normalpolicy/currentC9E4D8Ahost/27DC648CDSP; retainedonly^result_norm$ pluslogits.67tensorrecords:0/1warmuponecolumnidentical,2emptyresult_normcheckpointidentical,3firstactualpromptheadinputdiffNMSE0.000264430944/max0.24225235. Headinputvarianceexistsupstreamofheadunderprobe, so vocabularyhead-onlykernel explanation isinsufficient. Retentionstillchangeslifetime, notunprobedfirstoplocalization.

Logitsfirstbits0NMSE0.000414554051/max0.204733849; selected15(4602/264), margins0.0902500153/0.152362823. Fulltensorreport3equal/64differentcontainslatertoken-prefixeffects; onlyactualrecords3..18 (steps0..15) shareprefix. Newoptionalcomparator --skip-records3 --max-records16 strict1/all16diff, excludesstep16+largeinputchangesaftertokendivergence. It stillreadsfullframing/counts andchecksrequestedscopelength; defaultunchanged. No correctness/performance/MTPpromotion. Next upstreamboundaryprobe, notcontinuedhead-depthtuning.

## Head prefetch depth2 also shows no clear stage win

56325terminal0/all3CLI0, cap2actual2/VTCM1984512.15completeone-token1384-op batches: headmean2953.8/median2933/min2928/max3094/stdev45.574us; last2933/finaloperatorsum33.062ms. Compared8median2945/4median2934us: near-identicalcost, notrepeatedindependentpaireduntracedbenchmark. Cap2reduceslayout2752512bytesvs8butdoesnotdemonstratelatencygain or modelstability. No reason topromote shallowerdepth fromtheseprofiles. Default0normalpolicy retained.

Next selectivehead-inputcapture repeatsHello64withonlyresult_norm retained andlogits, normalheadpolicy. Capturedinputvariationcanestablishupstreamnondeterminismunderprobe; stablecapturecannotexcludeoriginalhazardbecauseOUTPUTretentionchangesalloclifetimes. Never qualifywholemodelcorrectnessfromprobe-suppressedrepeatability.

## Reduced head depth4 lowers VTCM without a demonstrated latency win

Cost-only serial56325 Hello16 basicprofile/MM2/FA1/GDN1/routerHVX/UB128/defaultworkers/fusion1/no logits/tensorretention. Caps8/4 bothCLI0; runtimeactualdepth8/4, VTCM4737024/2902016. Newparser checksselectedopcount1384complete andexactonehead2048:248320 x2048:1;19recorded/15matchedone-tokenbatchesperarm. Warmup/checkpointexcluded. Instrumentedhead8 mean2957.133 median2945 range2928..3123 stdev48.022us;4 mean2942.800 median2934 range2929..3030 stdev25.021us. Lasthead2929/2930; finalopsums32.973/33.240ms. ~11usmedian difference doesnot establishspeedwin; within-run samplesnotindependentrepeatedbenchmarks, unstablemodelqualificationnotpassed. VTCMreduction1835008bytes factualbutnotproofsystembenefit. Cap2actual2/VTCM1984512runpending. Parserpositive/incomplete/wrong-shape synthetic casespass.

## Headcap8 control also varies; ABBA does not attribute correctness difference to cap

59624 terminal0aggregate/all4CLI0. Cap8-a/b strictrepeat1 firstbits0 NMSE0.000346027369/max0.213671744; selected15(264/4602), CPU-independent repeat margins0.0288944244/0.00694847107; only16commonprefixrows. Cross8a/4a1, cap4repeat1 (bits0 butall64selectedsame), cap8repeat1. Existingintermittencyremains; cannotclaim cap4causedcrossdifferenceor cap4stabilizedmodel. Traced rates26.96/27.23/27.00/27.14 inABBAorder arediagnosticnotuntracedpairedbenchmark, neitherarmqualified. Nextcaps8/4/2basicprofile16verifyactualselectionandisolatedheadcost; usefulcostdataevenwhile modelcorrectnessgateunresolved, notpromotion.

## Headcap4 same-depth repeat is not bit-stable

Independent read-only cap4-a/b full64strictrepeat exits1: firstbits0 NMSE0.000177878099/max0.188775241, all64selectedtokensidentical/no rawargmaxdifference. Botha/bapplicationcleanup complete while serial59624 cap8-b pending. First cross8/4tokenfailure cannot independently prove cap4arithmetic/schedulingcausality because samecap4 itself varies. No logit/tokencorrectness/performance promotion; awaitcap8same-depthcontrol. Model instability remains.

## Headcap8 vs4 initial64 cross gate fails, attribution pending repeats

Serial59624 Hello64 expandedmodel FA1/GDN1/routerHVX/MM2/UB128/defaultworkers/fusion1/replay1/logitsonly. Both firstarmapplicationcleanup complete, fullHTP19037MiB/noCPUmodel. Read-only strictcross compare exit1: firstbits5 NMSE0.000193934919/max0.152682781; selected15 reference264/candidate4602, margins0.0288944244/0.0435295105, NMSEatselected0.00759991307/max1.15350521; only16commonprefixrows. Existingmodelnondeterminism means no cap4rootcauseclaim until bothsame-depthrepeats. Full64cross gate failed, no promotion. Host/DSP hashesC9E4D8A/27DC648C reverified. Actualselectiondepth needsprofilelog, requestedcap metadata aloneinsufficient.

## Bounded head-prefetch cap experiment implemented, not qualified

Default0 preserves existing policy. Hostenv GGML_HEXAGON_HEAD_PREFETCH accepts0/2/4/8/16 (other integer values warn/reset0); applies only ordinary quantized HVX one-row Q6_K output.weight withinput2048/output248320. Caps maxprefetch before existing fit/solver; DEBUGprobe logsactualselecteddepth/VTCM, doesnotforceunsupportedlayout. No quantization/compute/DSP/protocol change. Runtime runner resets0 bydefaultandrawdecodeexplicitlyresets0; baselineenvleakprevented.

BackupEDB9FFhostDLL+dirtysource retainedhead-prefetch-prechange. Initialbuild80771 failed undeclaredne01, correctedsrc0->ne[1]; build49937 terminal0, onlyhostggmlhex installedC9E4D8ADE6884DA62DF7A597DCEAC6582AC036236C06134231A405BAF8E6399E, installedDSP27DC648C unchanged. OtherarchDSP/buildcatalogrebuildsideeffectsnotinstalled. Serial59624 ABBA64 cap8/cap4/cap4/cap8 Hello diagnosticpolicy logitsonly, strictcross/bothrepeatgates pending. No new performance/correctness promotion, full128/CPU/MTP gates remain.

Head scoped trace peak DMA intervals=8 on all8workers, not16; this observes queued residence depth, not a direct kparams dump. Future sweep must compare againstactual8-depth behavior and verify layout rather than labelcontrol16 frommaxpolicy. Zero-length spans excluded from peak half-open interval accounting.

## Per-op transfer/compute intervals identify shape-specific experiments

Reader now aligns trace/op clocks to64bitbatchstart moduloUINT32, checksall events inside batch, validates completeness before exact first-tensor --tensor selection, clips spans toopwindows, reports peak overlapping intervals. Synthetic wrap/clipping expectedDMA[10,20]/HVX[12,20] passes; head/blk0gate/blk0down real parseexit0, zero badclocks. Not an arithmetic/kernel change.

Single-token output.weight Q6_K head2.937ms/VTCM4737024: perworker HVX_COMP1.917..1.928ms, DMA residenceunion2.910..2.911ms, overlapHVX1.898..1.909ms, outside1.001..1.013ms. Quant0.306..0.375us perworker; small comparedwithhead. Theseare instrumented, overlapping residence intervals, NOT exposed DMA-engine stalls or benchmark savings. Host maxprefetch16 doesnot establishactual16; VTCMfit canselectless.

Sample blk0 expert gate/up126us: thread0 quant0.387us, HVXcompute63.583us, DMAunion75.805us/outside44.015us. Expert down87us: quant4.271us, HVX42.043us, DMAunion61.693us/outside24.660us. Samples notall-layer bounds. Prefer per-shape transfer/scheduling experiments; don't infer globalquant optimization fromoneaggregate or blindlydeepen top8prefetch. Fullmodel instability remains, candidate27DC648C unpromoted.

## Complete single-token DMA/HVX trace; correct four-token batch misclassification

Capture74097 batch3 first matmul dims2048:4, so1994ops/75.394ms is four-token prompt/checkpoint work, NOT single-token early decode. The previous comparison caveat stands, but hypothesis of early one-token fusion loss is unsupported. Batch4 actual dims2048:1 uses1384ops, fused MM+ADD/WIN present. Batch35.249ms/operator sum34.364ms/head2.937ms under mode3. Reader exit0 all330566 reported/observed events match (47080,40738,40538,40538,40458,40458,40458,40298,0,0,0), no saturation/missing/unmatched/open events. Configured replay1 does not execute replay when profiling.

Thread0 traced activation quant145.162us/prep43.769us, HVX_COMP13.493ms, INIT295.861us/L2FLUSH435.486us. DMA submission-to-pop residence sum102.300ms but union20.601ms; overlap with traced same-thread HVX10.750ms, outside9.851ms. Outside range acrossworkers8.608..9.851ms, NOT DMA-engine busy or measured stall. Relative conversion uses average2112.063MHz; these overlapping thread/category totals cannot be added into walllatency. This single-token observation makes global quantization-only speed explanation weak, but does not exclude per-op quant hotspots or untraced prep. Next isolate per-op head/matvec transfer-compute overlap and prefetch depth with unchanged arithmetic, before whole-model paired performance and strict repeat/MTP qualification. Existing full-model repeat failure remains, candidate unpromoted.

## Early single-token trace plan differs from final16-token profile

Live capture74097 batch3 header1994ops/75.394ms, operator sum74.214ms; event counts below131072, but dump not terminal/completeness unverified. Early graph includes CONCAT/SSM_CONV/CPY and separate ADD instead of prior1384-op final16-token fused WIN/MM+ADD plan. Do not attribute75vs33ms to trace bookkeeping alone or compare as same-plan performance. Host source want_replay requires opt_profile==0; configured replay1 does not establish actual replay under profiling.

Provenance audit after apparently different warmup types: host SHA EDB9FF528857E8D0BB0622343549891FA576BD529485D6A362D2176D0072DC0D, DSP27DC648C23A3B0CA97947D89477B9D095E85717BD26DD8A9A731E26D7F92D20B match; expanded model20837295744bytes/mtime2026-09-26 20:52:12, loaded path correct. Prior stages16 and current trace both loader q4_1=5/q8_0=105/q5_K=60; existing findings inventory also confirmsmixed types. No evidence external model mutation from this observation. Full model hash not reread during active inference to avoid competing20GB I/O; complete provenance verification remains available after terminal capture.

## Trace overlap tooling and prefetch hypothesis refined

Capture74097 prompt batch2 has thread0 reported131072 at configured capacity; completeness not proven, do not optimize from truncated prompt events. Actual final decode capture still pending. Reader computes interval union intersection of DMA residence and all traced same-thread HVX categories; outside-HVX residence is not DMA-engine stall (may include untraced work, dispatch, barriers). Wrapped synthetic overlap8cycles/outside2 validated; multispan intersection11cycles validated.

Host ggml-hexagon.cpp quantized HVX selection around6117/6152 and fused path6863 uses max_prefetch16 only for src1_nrows<=HTP_MM_HMX_MIN_NROWS, otherwise2, then chooses deepest fitting VTCM layout. One-token top8 experts can have8activation rows, so this is not a universal16-depth decode policy. Hypothesis for later bounded experiment: vocabulary-head-only prefetch depth sweep16/8/4/2, verify actual layout/placement, unchanged model/arithmetic, strict repeated logits/CPU selected gates and paired uninstrumented latency. No speed conclusion from fit or overlapping DMA sums; no production change yet.

## Per-thread trace completeness verified on warmup, not decode

Live capture74097 first completed batch2054ops/58.590ms follows explicit warmup log. Correct earlier STATUS prefill label: no actual prompt/decode classification proven for this batch. Reader --batch0 reports observed counts exactly equal77498,73652,65454,65454,65254,65254,65254,65174,0,0,0; no saturation at131072/thread, unmatched stops, or open starts. Real-trace parse exit0. Thread0 DMA interval sum131.538ms, union33.344ms, HVX_COMP25.322ms; intervals overlap and DMA includes time queued until pop, not DMA-engine busy or exposed stalls. All conversions use batch-average cycles/usec, not per-event DVFS calibration. Must analyze actual final single-token decode batch before optimizing from these durations. Reader selected-batch streaming storage and negative completeness gate validated; no model arithmetic change.

## Current candidate fused-bias regression gate and stage breakdown

Session73399 terminal0, candidate27DC648C: fusion1/profile1 MM bias12 cases x32 x replay1/0 =768 stock-threshold passes. Actual fused MUL_MAT+ADD/hvx-tiled path verified in replay0 log. Matvec source audit: bias DMA pop_nowait removes the descriptor from software queue, then weight DMA completion waits occur in same chain before final bias addition; this is not standalone proof of a race. Chain ordering still must satisfy hardware contract, no speculative wait patch made.

Plain expanded-model prose16, MM2/FA1/GDN1/routerHVX/UB128/defaultworkers/fusion1/replay1, basic device profile and no tensor/logit retention: CLI0; last1384 operations sum33.197ms across19 recorded batches. MUL_MAT10.552ms (Q6_K vocabulary head2.938), MUL_MAT_ID_NX4.671, MUL_MAT+ADD3.009, MUL_MAT_ID2.707, GDN+CPY1.706, MUL_MAT_NX1.538, SSM_CONV+WIN1.470, GET_ROWS1.420, RMS_NORM+MUL0.852, FA0.414. This is an instrumented device sum, not uninstrumented token latency or paired speed comparison. Operator totals do not separate DMA, activation quantization, and compute; profile mode3 exposes per-thread trace events for targeted follow-up. Full-model repeat gate remains failed, no candidate promotion.

## DMA-contract focused attention and state controls pass

Session91496 terminal0: fusion1, FA1 eight batch/alias cases x32 with replay1 and0 =512 comparisons; GDN1 eleven model-shape/state cases x32 with replay1 and0 =704 comparisons. All pass unchanged stock gates. These1216 additional focused passes exclude regressions in the tested cases, not full-model repeatability, arbitrary lifetimes, or real-MTP rollback qualification. Full-model negative result remains decisive against promotion.

## DMA compiler-contract repair is insufficient for model repeatability

Session98433 completed both plain expanded-model prose128 runs with CLI0, MM2/GDN1/FA1/routerHVX/default workers/UB128/fusion1/replay1, logits only. Strict comparison exit1: first bits1 NMSE0.0027049565/max0.711775303; selected6 differs430/321, reference/candidate margins1.19086647/0.4793396. Only7 common-prefix rows compared before token divergence. Diagnostic26.97/26.95tok/s is not a performance acceptance result. Focused1280 passes remain valid but do not establish model stabilization or compiler reordering as root cause. Candidate27DC648C unpromoted, prior signed backups retained.

## DMA compiler-contract candidate builds; focused tests pass

DMAstart/linkreleaseasm nowmemoryclobbers, completionpopemptyasmcompilerbarrieraftervolatile donecheckcoversfastpath; no newhardwarewait/syncht/cacheflush, pop_nowaitunchanged. Baselineemittedreorderingwasnotproved, so candidateiscontractrepairnotprovenmodelrootcause. Prior1932583E/catalog/headerbackedupdma-contract-prechange. DSP-onlybuild32740terminal0 installed27DC648C23A3B0CA97947D89477B9D095E85717BD26DD8A9A731E26D7F92D20B, signtoolcatalogmembership0. NewmatmulLTOIRrelease/dmlinkconstraintscontain~memory. Hostlibsnotcopied. Reusefusion1/replay1and0all20x32pass1280stockcomparisons. Fullmodelrepeat/CPU/MTP/performancequalificationpending; no promotion.

## Combined singleworker/microbatch4 still fails selected token6

Candidate1932583E/plainprose128a/b MM2/GDN1/FA1/routerHVX/fusion1/replay1/UBatch4/HvxThreads1, bothCLI0/fullHTP19037.92, actualthreads1andub4. Strictrepeat1firstbits5NMSE.00432204872/max1.64874601; selected6(430/321), margins.451276779/.591995239;stopsafter7commonprefixrows. Diagnostic7.77/7.78tok/snotpromotion. EarlierUB4/8workerprofileprovedallmatmulHVX, sameUB4selectionrequestedhere; nofreshallkernelprofilefor1worker. Combinedrequestdoesnotstabilize, so multiworker/HMXprefill-only explanationsremaininsufficient. DMApublicationcompilercontractauditseparateunprovenhypothesis.

## Model descriptor audit does not establish adjacent-line hazard reachability

Handle21694terminal0, verbose+DeviceProfile1/plainprose16 expandedmodel/MM2/GDN1/FA1/routerHVX.19descriptorsegments18624records,8uniqueunalignednonweightpatterns allconv_state_last0/1 offsets4/8/16/80. Zero byte-disjointsharedlinepairswithinrecordedsegments. Segmentation/logcoverageandactualdirty/lifetimeslimitnegativeclaim; no proofallmodeladdressesalwayssafe, but focusedhazardNOTestablishedascurrentmodelrootcause. NoOUTPUTretention/callback, logging/profilecanaltertiming. Lastbatch1384ops33.440ms:MM10.579(head2.926),routedgateup4.708,MMADD3.003,routeddown2.730,GDNCPY1.723,WIN1.630,MMNX1.532,GETROWS1.445. Instrumentedonebatchnotpaireduntracedspeedclaim. Nextinternalmatmulactivation/scratchownershipaudit, beyondtensor-leveldirtytracking.

## Candidate prose CPU token gate still fails14; password untested

Frozen suite1932583E/defaultFA2GDN2/routerdefault/normalGGUF: math/code32pass, proseCPU/NPUCLI0butrequire-tokens1firstselected14(2014/15771), reference margin.324108124/candidate.343389511 reversed. Commonprefixrow14NMSE.00447142799/max.887376785. Samefailurestep/IDsasearlierEFCbaseline, notsamefull-logitproof. Suite34208terminal1stopsonprose,passwordNOTexecuted:2pass/1fail/1untested, not3/4. Diagnostictraced28.59tok/snotpromotion. CachelinefixvalidfocusedbutdoesnotqualifymodelCPUcorrectness/repeatability.

## Cacheline candidate code32 CPU selected-token gate passes

decode-correctness-v1 codecase CPU/NPUCLI0,require-tokens0all32, installed1932583E/defaultbaselinepolicy/normalGGUF. Crossbackendfirstbits0NMSE.00146429343/max.489736795; noCPUbitidentityclaim. Math/codepassed2cases, prose/passwordpending. DiagnosticNPU28.75tok/s traced/notpairedspeedacceptance. Remainexperimental: fullmodelrepeat alreadyfailed selected52 underdiagnosticHVXpolicy.

## Cacheline candidate math32 CPU selected-token gate passes

Frozen decode-correctness-v1 firstcase math: current1932583EDSP/defaultFA2GDN2/routerdefault/fullHTP19037.92/8workers/replay1/fusion1 normalGGUF unchangedhash. CPU/NPUCLI0 andrequire-tokens0all32. Numericalfirstbits0,NMSE.00187819494/max.466886163, crossbackendbitidentitynotrequired. DiagnostictracedNPU28.70tok/s notpairedspeedclaim. Remainingcode/prose/passwordpending; onecasepassnotfullqualification.

## Cacheline candidate fixes focused bug, not fullmodel repeat instability

Installed1932583Ecatalogmembershipverifiedsigntool/pa/c success; hostggmlhexEDB9FFandllama981E8D unchanged. Prose-cacheline128a/b CLI0/fullHTP, strictrepeat1: firstbits0NMSE.00253665054/max.642757416; firstselected52(271/198), margins.00100517273/.201197624, stopafter53commonprefixrows. Diagnostic26.71/27.04tok/snotpaireduntracedspeedclaim. Focusedcachelinefix1280pass doesnotestablishmodelrootcause or modelcorrectness; additionalinstabilityremains. Candidateexperimental, baselineEFCrollbackretained.

## Cacheline-overlap candidate fixes focused adjacent DMA failure

Explicitproducerhandoffnegativecontrolreplay1/0each18/20: same64offsetDMA passeswhenproducerconsumedbeforeoverwrite, onlyno-handoffaliasesfail unchanged. SupportsmaintenancecauseovergenericunalignedDMA. AdjacentADDstaticDMAgeometry differsfromoriginaldynamicreadarm.

Saved EFCbinary/catalog/source in cacheline-prechange. is_tensor_dirty candidate compares128lineboundswith64bitintermediates, skipszero-sizedtensors; retainsbyte-rangebookkeepingconservatively. DSP-onlybuild60993terminal0 installedsigned1932583EB642704F72CEA77A5FF7DDC9EFC24FB24E93627656A4F1A3A0C4B1ED, nohostlibrarycopies. Focusedreusefusion0/replay1and0all20x32 pass1280stockCPUcomparisons, includingpreviouslyfailing64offsetaliases. No arithmeticchange/tolerancechange. Candidatefixesreproducer; fullmodelrootcause/repeatability/CPUcorrectness/performanceunproven. Modelpairlaunchedseparately; notpromoted.

## Focused byte-disjoint same-cacheline alias failure reproduced

ExistingDMAreuseoracle adjacentarm: F32MMoutput64bytes atbacking0, unrelated concatDMA atbacking64, RMSconsumer. Replay1/0each original12cases10pass/twoaliasfail:rows1NMSE.005390220,rows9.000873050 vs1e-7. Oldoverlap/disjointcontrols pass32. Actualrows9produceraddress...1000 size64,DMA...1040 size147456, backingcapacity147584; no byteoverlap but128line shared.

Added128offset controls,test-onlybuild77229complete. Extendedreplay1 14/16: only64offsetaliasesfail;128offsetaliasesanddisjointcontrols pass32each. Supports cacheline-sharing hypothesis, not yet exclusionofallunalignedDMAhandling defects; next forceproducerhandoffbefore64offsetDMA andverifyactualkernel/flags. Firstattemptbackingorderassert wasconstructionerrorcorrectedbeforetheseexecutions. NoDSPpatch/modelrootcause/speedpromotion. Modelallocator128aligned butviews canhaveoffsets; modelreachability requiresactualaddressproof.

## Full-NPU HVX-only prefill control still diverges selected token33

Microbatch4 profileCLI0: n_ubatch4, HTPweights19037.92MiB/noCPUweights,9678matmulprofileentriesallhvx/zeroHMX, finaldecode1381ops32.067ms. Unprofiledprose128a/b bothCLI0 strict1, firstbits5 NMSE.000923040539/max1.01711631, selecteddifference33(430/303), commonprefixcomparisonstopsafter34rows. Marginsreference.198802948/candidate1.47491455, not merely tiny sampler tie. Diagnostic27.60/27.22tok/snotpaireduntracedpromotion. Smallerprefillalsochangesrecurrence/fusions/allocator, so not HMX-only causalexperiment; nevertheless HMXprefillnotnecessaryforthis policyinstability. MM2/fullNPU avoids invalidMM1CPUrepackdiscriminator. No productiondefaultchange.

## Exact-grid model-width matmul prefill really executes HMX

Auditprefill-exact-grid-replay-1.log.err confirms12newF32/Q4_0,k2048,rows9/13/24/32/33/64 execute-op hmx-tiled VTCM8038400..8366080. Replay0 log384actualhmx-tiledexecuteentries (12casesx32). Botharmsnewcases384stockCPUcomparisons pass768total; oldrandomQ4single-tokencontrols fail5.454e-6/5.483e-6replay1, not ignored or tolerancerelaxed. Identityweights/binarygrid do not cover dense accumulation, arbitraryaliases or fullmodelstate; this negativecontrol doesnot reproduce HMX tiletailcorruption.

## Separate GDN state writeback still varies across plain128 repeats

Handle89531 terminal0 aggregate, profile/plaina/plainb CLI0. Mask8126 actualfinalprofile1414ops33.270ms:30GDN1.577ms+30CPY.927ms=2.504ms versus fused30GDNCPY1.746ms in previousallfusionprofile.30WIN retained. Separatewriteback adds30ops and.758msstagedifference; totalprofiledelta.189ms includesotheropvariation. Unprofiledplain128pair strict1: firstbits10 NMSE.00036859493/max.324617147,128tokensidentical; diagnostic26.29/26.56tok/s, not speedacceptance. Neither convwindow nor GDNCPY removal individually establishes stabilization; no productiondefaultchange.

## Profile verifies conv-window saves60 dispatches and2.973ms device stage sum

Paired prose16 DeviceProfile1 runs CLI0, sameexpandedmodel/MM2/GDN1/FA1/routerHVX. Finaldecode allfusion:1384ops,sum33.081ms,30SSM_CONV+WIN1.612ms. Mask8062:1444ops,sum36.233ms,30CONCAT2.971ms+30CPY1.278ms+30SSM_CONV.336ms=4.585ms. NoSSM_CONV+WIN in mask finalbatch; othermainfusions/counts retained. Stage difference2.973ms,totaldifference3.152ms includesotheropvariation. DenseMM10.474/10.543ms,routedgateup4.585/4.641ms,MMADD2.966/3.003ms,routeddown2.661/2.674ms. Profile/warm finalbatch only, not untraced paired speed or first-difference correctness proof. Combined with no-window128 repeat failure, disabling convwindow is not a justified fix; retain productiondefault.

## Selectively disabling conv-window does not stabilize prose plain128

Fusion8062=8190minus128 (all other flag bits requested), expandedmodel/MM2/GDN1/FA1/routerHVX/replay1. Both plain128 runs CLI0, all128 selectedtokens same, strictrepeat1: firstbits0 NMSE.000714954633/max.330351353. Frozenreal-a versus no-windowplain-a alsofirstbits0, strict1, same128tokens. Diagnostic traced24.78/24.49tok/s versus allfusion26.62/26.67; not paired untraced speed acceptance. This removes one fusion request but changes graph/operation lifetimes too; runtime proof follows in separateprofiles. No numerical stabilization or production change justified.

## Second prompt: real MTP repeats and matches one plain run exactly

Prose sky explanation128 under MM2/GDN1/FA1/routerHVX/fusion1/replay1: plain a/b selected128 identical but first logitsbits3 (NMSE.00193076379/max.70274508), strictrepeat1. Real-a matches plain-a full128 target logits bit-identically strict0; independent real-b repeats real-a bit-identically strict0. Both real runs59/68 draft acceptance (9 rejected); comparator qualifies target rows only, not separate draft logs. Traced plain26.62/26.67 and real25.09/25.16tok/s are diagnostic, not speed wins. Plain numerical instability persists, so matching one plain run and real repeats do not establish CPU/full-prompt MTP qualification.

## Real MTP target128 repeats bit-identically across processes

mtp-real-allhvx128-repeat finished CLI0; strict comparison against frozen mtp-real-allhvx128 passes all128 target rows bit-identically with identical selected tokens. Both have52/74 draft acceptance. This comparator checks target rows, not separate draft-logit identity. Diagnostic23.07tok/s is traced, not an accepted speed result. Stronger real-path repeatability evidence, but unresolved plain128 instability and CPU correctness gates still prevent MTP qualification. No production changes.

## HMX GDN discrepancy is value-sensitive and not permuted-layout-only

Test-only build60637 complete. Added nine-tokenK1 contiguous random control plus structured contiguous/permuted controls: one-hot Q/K, gate0, beta1, state/value grid1/16,32seeded changes. AutoGDN2/replay0 yields9/11: random permuted fails total1.561e-6 (attention2.9149949e-7/state1.56796905e-6), random contiguous fails1.530e-6 (attention3.28569139e-7/state1.53669771e-6). Both structured variants pass32repetitions, actual HMXchunked verified. Passing9cases contribute288 graph comparisons; failures stop firstrepeat. Different layouts have different logical random/basis arrangements, so this is not exact same-input cross-layout identity. Structured controls simplify recurrence as well as arithmetic, hence support precision/value sensitivity but do not prove rounding-only cause or exclude all indexing bugs. Stock1e-7 retained, productionDSP EFC unchanged. Expanded forcedHVX11cases control also tested separately.

## HMX GDN discrepancy includes attention and final recurrent state

Diagnostic err override preserves total NMSE and1e-7 acceptance threshold. Repeated model-shaped auto/replayoff9/K1 failure splits attention NMSE2.9149949e-7 and state_slot0 NMSE1.56796905e-6. HMX source FP16 tiles make numerical precision a candidate explanation; no first incorrect operation or indexing bug established. GDN1 fullmodel repeat remains unstable independently, so this focused discrepancy is not the sole campaign failure. Test-only build56863 complete, sameproductionDSP retained.

## Nine-token model-shaped HMX GDN fails CPU oracle, HVX passes

Added optional32seeded-input repeats to existing GDN test at actual16key/32value heads,size128, group2/permuted, token1/2/4/9,K1/2. ForcedHVX replay1/0 all8 x32 pass512 graphs stock1e-7. AutoGDN replay1/0 each7/8, nine-tokenK1 NMSE1.561e-6 fails1e-7; failed repetition stream stops at firstfailure. Runtime HMX chunked scalar7690240VTCM versus passingK2 HVXrecurrent1048576. Replay not necessary for mismatch. No sole full-model explanation: unretainedGDN1 also already varied. Next split output attention/state error and inspect precision/indexing; do not loosen gate or call HMX discrepancy a proven indexing bug. Test-only build1297 complete, production EFC unchanged.

## Direct dynamic scalar table read survives reused DMA output

Existing dma_dead_reuse scalar_read arm now uses dynamic scalar broadcast, not unproven fused primer: F32MMdead[2048,2*rows] -> reshape[1,2048,2*rows] -> ADD with[32,2048,2*rows] -> unrelated CONCAT reusingdead -> RMS. Host predicate selects SCALAR notSCALAR_DMA because ne2>1; DSP worker uses global src1_plane. Fusion0/profile0 replay1/0 both8cases x32 pass512graphs,256scalar arms. Fusion1 same arms also512pass,1024total/512scalar arms. Exact alias address old/table/overwrite000001B31F179100 (width2), actual HTP ADD shapes and replay capture verified. No stale-output reproduced with direct read; no global cache correctness claim. No quant primer/tolerance change, DSP EFC untouched; testexe only.

## Primer tests pass, but runtime disproves intended clean-read coverage

Extended8case oracle replay1/0 passes512 comparisons with F32 primer; width2 unfused, width18 HMX. Revised zeroQ4 weights (exact zero contribution, no tolerance relaxation) and forcedMM1 passes another512; runtime confirms HVX/HTP but no MM+ADD fusion for rawQ4. Hence these cover additional residual/DMA alias sequences but not asserted direct CPU cached residual read. Do not promote clean_read parameter name into runtime evidence. Production EFC unchanged, only testexe/source/runner MMselector changed. Direct-read coverage still requires a supported fused/repacked quantized path or another validated CPU-global-read primitive.

## Clean-read handoff oracle extends dirty-producer coverage

Cache audit confirms every op destination is marked dirty by proc_op_req regardless of actual CPU/DMA producer, so next source flush can invalidate clean cached output too. No new missing-invalidation defect proven. Extended existing test_dma_dead_reuse with clean_read arm: dead F32 producer, independent F32 MM+residual usingdead, unrelated DMAconcat force-reusingdead storage, RMSconsumer. Intended fused residual directly reads globaldead after source clean/invalidate. Fusion0 would not establish this direct-read path, and rows9 may use different kernel policy; inspect actual execution. Eight cases (rows1/9, aliasfalse/true, clean_readfalse/true),32changing inputs each. Test-only build pending; production DSP untouched.

## Reduced submission size does not rescue unretained repeatability

OpBatch128 plain128 pair bothCLI0/fullHTP19037.92MiB, replay actually off after multiple flushes. Strict firstbits0 NMSE .000569868686/max .227771223 and selected mismatch15 (4602/264), margins .00424957275/.145418167. Frozen real versusa also differs selected15 but plain repeat variation prevents attributing to MTP. Small batches without retained tensors still vary; batch-size control is negative, default3072 retained. Diagnostic25.70/25.51tok/s not optimization result. Next audit clean-read cache state in reuse tests; dirty producer oracle passing does not automatically cover clean valid cachelines before DMA overwrite. Hypothesis only, no new defect established.

## Submission-size control separates retained-lifetime perturbation from batching

Added optional runner OpBatch parameter (default3072 unchanged) for plain128 repeat at128ops per submission, no callback/retained tensors. This changes cache-flush/queue/fusion boundaries and invalidates descriptor replay after multiple flushes (source flush_batch replay_flushes>1), despite requested replay1. Prior unretained replayoff128 already varied, so replay removal alone is not new explanation. Control can establish whether smaller submission boundaries stabilize the selected policy but cannot identify a kernel/cache cause from one pass. Installed EFC DSP unchanged; compare complete128 rows and audit placement. No result yet at this entry.

## All40 same-selection third run establishes intermittency

All40-dense16 c CLI0; frozena/c760tensors and16 full-logit/selected rows strict bit-identical. Priorb differs froma beginninglayer1 width9 prefill. All40-a also equals single-attn-checkpoint-a full16logits. Therefore exact pairs with added retention do not demonstrate causal suppression; same checkpoint selection can also produce exact repeats. Allocation/execution sensitivity remains a hypothesis supported by changing observed onset, but first-failing-layer attribution requires reproducibility and observation control. No production kernel edits or acceptance.

## Minimal attention residual retention passes, without decode op-count change

Layer1-attn16 a/b bothCLI0,779records exact and16 full-logit rows/selected tokens exact. Captured decode1384ops equals all40-only failedpair; full layer1 detail used1398. Tensor descriptor count differs2451->2473 and memory changes, so equal op count does not establish execution/allocation equivalence. Cannot locate defect downstream/upstream with a suppressed failure or infer causal stability from one pair. Repeat all40-only third process against frozena before designing narrower instrumentation. No production edits/performance acceptance.

## Full layer1 intermediate retention suppresses current early failure

Layer1-detail16 a/b bothCLI0;1767tensor records and all16 sampled logit rows bit-identical, strict gates0. Previous all40 endpoints pair varied startinglayer1. Additional named intermediates alter lifetimes and possible fusion, so this is suppression rather than proof of kernel correctness or production fix. Next single attn_residual1 checkpoint attempts attention/FFN discrimination while checking failure survival. EFC production DSP unchanged, no speed claim.

## Current all-layer probe varies from recurrent layer1

All40 retained endpoints plain16 pair bothCLI0, strict logit/tensor gates1;760records97equal/663different. All layer0 and warmup records exact; layers1..39 each differ on both prefills and15decode records. First9-row prefill l_out1 record81 rows0..5 exact,6..8 differ, NMSE .0000664059222/max .00434350595. Full logits differ fromprompt NMSE .0130977413/max1.21752167, all16 selected tokens equal. Unlike historical oldDSP/unfused/3token all-layer suppression, current EFC/fused/replay/16token configuration preserves early prefill failure. Layer1 is recurrent (full attention every fourth layer); inspect named intermediates next without assuming same onset survives additional retention. No unperturbed-production attribution, no kernel fix or speed acceptance.

## Dense early-layer retention does not remove repeat failure

Retain0..19 plus29/39 plain16 a/b: bothCLI0, strict logits exit1 firstbits0 NMSE .00032017979/max .204569817, first selected mismatch15 (264/4602).418tensor records384equal/34different, all0..19 exact and only29/39 differ in each prefill/decode. First9-row prefill at29 differs rows6..8; at39 differs rows2..8. Dense retention moved onset later again, reinforcing allocation sensitivity rather than monotonic sparse bisection. Final sampled-token mismatch has no subsequent decode, so paired tensors still share inputs. No performance or correctness acceptance; no DSP arithmetic edits. Added optional strict tensor identity gate, self114records exit0 and differing114records exit1 verified; default remains diagnostic.

## Extra checkpoint shifts first observed prefill difference before layer9

Boundary14-16 pair CLI0, all16 tokens identical, strict logits exit1 fromprompt NMSE .0113623985/max1.09743237.114records29equal/85different. Layer0 exact; layer9/14/19/29/39 each17 differing records (9-row prefill,4-row prefill,15decode). At layer9 first9-row prefill rows0..5 exact and6..8 differ, max .00232831389. Previous boundary9 layout had every layer9 record exact. Thus adding checkpoint14 demonstrably changes observed onset; cannot treat sparse outputs as monotonic production bisection. Next dense0..19 plus29/39 capture seeks first observed layer within one retained layout; still not untouched-production proof. No production fix inferred.

## Retaining layer9 exposes same-prefix prefill variation after that boundary

Plain boundary9-16 a/b both CLI0,16 selected tokens match but strict full-logit repeat fails from first row (NMSE .00255326411/max .499688864).95 intermediate records:44equal/51different. All layer0/9 records exact; layer19/29/39 each differ on15 single-token decode records and both9-row/4-row prefill records. First differing9-row prefill rows0..5 exact,6..8 differ. This is a useful distinction from previous sparse layouts: numerical variation already occurs during prefill, before any generated-token divergence. Captures are exported after graph synchronization but retained-output lifetimes perturb allocation. Interval10..19 is therefore probe-specific, not a production first-failing-layer claim. Next pair adds14 without removing prior retained checkpoints. Production EFC DSP unchanged; no fix inferred.

## Sparse retention shifts observed onset rather than stable bisection

Retaining0/19/29/39 plain16 pair: CLI0, firstbits3 NMSE .000873213309/max .431005478, all16 selected match.76records37equal/39different; new aggregate analyzer verifies layer19/29/39 each13 differences, layer0 and every remaining record exact. Thus first variation in this layout occurs before19, unlike previous layout's first after19. Do not claim monotonic production bisection. Added layer9 pair next retains prior boundaries and tests current observed interval; layouts/probe effects explicitly part of evidence.

## Sparse retained boundaries preserve full-logit variation

Plain16 with only l_out0/19/39 retained: bothCLI0, complete16 selected tokens allmatch but logits firstbits1 NMSE .00014866704/max .248159409.57tensor records28equal/29different; all layer0 and all captured warmup/prefill boundaries exact. Firstdecode layer19 exact/layer39 differs NMSE .000357052164/max .071290493. Subsequentdecode layer19 and39 differ. This locates first observed decode variation after19 in this probe-specific graph; it is not proof that production variation starts there or that prefill recurrent state is equal. Add layer29 capture next to split interval while checking drift remains.

## Numerical instability persists with one HVXworker

Both128 one-worker CLI0, fullHTP19037.92MiB placement retained. Strict logits firstbits33, NMSE .0011920827/max1.09820342, all128 selected tokens match. Multiple worker concurrency is not necessary for variation; no correctness/performance promotion. Diagnostic7.71/7.66tok/s. Comparator firstbit-row diagnostics added without gate changes. Next sparse retained layer0/19/39 pair keeps fullgraph fusions but changes lifetimes, so interpret probe-specific variation/suppression, not untouched-production first-divergence proof.

## Global MM1 is not a clean NPU kernel-family discriminator

Plain16 MM1/FA1/GDN1 triple passes two full-logit bitidentity comparisons, but model placement moves16560MiB toCPU_REPACK plus800MiBCPU, leaving1677.92MiBHTP.42/42 reported layers offloaded does not establish all matmuls onNPU. This repeat pass cannot implicate HMX arithmetic or qualify optimization; no promotion. RestoreMM2 and test one HVXworker with placement audit, existing optional selector only.

## Warmup is not required for plain variation

No-warmup128 a/b bothCLI0, strict firstbits3/selected112 (0/65915), NMSE .00167233585/max .611432552. Frozen real128 versus a likewise differs firstbits3/selected112. Initialprompt/firsttwo decode rows exact but later same-prefix floats vary. No-warmup is not a fix and default unchanged. Next optional MM1 controls remove remaining HMX prefill matmuls while FA/GDN alreadyHVX; this tests a policy discriminator, not production acceptance.

## Attention output reuse passes focused oracle

Extended prior batch oracle with aliasfalse/true: unrelated F32 matmul4096x2 executes before batch2 attention and batch destination reuses its32768-byte output; independent single1+1 remains reference. Initial build created view before backing tensor and aborted during allocation, not kernel execution; corrected order build84108. FA2/FA1 x actual replay1/0, visible7/13/32/45 x bothaliases x32 inputs allpass1024 graphs. Runtime proves exact shared address and replay capture6ops/16tensors in alias arms. This covers a cache handoff missing from original disjoint test, not arbitrary model aliases or proof no cache defect remains.

Zero-fill did not stabilize fullmodel. Next no-warmup plain128 pair tests dependence on prior empty warmup execution; optional runner switch defaultsfalse. No production arithmetic/policy change, EFC DSP retained.

## Zero initialization does not rescue all-HVX plain repeatability

Fill0 plain128 a/b complete, differ firstbits0 and selected104 (47/9764), row NMSE .00582894217/max1.2520566. First rawargmax95 is not selected mismatch (sampling rules can exclude rawtop). Fill0-a versus stable real128 firstbits5/selected104, NMSE .0054341586/max .937845945. Initial allocation zeroing is not sufficient; no production fix inferred. Extended attention oracle to force batch output onto unrelated dead F32 matmul output, retaining independent single-row comparisons, to test dirty-destination alias handoff. Build/test pending.

## Unprofiled attention oracle verifies actual replay; model instability persists replayoff

Profile0 FA2/FA1 x replay1/0 each4cases x32 allpass:512 additional graph comparisons,1024 total including earlier profiled arms. Actual capture uid1..4 (5ops,13tensors) and replay hits logged for both policies. Thus pure model-layout batch2/single1+1 arithmetic with these seeded masks/inputs is bit-identical within CPU and within DSP under real replay too. This does not cover model allocation aliases, recurrent state or stale KV tails.

Plain replayoff128 repeat varies fromprompt and selected104 (9764/47), proving replay not required for instability. Real replayoff128 completes and differs from that repeated plain at104. New allocation-fill0 paired repeats isolate RPC initial-memory sensitivity; no production fill policy adopted. Test-only source/exe updates, DSP EFC remains unchanged.

## Attention batch/single oracle passes HMX and HVX

Coverage correction from source: want_replay requires opt_profile==0, so profile1 disables capture/replay even with requested replay1. The512 passes establish profiled batch arithmetic only, NOT actual replay equivalence. Profile0 arms and runtime replay-hit proof still required. Replayoff plain128 repetition also fails initial bits and selected104 (9764/47), so model instability does not require replay. Earlier replayoffplain/real exact128 match is one pair, not a stable fix.

New focused test in existing backend harness uses Q F32[256,2,16] and K/V F16[256,256,2] with model-like permuted strides, F16 causal masks visible7/13/32/45,32 seeded changing inputs per case. Wholegraph outputs concatenate batch2 and single1+1 results; custom metric requires bitidentity within each backend separately, not cross-backend float equality. FA2/FA1 x replay1/0 all4cases pass:512 graph comparisons. Runtime profile proves actual hmx-pipe versus hvx selection for both widths. No isolated batch arithmetic failure reproduced; do not promote FA1 as a proven HMX bug fix. Profiling could perturb timing, and full-model state/lifetime behavior remains outside oracle scope.

All-HVX128 replayoff plain is strictly bit-identical to real MTP128 replay1 for all128 rows, despite differing from earlier plain replay1 atbits7/selected15. Unfused plain differs at prompt/selected15. These are dependencies/equivalent-output observations, not established causality amid plain repeat instability. Repeat replayoff and real replayoff controls live84985. EFC DSP unchanged; test-only build81963 complete and copied only testexe.

## All-HVX repeatability is not established at128

Repeated plain128 CLI0 emits complete128; strict logits fail from initial prompt NMSE .000402363347/max .206838369, while selected tokens all match. Repeated plain64 CLI0 passes all64 full-vocabulary bitidentity. Combined job5660 exit0 reflects only final passing gate, not both. Thus same-length numerical variation exists and token-length causality remains unproven. Reproduced64 pair is stronger evidence than a single pair but cannot qualify128/MTP globally. Replayoff and unfused128 controls next; no tolerance relaxation or production promotion.

## Longer all-HVX MTP gate fails because plain prefix changes

Allhvx128 plain/real complete128 each, strict fail first bits7/selected15 (4602/264), NMSE .00167076229/max .538464785. Real52/74 drafts accepted. Real64 and real128 first64 full rows remain bit-identical; plain64 versus plain128 differs first bits7/selected15. The64 matched pair is valid evidence but not stable global equivalence. Do not attribute this to token length until same-length repetitions distinguish nondeterminism; do not call HMX attention a proven root cause from the64 selector experiment alone. New plain128/64 repeated controls pending. All retained policies experimental, EFC unchanged.

## Real MTP64 bit-identical under consistent HVX GDN and attention

Handle72824 terminal0, both CLI exit0. GDN1/FA1/routerHVX with expanded MTP GGUF/EFC DSP: plain and real MTP complete64 target rows, strict full-vocabulary bitidentity passes; all selected tokens equal. Real cap1 accepts27/36 drafts including9 rejections, verification/history/rollback active, allfusion/replay1, no KV-only or fused verifier. Compared with FA2 control (first bits32/selected61), changing attention policy removes the remaining observed target discrepancy for this prompt. This establishes a passing64 same-policy control, not global MTP acceptance or a located HMX implementation defect.

Traced no-profile rates26.75 plain/23.72 MTP do not show speed benefit. FA1 plain differs from FA2 plain already at prompt and selected61. Therefore FA1 cannot be promoted as output-preserving optimization relative to existing FA2 target. Need128/multiprompt tests, CPU reference checks, then focused HMX batch1/2 attention oracle and faster output-preserving implementation.

## Consistent-HVX GDN removes initial MTP discrepancy, not all decode drift

Handle13443 terminal1 only from final strict selected-token gate; all CLI arms exit0. Zero-cap versus real-cap one-token controls with GDN1 pass bit-identical full logits. Real default-GDN versus real GDN1 also bit-identical, so changing plain K1 policy removes observed prompt discrepancy while real K2 arithmetic stays unchanged. Runtime profile confirms hvx-recurrent-scalar for nine-token K1 and K2 targets; history GDN+CPY fusion executes. Do not attribute this to a missing history fusion.

Full64 plain/real GDN1/routerHVX target comparison first differs in bits at32 and selected token at61 (1061/353), NMSE .00615652845/max .902138725; reference pair margin .0192584991, candidate reversed .0696849823. Both complete64; real27/36 drafts accepted. This is improved isolation, NOT accepted MTP. Diagnostic rates26.25/23.70 are not untraced comparable benchmarks. Next FA1 control keeps real draft verification/history and tests attention arithmetic across batch1/2. EFC baseline unchanged.

## F32 quantizer full-model rejection and rollback

q8-f32-raw128 fails CPU selected-token gate at step19 (13/303), earlier than EFC baseline step39. Row NMSE .000701287132/max .353906333, reference margin .0568733215, reversed candidate .0376052856. Diagnostic46.09ms/token; prose also still fails. Scalar quantizer corrects isolated identity case but is NOT an accepted full-model optimization. Retained F076 signed artifacts. build-q8-reference.cmd OFF build92127 succeeds and restores exact EFC17600... hash, OFF v81 cache, signed catalog membership verified.

## MTP prefill discriminator: GDN snapshot count selects different arithmetic

Zero-cap64 control aborts when runtime still drafts and asserts output reservation. It is NOT a valid no-draft decode. Its single captured target prompt row matches plain64 bitwise. Runner now guards zero cap to one-token prefill only. Successful zero/real cap one-token runs both emit9419, but strict logits differ NMSE .00173994209/max .809248924, reproducing original prompt gap without any verify round.

delta-net-base.cpp builds K1 GDN without history and K=n_rs_seq+1 with history. Both host and DSP permit HMX chunked GDN only with K1, scalar gate, dimension multiple64, n_tokens>=8. Real cap1 MTP requests n_rs_seq1/K2 and therefore HVX recurrent; plain/zero-cap nine-token system prefill can use HMX. Four-token following prefill may inherit differing states. This is a specific kernel-policy discriminator; buffer reservation/layout still changes, so runtime controls needed before claiming root cause. Consistent-GDN-HVX prompt and real64 tests running handle13443, no production policy change yet.

## Quantization reference does not explain prose drift alone

q8-f32-prose32 still fails CPU-selected gate step14 (2014/15771). Same-prefix NMSE .00710593797/max1.15200472; CPU pair margin .324108124, candidate reverse .130945206 (baseline reverse .19323349). Thus focused identity mismatch is corrected, but Q8 quantization precision alone does not rescue this full-model regression. Error is not uniformly reduced over prefix. Diagnostic trace reports46.31ms/token versus baseline traced35.69ms; no untraced paired speed result and no promotion. Longer raw Hello comparison still pending in handle13340. Do not generalize synthetic pass to end-to-end correctness.

## F32 activation-quantization reference passes previously failing identity cases

Added default-OFF nested DSP CMake option GGML_HEXAGON_Q8_F32_REFERENCE and build-q8-reference.cmd ON/OFF. Enabled v81 branch computes F32 per32-value max/d/inverse/roundf, stores FP16 scale, then reuses tiled replication/packing and integer-dot kernels. Q8_1 unchanged; no host ABI or model weights changed. This scalar path is for causal/numerical reference, not a final fast implementation.

Build55057 succeeds/signs/installs v81 F07662BA46E85191506D2B6D6B14B361BE3100999B62B03D310AAF094037B8EF; ON cache and signed catalog membership verified. Host DLLs unchanged. activation-f32-reference passes512 comparisons across both replay modes, including random Q4_0 identity cases that failed baseline5.45e-6 NMSE. F32/exact controls continue passing. Packing and dot arithmetic retained, so this isolates focused identity discrepancy to activation quantization calculation. No model-level attribution or speed claim yet.

Regression oracles q8-f32-mm-bias768, q8-f32-dma256, q8-f32-rms512 all pass. Total2048 passing focused comparisons including activation512. Signed artifact backup q8-f32-reference-candidate; EFC rollback alignment-preflush-candidate. Default source optionOFF, but configured/installed v81 is diagnosticON until explicit reset/rebuild or artifact rollback; never confuse these states.

## Four-prompt corrected baseline: three passes, one failure

Complete frozen32-token corpus on EFC baseline/original GGUF: math/code/password match all CPU selected tokens; prose differs step14 (2014/15771), CPU margin .324108124, NPU reversed .19323349, same-prefix NMSE .00676040833/max1.13722801. Suite stops at prose; password separately executed with same labels/settings and strict pass. This is NOT a passing full suite. References remain frozen for diagnostic quantizer comparisons. No baseline performance claim from traced timings.

## Broader prompt gates underway; masked-row MTP hypothesis rejected

Frozen decode-correctness-v1 math/code each pass32 complete CPU-selected tokens on corrected full-fusion NPU. Logits differ, so this is selected-token agreement, not CPU bitidentity. Prose/password still pending in live job54867. Original GGUF hash A4D2D76A1E12E4E945F610C649B8BEC72BFD0C6956299611C51F6266414B3827. Math CPU traced decode1.266sec/31 calls shows trace-enabled CPU can run normally; earlier CPU slowdown cannot be attributed to dumping solely from correlation.

Audited possible final-layer row-selection explanation for MTP prompt mismatch: qwen35moe crops before final FFN when embeddings_nextn_masked=true and after output norm when false. Context constructor initializesfalse for plain; MTP constructor explicitly sets targetfalse and drafttrue. Therefore that target flag does not distinguish plain versus MTP here. No graph change applied. Remaining candidates include recurrent-history mode (n_rs_seq), kernel/state policy, and output lifetimes; first-divergent runtime evidence still required. Do not hide mismatch by discarding MTP hidden rows.

## Identity-weight oracle separates precision-sensitive activation cases

New test_mm_activation_precision in existing test-backend-ops harness uses exact identity weights F32 or Q4_0, k128/256, single token,32 changing inputs, stock1e-7 NMSE. Weight Q4_0 identity values0/1 are exactly representable. With fusion0, activation-identity-replay1-replay-1 and activation-identity-replay0-replay-0: F32 controls pass, random Q4_0 cases fail first NMSE .000005454/.000005483 identically in both replay arms.

Extended exact controls force each32-value activation block to maximum127/64 with integer/64 entries. Scale1/64, inverse64, maxima and input values are exactly representable in both arithmetic paths. Build22839 exits0 and copies only testexe; installed DSP hash remains EFC17600... . activation-exact-replay1-replay-1 and activation-exact-replay0-replay-0 each pass6/8 cases: all4 F32 cases and both exact Q4_0 controls pass32 inputs each (384 total passing comparisons across both modes); both random Q4_0 cases fail same first NMSE. No tolerance relaxation. This isolates precision sensitivity in a one-op graph; it does not prove full-model step39 difference originates here or exclude every other error. Next useful experiment is CPU-compatible F32 activation quantization with identical tiled layout, opt-in and oracle-gated before model/performance testing.

## Corrected-kernel MTP still differs before verification

alignment-chat-plain64 and alignment-chat-mtp64 complete64 target logits/selected rows, same lossless F32-MTP model/routerHVX/all fusions/replay1. Gate fails step15:4602/264, reference pair margin .127437592, candidate reversed .0328998566. Prompt row already differs NMSE .00173994209/max .809248924, identical to earlier observed prompt delta. Cannot attribute that initial difference to acceptance rollback. MTP27/36 drafts accepted, still NOT correctness accepted.

Stage diagnostic: draft decode230.793ms/36 calls (median6.123ms), two-token catchup84.491ms/36 (median2.241ms); target counter deltas2167.368ms/36 include a nearzero initial sample and delayed-counter ownership, so do not treat these as exact per-verify timings. Traced/timed CLI reports26.34tok/s plain versus23.02 MTP, not frozen untraced speed evidence. KV-only and fused verification remainoff. Need first-divergent prefill tensor/graph-policy audit and one-/two-token target arithmetic equivalence.

## Replay also preserves128 rows

alignment-replayoff128 versus normal alignment-npu128 passes strict128-row full logits and selected-token identity. Together with unfused128 identity, neither fusion nor replay is required for remaining CPU token40 gap on this prompt. These controls preserve output rather than relaxing CPU gate.

## Full-fusion correction preserves128-row unfused result; CPU margin flips

alignment-cpu128 and alignment-npu128 complete128 rows with complete selected logs, original GGUF/raw Hello. Strict CPU selected-token gate fails zero-based step39: CPU279 versus NPU1608. CPU selected-pair scores22.2513428/22.2421513, margin .00919151306; NPU22.2472/22.2552643, reversed margin .00806427002. Same-prefix logits NMSE .00108732384/max .358676672. Do not compare later rows as shared-prefix numerical results. Narrow margin does not prove correctness or an expected rounding cause; longer CPU equivalence is NOT achieved.

alignment-unfused128 versus normal alignment-npu128 passes strict128-row full-vocabulary bitidentity and all selected tokens. Unfused also fails CPU at exactly step39 with identical pair/error. Thus this observed remaining CPU/NPU gap does not require fusion. The alignment correction preserves the unfused NPU trajectory over128 tokens while retaining all fusions. Replay-off long control is still pending. CPU reference traced eval431.441seconds; earlier16-row CPU traced eval39.682seconds but sampling time31.59ms, so tracing overhead alone must not be assumed to explain CPU evaluation slowdown.

Read-only arithmetic audit supplies possible differences, not attribution: DSP quantize_block_f32_q8_0_tiled converts F32 activation/maxima to F16, computes F16 scales/inverses and rounds quantized integers; CPU Q8_0 reference uses F32 maxima/division/inverse before rounding and FP16 storage of scale. Both group32 values (tiled128 groups four blocks), so QK_Q8_0_TILED128 is not itself a128-value quantization scale. CPU Q6_K also uses Q8_K activation whereas DSP tiled path uses its Q8_0 layout. Need targeted numerical evidence before altering policy or treating CPU divergence as harmless.

## Fused matvec residual alignment reproduced and corrected

The new existing-harness mm_bias test compares whole graphs to CPU with32 changing inputs: F32 weights k32/2048, output n32/64/256, disjoint or residual-alias destination. Fusion0 passes768 comparisons across replay1/0. With preflush-only DSP and fusion8, n32/64 fail in both destination arms while n256 passes. For k32 failures NMSE .288578213/.149568974; k2048 .002702658/.002193338. Replay1 runner stops on failure, so prepatch fused replay0 was not measured.

hvx_add_f32_uaa requires128-byte-aligned src0/src1. The small F32 matvec splits residuals into thread slices that can start16/32 bytes apart; the caller supplied those unaligned residual pointers. Replacing the two matvec callsites with existing alignment-aware hvx_add_f32 dispatcher retains aligned fast paths. New installed/signed v81 EFC17600F16F21B612DF2A43EC53B2F46967F95492642FA7D9A963C4682FCD46 passes all768 fused comparisons with replay1/0. Logs confirm actual MUL_MAT+ADD fusion. DMA-dead-reuse256 and RMS-reuse512 regression comparisons pass.

Normal original-GGUF raw Hello, all fusions/replay enabled, raw-alignment-after-a/b:16 full-vocabulary logits rows bit-identical, selected tokens complete/equal. Candidate passes CPU16-token gate and is bit-identical to the previously passing unfused NPU16 rows. Thus this patch resolves the observed fusion-driven step1 divergence while preserving fusion, not merely suppressing it. CPU/NPU float differences remain; longer/multiprompt/MTP correctness not established. Trace timing is not a benchmark. Signed artifacts preserved in alignment-preflush-candidate; rollback artifacts retained.

## MUL_MAT+ADD alone reproduces normal failure

mask8 completes and CPU selected gate fails step1 (271/353), identical first-decode delta to all1. mask50 completes and strict16-row logits equal unfused. Handle22967 terminal. Paired matmul fusions preserve this test; residual-add matmul alone sufficient to reproduce. Profile mask8 shows70 MUL_MAT+ADD across F32 ssm_alpha+bias and quantized output projections. No root-cause claim yet.

Specific alignment lead: hvx_mv_2d residual-add calls hvx_add_f32_uaa with source2 at vtcm_src2+src0_start_row. A32-output F32 projection may split into4float slices across8threads, giving16byte offset, incompatible with aligned source2 if suffix contract matches existing helpers. Repacked matvec has32-row slices and is aligned. Need inspect generated helper definition and parameters, construct F32 width32+bias chain oracle with changing inputs, then repair/test safely. Also direct-store generic MM paths add residual after writing output and may need alias coverage, separate hypothesis. Installed/source preflush candidate unchanged, no live inference or promotion.

## RMS-multiply-only passes identity

mask58 completes exit0, CPU selected gate exits1 at step1 (271/353), same first-decode delta as all1. Matmul/allreduce group sufficient to reproduce. Handle77198 terminal, split8 versus50 live22967. Source and signed DSP unchanged.

raw-preflush-mask4 completes exit0;16 CPU selected tokens match and all16 full NPU logit rows exactly equal unfused. Profile2114ops/39550us,121 RMS_NORM+MUL/697us; kernel executes rather than flag-only pass. GDN-copy and complementary group also individually preserve unfused. Remaining mask58 live77198. Next MMadd8 versus pairedmatmul/allreduce50 if reproduced. No binary changes, source preflush candidate, no accepted performance claim.

## GDN-copy not sufficient for observed failure

mask64 completes exit0, passes16 CPU selected-token gate and16-row bitidentity to unfused. Profile2205ops/39996us, GDN+CPY30/1814us. mask62 completes and selected-token gate fails at step1 (271/353) with same .0150917313NMSE/max1.61351246 as all1. GDN-copy alone preserves this test, remaining group reproduces failure. Handle22248 terminal.

Next RMSmul-only mask4 versus matmul/allreduce mask58 launched sequential16-token captures withPROFILE1/replay1/routerdefault. No runtime/source change; preflush6D48E2F... remains experimental. Need identify actual failing family rather than disable all fusion. Diagnostic profile timing not benchmark.

## Complementary family mask preserves unfused output

mask2944 completes exit0, passes16 CPU selected tokens and strict bitidentity16 rows to fusion0/replay1. Last batch1692ops/34436us: active SSM_CONV+WIN30(1533us),ADD+NW40(362us),ADD+SG40(285us),MOE+WNORM39(119us),MOE+TOPK1(22us), GDN unfused30(1679us), head2925us. Profiling checks actual invocation rather than treating enabled flags as proof. Diagnostic sums only, not comparative speed claim.

mask3070 completes and strict bitidentity16 rows with unprofiled all1 candidate confirms familymask equivalence and no observed profile-induced changes in that pair. Handle51662 terminal. Narrow split64 versus62 now live22248,16tokens each, peropPROFILE1/replay1/routerdefault on unchanged signed preflush6D48E2F... . Candidate not promoted.

## Matrix closes replay discriminator; family profiles started

mask126 first group completes exit0; CPU token mismatch step1 (271/353), first-decode .0150917313 NMSE/max1.61351246 equals all1. Profile last1924 ops sum37924us, fused counts121 RMS_NORM+MUL,70 MUL_MAT+ADD,50 MUL_MAT_NX,40 MUL_MAT_ID_NX,30 GATED_DELTA_NET+CPY. Head2951us, MMIDNX4536us,totalMM10307us, GDNcpy1790us. These are one diagnostic final batch on differing trajectory, not paired benchmark gains. Next mask64 versus62 isolates GDNcpy from rest; complementary2944/all3070 still running51662.

Final fusion0/replay0 control complete handle13906; strict16-row logit identity with fusion0/replay1. Fusion1 replay0/1 also mutually exact but CPU selected-token mismatch at step1. Fusion-off16 tokens match CPU. Source/binary preflush unchanged across matrix. Identifies fusion-on execution difference in this prompt, not a universal fusion defect.

Handle51662 runs family-mask126/2944/3070 sequential16-token raw captures with peropPROFILE1, replay1/routerdefault. Need prove actual fused opcode counts and compare CPU token gate; mask3070 should reproduce all1, but must verify empirically. Use parse_last_batch/parse_profile for DSP operation times separately from tracing/logging overhead. No speed promotion, no runtime edits this entry.

## Fusion-off/replay-on identity discriminator

fusion1/replay0 completed: same CPU selected-token difference at step1 (271/353), NMSE .0150917313/max1.61351246. Strict comparison to normal fusion1/replay1 passes all16 logits bit-identical. Thus replay is not necessary for this fusion-on trajectory difference. Final fusion0/replay0 matrix control remains in13906; next family-mask bisection.

raw-preflush-isolated completes and passes16-token CPU sequence. raw-preflush-fusion0-replay1 also completes exit0,2235-op replay, CPU token gate passes16. Strict comparator against isolated logits passes16/16 bit-identical despite replay1/routerdefault versus isolated replay0/routerHVX. Supports fusion as remaining discriminator in this prompt, pending fusion1/replay0 result. Handle13906 remains live for that run and fusion0/replay0 control.

Decode runner Fusion option now allows0..8190 backend masks with default1 unchanged. Existing flags126 groups allreduce/RMSmul/MMadd/MMNx/MMIDNx/GDNcpy;2944 groups SSMwindow/MOEadd/sharedgate/routerweightnorm. GDNstate/RMSscale matchers are not called even under all1; mask3070 therefore represents currently invoked families. Need actual invocation proof in profiling if bisection proceeds. Installed/source preflush6D48E2F... unchanged, no promotion.

## Reuse failure is independent of replay

raw-preflush-default-logits completed exit0; CPU selected-token gate fails at step1 (CPU271 versus candidate353). Prompt numerical delta matches old prompt (.000254574 NMSE); first decode NMSE .0150917313/max1.61351246. Thus corrected narrow handoff does not repair the normal fused/replayed model trajectory. Isolated raw candidate still running in31983.

With installed original52A4... DSP and ReplaySettings0, dma-dead-mm-old-replay-0 exits1: alias rows1 NMSE1.993436722, rows9 NMSE1.999657181; disjoint controls pass32 comparisons each. Same failure class as replay1. Runner explicit replay selector defaultboth unchanged. Old binary failures are first-iteration stops, not full128 alias comparisons.

Logs verify alias dead/concat pointers equal and disjoint pointers differ. Widthrows1 producer reports hvx-tiled (output2rows), rows9 hmx-tiled (output18rows). HMX implementation sends output_chunk direct DDR pointer to transfer_output_chunk_threaded. Narrow test not proof all full-model variation originates here. Reinstalled signed preflush6D48E2F... and verified catalog. Raw normal/isolated candidate captures live handle31983; no promotion.

## Focused DMA reuse failure corrected by destination preflush

Revised F32 MUL_MAT dead producer test builds68720 exit0. Old signedDSP, replay1: disjoint rows1 and9 pass32 changing inputs each; forcedalias rows1/9 fail immediately with NMSE1.993436722/1.999957541 (unchanged1e-7). Old runner exits1 and does not run replay0. Thus not256 old comparisons passing. Source graph appends dead producer via existing gf before concat; no unused source dependency that would preflush old output.

Installed preserved signed preflush candidate6D48E2F... so/catalog, signtool verifies. Candidate replay1/0 all4 cases x32 x2 =256 comparisons pass. Host unchanged. Need old replay0 negative control and inspect actual direct-store producer/alias selection. Full-model normal trajectory remains unaccepted. Candidate installed for further diagnostics; earlier old-installed state superseded.

## Raw trajectory discriminator and initial reuse test

Matched raw captures complete handle78041. CPU vs normal oldDSP selected-token difference at step8:1510 versus2614. CPU vs isolated oldDSP (fusion/replay0, routerHVX) passes16-token sequence gate; float logits differ, prompt NMSE .000254574/max .2945 in both modes. Thus separate normal decode gap depends on at least one of those settings under this prompt; not yet specific attribution.

Initial test_dma_dead_reuse uses dead SCALE producer -> unrelated dim1 contiguous CONCAT -> RMS, wholegraph stock1e-7 tolerance,32 changing inputs each, rows1/9, forcedalias/disjoint, replayboth. All256 CPU comparisons pass on oldDSP. Logs prove dead/concat address equality in alias arms and3-op capture/hit; width9 here is3-op captured too. However unary SCALE writes through DMA, not cached DDR stores, so this is not the intended dirty-cache reproducer. Revised producer to F32 MUL_MAT with32-wide dot product; build68720 live, no runtime DSP install. Runner reuse pattern selector added with unchanged RMS default. Need prove kernel/direct store selection and complete oracle. Installed old52A4..., source preflush candidate.

## Matched raw token/logit discriminator launched

Decode runner gains opt-in full logits trace and selected-token detail log, Replay/Fusion options default1, RouterHvx defaultoff. Normal defaults preserved. Handle78041 runs raw-cpu-logits, raw-old-default-logits, raw-old-isolated-logits sequentially,16 tokens/original GGUF/raw Hello/c256. CPU trace latency is much higher than untraced rawCPU and cannot serve as performance evidence. Need complete shared-prefix comparisons before interpreting numerical/trajectory differences. Installed old signedDSP unchanged, source preflush candidate.

Potential minimal dead-buffer test design: compute a cached scale output, then make a contiguous dim1 concat DMA output view_src alias that dead output with unrelated leaf sources, followed by RMS consumer. Existing router test already uses output view_src to force dead-buffer reuse. Must confirm graph order and actual alias addresses, direct DMA selection and no fallback; do not claim a reproducer until old fails and candidate passes at unchanged reference tolerance.

## Raw CPU path does not corroborate either normal NPU transcript

dst-preflush-cpu-raw-reference exits0, handle92024 terminal. Runtime verifies0/42 offloaded and CPU_REPACK model buffers.16-token completion output: Hello followed by I am trying to use the `get` function to get the value. Existing oldDSP transcript begins following code/PHP; preflush begins AWS CDK/VPC. CPU raw output therefore agrees with neither observed normal trajectory. Comparisons so far are text-prefix evidence with different generation lengths, not selected-token gates; next collect matched raw token/logit traces and isolate fusion/replay/router precision. CPU short timing30.74tok/s versus chat reference .21 is unexplained and must not be promoted as comparative benchmark. No model inference live; installed oldDSP, source candidate.

## Independent CPU reference supports isolated candidate trajectory

dst-preflush-cpu-reference16 tokens completes exit0 (handle16121). Verified0/42 GPU-offloaded layers. CPU selected tokens all match candidate-long-a/b; old rollback-a/b select264 instead of CPU/candidate4602 at step15. CPU and NPU logits differ even at prefill (NMSE .0280211797/max1.9750452), so this is selected-token evidence, not float identity or complete backend correctness. CPU diagnostic rate .21tok/s. Candidate repeatability plus rollback variation plus CPU token agreement strengthens correction hypothesis in isolated configuration only.

Comparator --require-tokens requires complete token logs/equal row counts and nonzero exit for differing selected sequence, without rejecting expected float differences. CPU vs candidate exits0, CPU vs old-b exits1. Raw-prompt CPU reference launched handle92024, dst-preflush-cpu-raw-reference16 tokens using original model/completion settings; decode runner supports -Device none/ngl0 with unchanged default. Need reconcile normal transcript difference, broader prompts/configs, MTP, and dedicated DMA reuse test. Candidate not promoted; installed old52A4..., source candidate.

## Old-binary challenge reproduces nonrepeatability

dst-preflush-rollback-long-a/b both exit0,16 rows; strict comparison exits1. Prompt logits bit-identical; first decode NMSE .00139511853/max .705013514. All16 selected tokens match within old pair. Candidate pair16/16 bit-identical, old variation returns after signed-binary rollback with unchanged host. Supports repeatability effect of candidate, not yet actual numerical correctness. Cross-binary selected token differs at step15.

Stage runner now supports explicit -Device none and ngl0, defaultHTP0/ngl99 unchanged. CPU reference running handle16121, label dst-preflush-cpu-reference,16 tokens, same expanded model/sampling/context/batches. Log verifies0/42 offloaded layers and CPU/CPU_REPACK weights. First prompt selected token9419 matches HTP, CPU logits differ. Need complete common-prefix token/logit comparison. CPU rates not comparable to NPU diagnostic rates. Installed oldDSP52A4..., source preflush candidate, no promotion.

## Normal transcript gate fails despite isolated repeatability

Candidate-long-a versus rollback-long-a: prompt logits bit-identical, first decode NMSE .000720737427/max .467846513. Selected prefixes agree through step14, diverge at step15 (candidate4602 versus old264). Thus destination preflush alters the isolated trajectory too, not only the production fused/replayed transcript. This cross-binary comparison does not adjudicate correctness.

dst-preflush-default-regression completes exit0,28.06 tok/s/35.64 ms,128 runs,126 reuses,1381-op replay. Trimmed output differs from topk-alias-control: repetitive AWS CDK/VPC response versus PHP PDO response. Full default output equality fails. Do not accept candidate merely because isolated no-fusion/no-replay pair is repeatable; an independent reference is needed to adjudicate altered outputs. No speed promotion from single sequential regression.

Saved signed candidate so/catalog to dst-preflush-candidate. Restored installed DSP52A4D490.../catalog from pre-cache-syncht with no inference live; signtool verifies. Host EDB9.../llama981E... unchanged; source still contains preflush candidate. Old-DSP16-token paired challenge running handle84426, labels dst-preflush-rollback-long-a/b. Earlier installed-candidate/live32488 entries superseded. Compare complete traces and tokens, then independent-reference and production-config tests.

## Longer unretained repeat replicates identity

dst-preflush-long-a/b complete with16 rows each and complete selected-token logs. All16 full248320-vocabulary rows bit-identical; all selected tokens agree. Two-token decode short pair likewise exact. Settings unchanged: fusion/replay0, routerHVX, expanded GGUF in plain mode, no callback/retention/fill. This repeats the observation over15 decode steps but does not cover production fusion/replay, arbitrary prompts, MTP or long-context correctness.

Comparator adds optional --require-identical with complete token alignment/equal row count checks, nonzero exit on any logit/token difference and total compared-row report. Strict short and long pairs exit0; negative cache-syncht plain pair exits1 with existing numerical drift. git diff --check passes. Host hashes unchanged EDB9... / llama981E... . Default regression remains live handle32488; candidate6D48E2F... not promoted. Next signed binary rollback challenge at matching16-token settings.

## Destination preflush first full-model repeat: bit-identical

dst-preflush-plain-a/b both exit0, handle83377 terminal. Comparator checks three full248320-vocabulary rows, finite values and matching selected-token aligned prefixes: prompt and both decode rows bit-identical, NMSE0/max0. This pair uses no retained tensors/callback/fill; fusion/replay off, targeted routerHVX, separate expanded model in plain mode. Prior unretained repeats varied at decode. This is promising evidence, not proof that dead-buffer DMA handoff caused the defect or that all configurations are correct.

Additional16-token pair and normal128-token regression running sequentially handle32488, labels dst-preflush-long-a/b and dst-preflush-default-regression. Candidate remains installed, not promoted. Need replicated longer comparison, rollback challenge, unrelated reuse reproducer and latency measurements. Source/installed DSP6D48E2F... unchanged this entry.

## Destination preflush candidate built and narrow oracle passed

Conservative experiment flushes tracked dirty destinations after assignment and before mark-dirty/execute in proc_op_req. Intent: clean/invalidate old cached data before storage reuse by a DMA writer. Does not establish that this handoff caused observed drift. Helper arrays have10 slots versus4 outputs. All DSP architecture targets compile; build75714 exits0; installed v81 signed catalog verifies. Installed SHA2566D48E2F15B0EDEAEC3AF07F3F58D2FF3B2B5E90B9C2D72183512527D8E18B930. Pre-cache-syncht signed rollback preserved.

dst-preflush-rms replay1/0 passes512 original-tolerance CPU comparisons. Full-model plain repeats running sequentially handle83377, labels dst-preflush-plain-a/b; three tokens, expanded F32 model, routerHVX, fusion/replay0, logits trace without tensor retention/callback/fill. Need inspect complete results and latency before any promotion. Prior rollback regression trimmed transcript also matched topk-alias-control.

## Cache-completion negative experiment and rollback

Trailing syncht candidate fails to remove plain decode variation: first decode logits NMSE .00118588821 / max .625419617, second .000501184986 / max .362013459. Prompt logits exact, all three selected tokens match. Candidate regression 28.68 tok/s / 34.86 ms, 1381-op replay and exact trimmed baseline transcript. RMS oracle previously passed512 comparisons; that narrow pass did not predict full-model repeatability.

Preserved candidate artifacts in cache-syncht-candidate; reverted source and restored pre-cache-syncht signed DSP/catalog. Signtool verification passed; installed SHA25652A4D490D48524236D300F53C2A8DFDB4DD81886ED04F71BB89C9901C2453F52. Rollback regression completed exit0, 28.34 tok/s / 35.29 ms, 128 runs, 126 graph reuses, 1381-op replay. No paired speed claim. Old candidate-running entries superseded.

Next cache-ownership hypothesis: proc_op_req flushes inputs before assigning outputs, then marks outputs dirty without a preflush. DMA output can potentially reuse unrelated dirty dead-tensor storage. concat-ops.c has direct DMA writes to tensor DDR, confirming this class exists. Existing explicit inplace oracle flushes its reused storage as an input and does not reproduce that handoff. Destination preflush candidate remains unimplemented/unproven.

## Cache-completion candidate installed; numerical model gate pending

Rebuild-gdn-state.cmd handle93529 completes exit0. Installed signed v81so verifies via signtool /pa /c; candidate SHA25644BEB26F78AF070E62B249562C22298304A23F4BA8A4AE8F81FB3969F555D869. Pre-cache-syncht backup preserves old so52A4D490... and its signed catalog/host stack. ABI/op layout unchanged; candidate only appends syncht after cache-line clean/invalidate loops. All four DSP architecture builds succeed, only v81 installed for this device. No general architecture correctness claim.

cache-syncht-rms replay1/0 runs exit0, eight cases per setting, 512 changing-input CPU comparisons total at stock1e-7 NMSE. Width1 capture/hit remains covered; width9 repeated graphs are not replay-capture proof. End-to-end plain repeat and default regression are running sequentially in handle18117, labels cache-syncht-plain-a/b and cache-syncht-default-regression. No callback, retained outputs, or allocation fill in the plain pair. Candidate is installed to test the cache hypothesis, not accepted as a numerical/performance win. Must compare logits and selected tokens and check default transcript/timing; revert candidate if hypothesis fails or safety/performance gate fails. Earlier pending-build statements are historical.

## Attention result retention and cache-completion candidate

`retained-final-attn-{a,b}` retains every l_out endpoint plus attn_pregate-39, plain three-token expanded model/router HVX, replay/fusion off, fill unset. 243/246 records equal. Captured attention results match bitwise; differences only in l_out-39 non-final rows: warmup width2 row0 max .202561557, width9 row1 max2.55989604, width4 row0 max .0307788104. Actual decode boundaries, sampled logits and selected tokens exact. Under this allocation-changing probe the observed prefill output variation is downstream of a repeatable captured attention result; this does not prove all attention/cache inputs correct or locate the untouched first failure. Both runs exit zero.

Cache completion lead: hex-utils.h::hex_l2flush emits dccleaninva per line but no trailing transaction synchronization. Existing htp-fence.h read path does use trailing syncht. Qualcomm V71 PRM section5.11 says syncht completes previous memory transactions and shows dccleaninva followed by syncht to ensure memory completion: https://docs.qualcomm.com/bundle/publicresource/80-N2040-51_REV_AB_Hexagon_V71_ProgrammerS_Reference_Manual.pdf . V73 PRM defines cache clean/invalidate behavior: https://docs.qualcomm.com/doc/80-N2040-53/80-N2040-53_REV_AB_Qualcomm_Hexagon_V73_Programmers_Reference_Manual.pdf . Applying this requirement to the v81 DMA handoff is a hypothesis, not a demonstrated diagnosis; other queue/kernel synchronization may already suffice.

Candidate adds one trailing asm volatile syncht with memory clobber to hex_l2flush. Saved installed host/llama/ggml/completion and signed v81so/catalog in npu_campaign/pre-cache-syncht before editing. Original DSP SHA52A4D490... preserved. Rebuild-gdn-state.cmd running handle93529; v81 linked, packaging still pending at this entry. No inference concurrent. Must verify completed install/signature, run changing-input reuse oracle, repeat plain logits WITHOUT retention/callback, and default benchmark before accepting. No correctness/performance claim yet; rollback artifacts available.

## Final-block intermediate retention - suppression, not first-divergence proof

Previous turn verified final-block pair completion while answering the Linux migration question. `retained-finalblock-{a,b}` (all l_out endpoints plus named layer39 intermediates) compares 498/498 tensor records bitwise equal, including warmup/prefill rows that previously differed. All three sampled logits and selected tokens exact. This demonstrates further probe-induced suppression; it does not identify the unperturbed faulty operation. No per-op callback/cuts, but tensor lifetimes and allocations change.

New `retained-final-qg-{a,b}` narrows added retention to Qcur_full-39 while retaining all l_out endpoints. Three-token plain expanded model, router HVX, fusion/replay off, allocation fill unset, no callback. 245/246 tensor records bitwise equal. Every Qcur_full-39 record matches, as do all actual decode boundaries; only l_out-39 four-token prefill row 0 differs NMSE .0000148632209 / max .035435915; its other three rows are exact. Nine-token prefill matches in this pair, unlike the earlier all-layer-only pair. All sampled logits and selected tokens exact. These individual repeat pairs do not prove Q/gate retention causes suppression: different allocation layouts/addresses and intermittency remain confounders. The remaining row variation occurs downstream of a captured repeatable Q/gate projection under this probe, but K/V, norms, state, attention, and FFN are not yet excluded.

Both model pairs exit zero. No code edits or runtime rebuilds this continuation. Diff checks pass; no inference remains. Handles24587 and86365 are completed and supersede prior live statements. Next selective attention-result retention or allocation-preserving DSP snapshot; any snapshot needs explicit DMA completion/cache-coherence validation and a check that it preserves the original failure. No numerical fix or speed gain claimed.

## Layer-38 gap and deterministic initial RPC storage

`retained-gap38-{a,b}` repeats plain three-token expanded-model router-HVX diagnostics, fusion/replay off, retains l_out-0 through 37 plus 39 but not 38. 231/234 records equal. Every earlier retained boundary exact. l_out-39 nine-token prefill row 1 differs max 2.55989604; final row exact. First decode hidden NMSE .000152028938 / max .0900629759; logits .0000873846167 / max .15297699. Second hidden .000184846003 / max .103885531; logits .0000819209622 / max .149787426. Prompt sampled logits and three selected tokens match. This narrows observed variation to 38-39 UNDER RETENTION; not proof of the unperturbed first defective layer.

Added opt-in GGML_HEXAGON_ALLOC_FILL byte (0-255), validated before RPC allocation, fills whole new block before mapping. Runner -AllocFill defaults -1/unset; benchmark and RMS oracle runners clear the environment variable. Does not add nodes, preserve lifetimes, or insert per-op sync. It changes initial bytes and startup memory traffic, including model buffers/guard pages; therefore not an observation-equivalent probe. No default filling and no persistent model-file changes.

`alloc-zero-{a,b}`, no retained/callback tensors, plain three-token expanded model/router HVX, fusion/replay off, explicit byte 0: runtime markers prove initialization executed, including compute allocations. Prompt logits match bitwise. First decode varies NMSE .0013185266 / max .643049002; second .000446354082 / max .399578571. Selected tokens all match. Initial RPC contents are not a sufficient explanation; this does not rule out later stale/partial writes or cache/DMA ownership. Timing is diagnostic, no speed claim.

Rebuild-host-only.cmd exited zero; installed host SHA256 EDB9FF528857E8D0BB0622343549891FA576BD529485D6A362D2176D0072DC0D; relinked llama.dll 981E8D385E324BB3474E632A68CC0D03F3B65DC035067DEB663042EBD7009BAB. DSP unchanged 52A4D490D48524236D300F53C2A8DFDB4DD81886ED04F71BB89C9901C2453F52; OpenCL stub preserved. Live default process module enumeration verifies actual pkg-wos llama.dll/ggml-hexagon.dll/ggml-opencl.dll paths. `alloc-fill-default-regression` exits zero, 28.93 tok/s /34.57 ms over128 runs,126 reused graphs,1381-op replay, exact trimmed topk-alias-control transcript; zero fill markers. Not a new benchmark pair or numerical fix. Parse and diff checks pass.

Next `retained-finalblock-{a,b}` retains all layer outputs plus every named layer39 intermediate using (^l_out-[0-9]+$|-39$), to locate retained-probe prefill variation. This further perturbs lifetimes and cannot prove production first divergence. Running handle24587; wait/check same handle before interpreting captures.

## All-layer replication, late-gap narrowing, and explicit RMS reuse oracle

`retained-thirtynine-{a,b}` filter ^l_out-([0-9]|[123][0-9])$ retains ALL 40 layers, despite its filename. It repeats the all-layer result: both actual single-token decode layer outputs and all three sampled logit rows are bitwise equal. Differences only at l_out-39 nine-token prefill row 1 (max 2.16698694) and four-token prefill row 0 (max .035435915); 238/240 records equal. No final-block attribution follows from a probe that suppresses variation.

`retained-thirtyseven-{a,b}` retains layers 0-36 and 39, not 37/38. Every earlier retained boundary is bitwise equal; l_out-39 differs in nine/four-token prefill and both decode steps. 224/228 records equal. First decode hidden NMSE .000133517906 / max .0625141859, logits .0000685480255 / max .140069485. Second hidden .000184786644 / max .103885531, logits .0000800257585 / max .145386696. Prompt sampled logits and all three selected tokens match. Observed variation is in layers 37-39 under this changed allocation; not production-equivalent first-divergence proof.

Added test_rms_reuse_chain inside existing test-backend-ops.cpp: input scale then three RMS_NORM -> MUL with row-vector scale -> ADD with unchanged residual stages. Explicit inplace arm forces all stages to share work storage; disjoint control uses separate tensors. Shapes F32 [2048,1/9,1,1] and [256,16,1/9,1], 32 deterministic input/scale changes each, complete graphs compared to CPU with unchanged stock 1e-7 NMSE. Reproducer build-router-test.cmd then run-rms-reuse-oracle.ps1 -Prefix rms-reuse-1; fusion disabled, replay=1/0. All eight cases per arm pass, 512 graph comparisons total. Width-one graphs capture 10 ops and replay; token-width-nine graphs do not capture, so only the former prove descriptor replay. Inplace init/add-tensor logs prove reuse_output and initial node_1 have identical data/offset (e.g. 2048x1 offset 28672); only three distinct tensor descriptors versus 12 disjoint. This does NOT simulate gallocr's general dead-buffer reuse, attention caches, routed experts, or concurrent DMA lifetime. It is a useful negative reproducer, not a numerical fix.

Build/copy affects test executable only; installed runtime binaries unchanged. Four model runs and two oracle processes exit zero. PowerShell parse and both worktree diff checks pass; no inference/test remains. Next shrink unprotected gap to layer 38, then late-block/state snapshot or stronger realistic reuse graph. Rates from retained probes are not optimization measurements.

## Late range and attention selector discrimination

Prior turn was progress: retained through layer 29 while preserving decode variation. New `retained-thirtyfive-{a,b}` repeats the same three-token configuration, protecting l_out-0 through l_out-34 and l_out-39. 213/216 records bit-equal; every captured earlier boundary exact. Final output differs only in four-token prefill row 0 (max .035435915) and both decode steps. First decode hidden NMSE .0000586754487 / max .0625141859; logits .0000339259209 / max .107252836. Second decode hidden .000626243285 / max .143603563; logits .000240532719 / max .261002064. Prompt sampled logits and three selected tokens match. The first observed difference is somewhere in layers 35-39 under this probe; retention can suppress earlier bugs, so this is not untouched-production first-divergence proof.

Added explicit -FaSelect and -GdnSelect controls (1 HVX, 2 automatic HMX/HVX) to the diagnostic runner, default 2. Default decode runner also resets both to 2 to prevent inherited settings. No runtime changes. Host eligibility code confirms FaSelect=1 prevents HMX attention, without inserting graph nodes or retained lifetimes. `attention-hvx-{a,b}` runs plain three-token decode, router HVX, replay/fusion off, no callback/retained tensors, DeviceProfile=1. Attention profile rows verify HVX in both runs. Prompt logits repeat bitwise, but first decode differs NMSE .000578341347 / max .502464771; second .000306903768 / max .281970978. Selected tokens all match. Attention HMX is therefore not necessary for variation in this configuration. This does not prove HVX attention correct or exclude all attention/cache defects.

Final decode device op sums 40.675/40.531 ms, 2235 ops each; 10 attention ops .414/.408 ms; 30 GDN ops 1.580/1.585 ms; 30 CONCATS 3.030/3.017 ms; 120 routed matmuls 7.957/7.981 ms; output head 2.929 ms each. Six batches per run. Verbose per-op profiling inflates wall time dramatically, so CLI 3.21/3.03 tok/s is not a benchmark or candidate regression. No speed promotion. Four runs exit zero; runner parse and diff checks pass; no inference remains. Next retain through layer 38 to discriminate final-layer versus recurrent preceding blocks, then construct a realistic alias/reuse correctness oracle. Runtime hashes unchanged.

## Narrow retained ranges - variation survives through layer 29

Continuation of the whole-batch retained-output experiment, no runtime edits. Each pair uses three-token plain decode, the exact-expanded MTP model, router HVX, fusion=0/replay=0, retained outputs plus sampler logits, no evaluation callback. Filters retain an initial layer range AND l_out-39. All selected tokens match; sampled prompt logits are bitwise equal in every pair.

- `retained-early-{a,b}`, layers 0-9+39: 62/66 tensor records equal. All early boundaries exact; final output differs in nine/four-token prefill and both decode steps. First decode hidden NMSE .0013283174 / max .141670227; logits .000379277216 / max .444935381.
- `retained-half-{a,b}`, layers 0-19+39: 123/126 records equal. All early boundaries exact; final output differs in nine-token prefill row 1 and both decode steps. First decode hidden NMSE .000168060239 / max .0429149866; logits .0000666642414 / max .175051689.
- `retained-thirty-{a,b}`, layers 0-29+39: 183/186 records equal. All early boundaries exact; final output differs in four-token prefill row 0 (max .0307788104) and both decode steps. First decode hidden NMSE .000247829958 / max .0621948242; logits .00012613972 / max .185050964. Second decode hidden .000494802507 / max .145060658; logits .000220710694 / max .257750511.

Thus layers 30-39 contain the first OBSERVED repeat variation under the 0-29-retained configuration. This does not locate the first production defect: the probe changes lifetimes/inplace reuse, and repeat equality is not a CPU correctness oracle. Earlier defects can be suppressed or deterministic. Rates are diagnostics only. Next test can retain through layer 34, then isolate late-layer kernels with unchanged graph allocations.

Audited full `htp-tensor.c`, `hex-utils.h::hex_l2flush`, and op preparation ordering. `make_tensor_clean` leaves interior subranges conservatively dirty; no undertracking established. Both serial/worker flush helpers align cache lines. Sources flush before outputs are marked dirty and executed; flushing stale destination aliases before DMA overwrite is a possible experiment, not a proven omission. Do not change installed cache handling without an isolating correctness test and regression gates. Existing runtime hashes remain authoritative; all six runs exited zero, both worktree diff checks pass, no inference process remains.

## Retained-output continuation - whole-batch probe also changes repeatability

Previous turn was progress: demonstrated hidden-state variation before the head and probe-induced suppression at layer boundaries. Added default-off LLAMA_RETAIN_TRACE_FILE / LLAMA_RETAIN_TRACE_FILTER instrumentation. Matching graph tensors get GGML_TENSOR_FLAG_OUTPUT during construction, so the allocator keeps their storage alive and does not reuse them as inplace parents. After graph_compute returns, the context synchronizes the completed graph once and reads those retained outputs. No evaluation callback, per-layer sync, or extra copy op is inserted. Serialization uses the existing 88-byte tensor-trace format and reader. The retained flag nevertheless changes allocation/lifetimes, so this is NOT an allocation-equivalent production probe.

`retained-layer-{a,b}`, three-token plain, all l_out layer boundaries retained, fusion=0/replay=0/router HVX: every single-token decode layer output and all sampled logits now match BITWISE. Differences remain only at l_out-39 in warmup (first row of two), nine-token prefill (row 1 only, max abs 1.94534349), and four-token prefill (row 0 only, max abs .0307788104). 237/240 logical records match. Thus decode variation can be suppressed by output lifetime/allocation changes without per-layer scheduling cuts. This strengthens allocation/reuse sensitivity as a lead, but does not identify the faulty operation or prove a fix. Unstable unused prefill rows remain relevant to MTP hidden streaming.

`retained-layer-profile`, separate device diagnostic, proves six complete HTP batches spanning layers 0-39, one per microbatch; actual decode has 2235 ops and ~64 ms DSP (fusion disabled). No capacity-flush line. Scheduler reserve has one split, 55.25 MiB HTP compute buffer versus the earlier non-retained reservation around 21 MiB. Probe rates are not optimization claims. Next discriminator retains only l_out-39, to separate broad layer-input lifetime protection from a local final-layer allocation effect.

Rebuilt llama.dll SHA256 `342ED94B44FBE60546D7B5D2B1AA14EF8D7ABC3BEC956048AF7874FB5D0C3956`; Hexagon host `2823FD01...` and DSP `52A4D490...` unchanged. Runner adds -RetainTrace/-RetainFilter and refuses combining retained and callback tensor probes; default decode clears both mechanisms. Current serialization is for single-context diagnostics, not mixed target/draft attribution.

Narrowed retention to ONLY l_out-39 (`retained-last-{a,b}`): variation survives. First decode l_out-39 NMSE .00334537346 / max abs .137812138; logits NMSE .00120552197 / max abs .627546549. Three actual selected tokens still match. Nine-token prefill row 1 and four-token prefill row 0 remain unstable; their final rows match. Thus protecting only the final layer output from inplace reuse is not sufficient, and the all-layer repeatability result cannot be called a final-normalization fix. Broader lifetime/allocation changes or inter-layer work remain relevant. Next: narrower retained layer ranges or a DSP snapshot that does not alter allocator lifetimes. No optimization is enabled from this evidence.

Default-off post-rebuild regression `retained-probe-default-regression`: 28.77 tok/s / 34.76 ms over 128 decode runs, 126 graphs reused, 1381-op replay. Trimmed output exactly matches earlier topk-alias-control. Single run, no new speed-pair claim. All retained diagnostics exit zero; scripts parse and diff checks pass. No inference process remains. Goal remains incomplete: this is localization infrastructure/evidence, not a numerical fix or finished MTP.

## Intermediate tracing continuation - first two layers match exactly

Previous turn was progress: reproduced plain decode logit variation with fusion and replay both disabled. Added a separate, default-off filtered tensor callback in common/debug.cpp, activated only by LLAMA_TENSOR_TRACE_FILE. Unlike the stock text-debug callback, ask=true returns true ONLY for matching tensor names. The probe captures tensor type/op/name/dimensions/strides and raw storage bytes; compare_tensor_traces.py compares logical elements using stored strides, not padding bytes. This still introduces scheduler cuts and is diagnostic instrumentation, not a proof about an untouched fused graph. The runner exposes -TensorTrace/-TensorFilter; the normal decode runner removes both tensor-trace environment variables.

Two repeated three-token plain runs (`tensor-probe-{a,b}`), expanded model, router HVX, fusion=0, replay=0, filter `(-0$|-1$|token_embd)`: all 648 selected records match BITWISE. Reader self-comparison also passes. Yet prompt logits match bitwise and the first decode logits differ (NMSE 0.000735599595, max abs 0.568313837). All three actual selected tokens match. Therefore synchronization at these early-layer probe points does not eliminate the logit variation, and no differing logical element was found among the captured first-two-layer outputs. This redirects the next probe to later layers rather than assuming the first recurrent layer is at fault.

Added layer-boundary filter `l_out-[0-9]+$` to locate the earliest differing layer with less capture data. Build and script checks pass. Installed common library at first probe is SHA256 `8647D0622F0D77C14055432C1E88C46FD219EB323F1F383162E81314869ED52E`; Hexagon host/DSP unchanged. The capture is intended for current single-context diagnostic runs; it does not tag mixed target/draft contexts and must not be used to assert cross-context alignment without additional tagging.

Layer-boundary repeated captures `layer-probe-{a,b}`: 239/240 records match; only prefill `l_out-39` shape [2048,4,1,1] differs (NMSE 7.99136834e-6, max abs .035435915). Row-level comparison localizes that difference to logical row 0; rows 1/2/3 match exactly. All layer outputs for the two single-token decode steps match, and all three sampled logit rows now match bitwise. Therefore per-layer synchronization/batching perturbation suppresses the decode variation in this pair. It does NOT locate the first divergent decode layer in the unperturbed batch, and must not be presented as a fix. The unused-for-plain-output prefill row can matter to MTP hidden-state streaming, so matching final prompt logits does not prove all hidden rows stable.

Next probe captures only h_nextn/result_norm/result_output to distinguish the final hidden/head boundary from many layer-boundary cuts. This is still experimental synchronization evidence. Default decode behavior remains unchanged; tracing runs are not rate claims.

Head-only repeated captures `head-probe-{a,b}` (18 records per run) preserve variation: first decode h_nextn/result_norm NMSE 0.00466323138, max abs 1.18711853; result_output NMSE 0.00128223558, max abs .717455745. Thus the input to the LM head already differs, so neither host logits readback nor the head alone explains the variation. The sampled logit NMSE/max error match the captured result_output comparison. Earlier nine-token prefill h_nextn differs ONLY in logical row 1 (max abs 4.61447906); all other rows, including its final row, match. Again, final prompt logits alone hide unstable hidden rows that MTP may consume.

All three selected tokens match. Callback-at-every-layer and callback-at-final-head produce different repeatability behavior, which supports synchronization/batching sensitivity but does not prove the precise race or operation. Next engineering step should retain/copy selected outputs inside the production batch and retrieve them only after completion, or use a DSP-side snapshot, rather than infer the unperturbed first divergent layer from a probe that suppresses it. These changes must carry a probe-equivalence check because allocation/lifetime changes can also perturb kernel selection and aliases. All diagnostic runs exit zero, reader self-test is exact, script/diff checks pass, and no inference process remains.

## Logit tracing continuation - plain decode is not bitwise repeatable

Previous turn was progress: isolated HMX router expert-ID drift and validated an opt-in precise router. Added default-off `LLAMA_LOGITS_TRACE` at the existing common_sampler_sample synchronization point, before sampler transformations. It logs raw candidate IDs/score hashes/top-two scores and optionally writes full ID/F32-bit rows to LLAMA_LOGITS_TRACE_FILE. This does not install a per-op debug callback or split the production graph. Added selected-token logging after sampling so masked raw argmax is not mistaken for the emitted token. The binary captures contain context IDs, row IDs, and ordered candidate payloads; target is the first sampled context, while draft is separate and has only ten backend candidates in this configuration.

`run-mtp-stage.ps1 -LogitsTrace` selects a label-specific binary and resets the trace when absent. `compare_logits_traces.py` compares full ordered rows, validates finiteness/ID layout, reports NMSE/max error, and aligns target rows by sampling order. With selected-token logs it stops only at an actual selected-token difference; older captures lacking them stop conservatively at raw-argmax difference. Trace rates include dump overhead and are not benchmark claims.

Same expanded model, router-HVX on, 64 generated tokens, independent draft KV: `target-logits-plain` and `target-logits-mtp` differ in initial prompt logits before any accepted draft: NMSE 0.00173994209, max absolute error 0.809248924. Their first raw-argmax difference is step 36, but that is NOT proof of the first emitted-token difference: special-token masking can change raw argmax. Target contexts produce 64 full 248320-value rows each; the MTP draft produces 36 ten-candidate rows. Model loader/contexts differ: plain excludes unused layer-40 weights and has n_rs_seq=0; MTP loads them and has n_rs_seq=1 for rollback. This points to prefill/layout/kernel/state differences, not necessarily acceptance rollback.

Crucial control: `target-logits-plain-repeat` matches the first plain prompt logits BITWISE, but its first single-token decode differs: NMSE 0.00110781633, max abs 0.70877552. The first raw-argmax difference is again step 36, yet their complete trimmed transcripts match exactly. Thus transcript agreement alone concealed numerical variation, and attributing all observed drift to MTP would be wrong. The prior three captures predate selected-token logging. Subsequent replay-off short repeats include it and are the next discriminator. Timers/runtime behavior remain default off; latest llama-common.dll SHA256 `B7F64A8B17A508BAB6587C95A48002898D924378A4174F986FD25F9DA4D853B3` includes the finalized selected-token trace. Hexagon host/DSP hashes remain unchanged from the router-HVX continuation.

Replay-off repeats `target-logits-noreplay-{a,b}` (eight tokens) again have bitwise identical prompt logits and differing first decode: NMSE 0.00129937492, max abs 0.65521431. Selected-token alignment is available and all eight actual selections match, confirming that these compared rows share the same prefix. Descriptor replay is not the sole cause. Added explicit -Replay and -Fusion controls to the runner; default decode runner resets fusion/replay and trace so diagnostics cannot contaminate rate runs. Next discriminator is fusion-off repeated capture, then a preserved-fusion state/tensor probe if variation remains.

Fusion OFF and replay OFF repeats `target-logits-unfused-{a,b}` also match prompt logits bitwise and differ on first decode: NMSE 0.00121566882, max abs 0.759274006. All eight selected tokens match; no raw-argmax difference in this short prefix. Thus neither fusion nor descriptor replay is necessary for the numerical variation. Do not call either mechanism the proven culprit. The next probe must isolate state inputs/prefill state storage and underlying one-token kernels, potentially using the now-unfused path as an instrumentation control. No defaults were changed to the slow diagnostic settings. The comparison script self-comparison returns zero error and no bit/argmax difference. Builds, scripts and diff checks pass; all trace runs exit zero and no inference process remains.

## MTP continuation - dot-to-router oracle exposes HMX expert-ID drift

Previous turn was progress: lossless BF16 gate expansion removed CPU boundaries and improved MTP placement. Extended the existing TOPK_MOE graph with a virtual logits producer, retaining all previous alias test behavior. New `test_mtp_router_chain` produces 256 router scores with F32 matrix multiplication (k=2048), BF16-representable gate values, and deterministic F32 hidden rows; it then executes the actual softmax -> ordered top-8 -> normalization chain. Tests run 32 changing inputs at token widths 1/2/9. Expert IDs compare exactly; weights retain the stock 1e-7 router NMSE threshold. No sorting, tolerance relaxation, or skipped failure.

Default HMX policy: widths 1/2 pass; width 9 fails on its first input (seed 853810), ID element 47: HTP 98 versus CPU 234. Weight NMSE 1.88e-7 also exceeds the 1e-7 gate. The same failure reproduces with replay enabled and disabled (`mtp-router-chain-replay-{1,0}`); this is not explained by replay. Trace selects `hmx-tiled` for 2048:256 x 2048:9. The HMX path converts inputs to FP16, so precision loss is a supported explanation, not proof of a kernel memory bug. Standalone matmul passing at NMSE 5e-4 did not establish ordered expert-ID equivalence.

Global MM_SELECT=1 (HVX) passes all three widths, both replay arms. Added default-off `GGML_HEXAGON_ROUTER_HVX=1` to the host selector: only F32 x F32 matmuls whose output name begins `ffn_moe_logits` bypass HMX. Shared gates and unrelated matrices retain their prior policy. This name-based prototype keeps the change tightly scoped but needs broader model/name coverage before promotion. Targeted selection passes all 192 comparisons, with the width-9 trace now `hvx-tiled`, VTCM 1,122,304 bytes versus HMX 3,295,232. Reproducer: build-router-test.cmd, then run-mtp-router-chain.ps1 -Prefix <unique> -RouterHvx; omit the switch for the preserved failing control, or use -MmSelect 1 for global HVX. The script resets both policies explicitly and reports failures without stopping before the second replay arm.

Runtime host rebuilt with the gate off by default. Installed ggml-hexagon.dll SHA256 now `2823FD01D22F7EDDF095DBFE4D1DF8B2E14EBC45A0DD16C04562A2AEFE37E654`; DSP SHA256 unchanged (`52A4D490...`). OpenCL stub retained. This fixes the synthetic router chain when enabled, not MTP target equivalence. First-divergent target tensors and accepted/rejected-cache invariants remain open; this continuation prioritized the newly proven router defect over speculative sampling instrumentation.

Frozen expanded-model 64-token MTP pair: `mtp-router-hvx-control` 26.62 tok/s, 27/35 accepted; `mtp-router-hvx-candidate` 26.44 tok/s, 27/36 accepted. Transcripts differ at character 53; candidate/plain comparison differs at character 134. No speed or end-to-end correctness promotion. Device traces confirm actual layer-40 router selection: nine-token prefill/catch-up uses HVX 56 us versus previous HMX 40 us (16 us premium); one-token router ~20 us and two-token ~20-21 us unchanged. These are separate diagnostic runs, not a statistical latency claim. Existing direct-logit alias oracle runner now anchors its parameter filter so the new intentionally failing default dot-chain case does not silently broaden its original coverage.

Post-rebuild normal-decode regression (`router-hvx-default-regression`, flag off): 128-token Hello, 28.72 tok/s / 34.82 ms, 126 graphs reused, 1381-op replay. Trimmed generation exactly matches `topk-alias-control.log` from the earlier frozen baseline. One run, not a new best pair. Anchored original alias-test filter selects exactly three cases and passes all 96 changing-input comparisons (`router-alias-scope-check`, alias enabled + replay). Builds, diff checks and script syntax pass. No inference/test process remains.

## MTP continuation - lossless gate expansion removes CPU splits

Previous turn was progress: it measured catch-up and identified CPU router/shared-gate boundaries. No inference process was live at continuation start. Created a separate `Qwen_Qwen3.6-35B-A3B-Q4_0-mtp-f32.gguf`; the original patched Q4_0 file is unchanged. Only `blk.40.ffn_gate_inp.weight` and `blk.40.ffn_gate_inp_shexp.weight` are expanded from BF16 to F32. This is not quantization: each BF16 bit pattern is shifted into the high 16 F32 bits, preserving all finite values exactly.

`build-expand-mtp.cmd` builds the converter and independent readback verifier. The converter refuses an existing destination and uses gguf metadata infrastructure to recalculate every offset, then streams unchanged payloads into the new file. `/MD` is required because FILE pointers cross the ggml-base DLL boundary; the first `/MT` build failed with an empty destination. Only that confirmed zero-byte failed output was removed before retry. The successful new file is 20,837,295,744 bytes versus original 20,836,243,072. Independent verifier passed: all 20,824,199,680 unchanged payload bytes match and all 526,336 expanded values match bitwise. The source is never opened for writing.

Candidate loader places every tensor on HTP, with no CPU model buffer. Draft reserve is one scheduler split (previously five), and graph replay captures a single 53-op warmup/47-op prompt-catch-up graph. First 64-token candidate `mtp-f32-gates-candidate` reached 26.46 tok/s (37.80 ms/token), 27/36 drafts accepted. The preceding BF16 control's 20.13 timing is excluded from speed comparisons because it overlapped file-level verification. The two generated transcripts differ at character 53. This experiment addresses placement cost, not MTP target equivalence.

Eight-token device diagnostic `mtp-f32-gates-profile.detail.log`: actual single draft is one 40-op submission, 5.30-5.40 ms, instead of three submissions totaling ~5.37 ms; device arithmetic cost barely changes while host round trips disappear. Two-token catch-up is one 48-op submission, 1.46-1.59 ms steady, versus three submissions ~1.6 ms. Head/TOP_K still dominate draft (~2.94/0.99 ms). Two-token target verification remains ~54 ms, now the largest end-to-end obstacle. Device verbosity imposes heavy logging overhead; these logs are not throughput claims.

Extended existing test_mul_mat infrastructure with MTP gate shapes (k=2048, outputs=1 or 256, tokens=1/2/9), 32 deterministic changing inputs per case, weights rounded to BF16-representable F32 values. Unique graph IDs permit changing-input replay through the previously added repetition mechanism. All 192 CPU comparisons passed (`mtp-f32-gate-oracle.log`, exit 0), using stock matmul NMSE tolerance 5e-4. This does not compare CPU BF16 kernels against CPU F32 kernels, does not guarantee exact expert IDs, and uses synthetic rather than model weights. Next quality work needs the dot-to-router chain with ordered IDs and first-divergent target tensors, plus acceptance/rejection cache state.

Clean serialized 64-token pair (no concurrent verifier/build): `mtp-f32-clean-control` 19.59 tok/s (51.04 ms/token), `mtp-f32-clean-candidate` 26.39 tok/s (37.90 ms/token); both accept 27/36 drafts. The candidate's generated text exactly matches its earlier 26.46 run, SHA256 `29e6d49a34a817f5299dd208a72bd65319bd5181631458432fc49d7129f079b1`. It differs from the clean BF16 control at character 200 and from the earlier plain CLI at character 134. Thus the placement gain repeats (~35% over the clean BF16 control), but the candidate is still slower than plain decode and fails target-equivalence. Do not replace the default model or enable MTP as an accepted optimization.

Lower-overhead synchronized host profile `mtp-f32-host-profile`, 32 tokens: median draft decode 6.045 ms, two-token catch-up 2.223 ms, target counter delta 56.731 ms (one startup .033 ms excluded from any meaningful interpretation). Compared with the earlier independent-KV BF16 diagnostic, catch-up improves from 15.214 ms and draft from 13.171 ms; target stays essentially unchanged. Final one-token catch-up 1.804 ms. Profile rate 27.77 is diagnostic only and uses a different 32-token acceptance mix (14/16), not an accepted speed record.

The first gate oracle used per-node evaluation, so graph IDs alone did not prove replay. Changed only this derived case to whole-graph evaluation and reran `run-mtp-gate-oracle.ps1`: 192 changing-input CPU comparisons with replay enabled and 192 disabled passed. Verbose stderr now proves one-op capture for the four token-width 1/2 cases and a replay hit. The width-9 cases pass but do not log replay capture. Logs `mtp-gate-oracle-replay-{1,0}.log[.err]`. Build/diff checks pass, scripts parse, and no inference/test process remains. Default decode runtime/DSP are unchanged by this continuation; only test binary and separate model artifact were added.

## MTP continuation - catch-up overhead and CPU boundaries

Built optional host-stage instrumentation in common/speculative.cpp and a zero-output graph-pruning experiment in qwen35moe.cpp. Both are environment-gated, default off. The first pruning attempt omitted allocation dependencies and asserted in backend tensor upload; repaired it by retaining the attention output graph dependency and creating out_ids only on the full path. Repaired runs exited zero. This is not correctness proof.

Independent-KV standard MTP itself failed exact greedy transcript comparison against non-speculative CLI: `mtp-stages-plain` versus `mtp-stages-off`, first differing character 53. Profiling also changed later text. Therefore neither synchronized rates nor superficially plausible answers establish quality. Reproduce with run-mtp-stage.ps1 and compare_cli_transcripts.py. All tests keep LLAMA_FUSE_MTP_VERIFY unset and router alias off.

Frozen repaired-binary 64-token pair: `mtp-kv-control2` 20.43 tok/s, 28/35 drafts accepted; `mtp-kv-only2` 21.96 tok/s, 27/36 accepted. Exact trimmed generation hashes differ: control `be37a5bb0d40f80ff0ca3a4162aa3820c687628777cbdf373c34ecc16d1c400d`, candidate `9d1a265be56a126708ab3e595331537213d2e781ee896eb014bb6aa2258741f1`; first difference character 164. No speed promotion.

Host diagnostics `mtp-stages-profile2` and `mtp-kv-only-profile2`, 32 tokens, synchronized and --perf:

| Steady stage median | Full catch-up | KV-only experiment |
| --- | ---: | ---: |
| Two-token target counter delta | 56.142 ms | 55.738 ms |
| Two-token catch-up | 15.214 ms | 1.220 ms |
| Single draft decode | 13.171 ms | 20.472 ms |

There are 16 draft/catch-up rows per arm. One initial two-token target delta is a startup artifact, not a genuine verification latency. Earlier `mtp-stages-profile` target values were all zero (perf disabled and incomplete counter selection); parser warns. Hidden-history copies are only microseconds. The one-token final catch-up shape is 50.774 ms control versus 1.150 ms candidate, which deserves separate allocation/scheduling investigation.

DSP traces `mtp-kv-dsp-control.detail.log` and `mtp-kv-dsp-candidate.detail.log` (eight tokens, verbosity 5, device profile 1) are not rate runs. Normal two-token catch-up consists of attention ~0.76 ms, FFN ~0.73-0.89 ms, hidden norm/residual ~0.09 ms in three HTP submissions. Candidate catch-up is one 15-op segment ~0.62 ms. The normal catch-up has no LM-head op: output-row pruning already eliminates it. Actual draft consists of ~0.69 ms attention, ~0.47 ms FFN, and ~4.21 ms final segment, of which head MUL_MAT ~2.94 ms, TOP_K ~0.99 ms, PAD ~0.17 ms. Target two-token verification is 2024 ops and ~54 ms; final single-token target is 1414 ops/~46 ms in this diagnostic setting.

Loader evidence: `blk.40.ffn_gate_inp.weight` (BF16) and another weight fall back to CPU; draft reserve says five scheduler splits, target reserve one. No fit_op-capacity flush line occurs. The three HTP batches are scheduler segments separated by CPU work, not demonstrated descriptor-capacity flushes. `mtp-sched-boundaries.detail.log` with GGML_SCHED_DEBUG=2 proves split #1 runs the router `ffn_moe_logits-40` MUL_MAT on CPU, and split #3 runs the shared-expert gate MUL_MAT on CPU (the truncated weight name is blk.40.ffn_gate_inp_; its 4 KiB size identifies the 2048-element BF16 gate). Large host-versus-DSP catch-up cost is consistent with CPU work/transfers/scheduling, but timing attribution requires finer instrumentation. Next: test lossless BF16-to-F32 weight expansion and compare KV tensors plus acceptance/rejection state before adopting pruning. Keep all experimental MTP paths opt-in. All diagnostic runs exited zero and no inference process remained after the boundary trace.

## Next continuation - forced router aliases and changing-input replay pass

Previous turn classification: progress (installed the alias-capable DSP library, measured both paths, and added arithmetic coverage). No live inference/build process at the start of this turn.

Extended the existing TOPK_MOE case with `alias_mode`: 0 disjoint, 1 normalized-weight output shares dead logits storage, 2 full index row shares dead logits storage. These views force reuse without relying on the allocator to happen to overlap. Verbose add-tensor logs prove identical data addresses/offsets for logits and the selected output. Both aliased cases pass CPU comparison with the fused kernel executed. Selected expert IDs are compared exactly; normalized weights retain the stock NMSE tolerance and original order.

Added test repetitions with reinitialized inputs on the same graph/allocation. The first repeated test did not exercise replay because the backend requires `graph->uid != 0` and the stock harness leaves it zero. Assigned unique stable nonzero IDs only to repeated test graphs, using the same internal-header access pattern as test-alloc. Capture/hit logs now prove actual replay. Inputs for these computed-logits cases are deterministic mt19937 seeds `0x51A17 + iteration`, uniform [-2,2], so failures are reproducible. All three shapes pass 32 iterations in each of fused+replay, overlap-fuse-off+replay, and fused+no-replay: 288 CPU comparisons total. Aliased control graphs capture 5 ops; fused graphs capture 2. This is a CPU numerical oracle, not a claim of bit-identical weights. Reproducer: build-router-test.cmd then run-router-oracle.ps1. Logs `topk-oracle-fused-replay`, `topk-oracle-control-replay`, `topk-oracle-fused-no-replay`.

Ran three additional same-binary control/candidate prompt pairs with seed 1234, temp 0, context 256, batch/ubatch 128, profile off, 64 requested generated tokens. Each has 63 measured single-token decode evaluations; the first generated token comes from prefill. All exited 0, reused 62 graphs, and have steady replay op counts 1381/1264. Candidate flag is still opt-in.

| Prompt class | Control tok/s (ms/token) | Candidate tok/s (ms/token) | Normalized transcript SHA256, both arms |
| --- | --- | --- | --- |
| Python Fibonacci / code | 28.66 (34.89) | 29.20 (34.25) | `1660283955CE35C4120EBA9916C389C0005F0DDE4DC9F1AB4BF2038FAADC3999` |
| Arithmetic, 17 x 23 | 28.78 (34.75) | 29.26 (34.18) | `8DB454508F76A901CC456EBE7F23E6B06B1CBE2C5458527EFB256AEB43FEA26C` |
| Sky / prose | 28.72 (34.81) | 29.00 (34.48) | `481B126E72C186DB64A68EDE5137310B45825B4A294DA0B89479F42DA1404F4B` |

The SHA256 oracle trims leading/trailing whitespace then hashes UTF-8 text. Full transcript contents are compared case-sensitively; this is not a semantic-similarity gate. Reproducer: run-router-gate.ps1 -Prefix <new-prefix>. Current logs use prefix router-gate-1. Host/DSP library hashes are unchanged from the prior continuation. The decode runner now explicitly selects replay=1 and verbose=0 so a preceding oracle invocation (which ends with replay disabled) cannot silently contaminate rate runs. Build/test changes did not rebuild the runtime host.

This gives direct evidence for both alias layouts and changing-input replay and a modest 0.33-0.64 ms/token speed reduction across three prompt classes. It does not explain the initial ungated-build password mismatch, prove deeper contexts, or establish MTP target equivalence. Leave all-layer routing opt-in for now. The current common debug callback always asks for every tensor, even with a filter, which splits every op and prevents the fused execution under investigation; avoid using that callback as proof about the production fused graph. A future first-divergent-tensor probe must preserve the relevant fuse.

Read-only MTP audit reconfirmed that shared draft KV has mutable tensor storage but `seq_rm` and `apply_ubatch` return early when `other` is set. Thus the bonus draft write does not commit cache-cell metadata like the independent draft context. A correct fused-tail design needs an explicit writable draft/verification ownership scheme, with accepted/rejected-prefix tests; enabling the existing shared-cell view remains unsupported by the evidence. No MTP runtime behavior was changed this turn.

## Continuation - all-layer top-k: measured, not enabled

The previous turn was progress: it established the alias rejection and retained a signed rollback pair. No live inference/build process remained at continuation start. Installed the already-built alias-capable v81 skel (848432 bytes, SHA256 `52A4D490D48524236D300F53C2A8DFDB4DD81886ED04F71BB89C9901C2453F52`) through `rebuild-gdn-state.cmd`, which builds, generates the catalog, signs, and copies the host. `signtool verify /pa /c` accepted the installed so. The host now allows input/output overlap only when `GGML_HEXAGON_MOE_TOPK_ALIAS=1`; output/output overlap is always rejected. Default/unset retains the previous host policy. The other correctness-negative matchers stay disabled.

The first ungated trial used all 40 router fuses (1264 ops) and reached 29.00 tok/s / 34.49 ms, but differed from the preceding PHP control at the PDO password. After adding the gate and rebuilding, four fixed-seed runs on the same host/skel were: control 28.70 / 34.85 ms, candidate 28.95 / 34.55 ms, candidate 29.00 / 34.48 ms, control 28.96 / 34.53 ms. All four later transcripts match exactly after trimming trailing output whitespace. All exit 0, reuse 126/128 graphs, have one scheduler split and one logits copy. A 29.00 candidate beside a 28.96 control is not persuasive wall-time improvement; the first mismatch prevents default enablement. No model quality or MTP completion claim follows from these canaries.

The existing backend TOPK_MOE cases lacked the actual single-row 256/top-8 normalized shape. Added that case with a computed logits producer and ordered comparisons, so index/weight association cannot be hidden by sorting the outputs. CPU comparison passed; verbose logs confirm MOE+TOPK was submitted. The harness allocated disjoint buffers, so this is arithmetic coverage only, not an alias or replay proof. Reproducer: `build-router-test.cmd`, then `test-backend-ops.exe test -b HTP0 -o TOPK_MOE -p computed_logits=1` with OPBATCH=3072, MOE_TOPK_ALIAS=1, VERBOSE=1, and the normal ADSP_LIBRARY_PATH. Log: `topk-alias-backend-test.log[.err]`.

Profile control/candidate use the same fixed Hello input, four tokens, PROFILE=1 and log level 5. Last warm batches:

| Stage | Control | Candidate |
| --- | ---: | ---: |
| DSP whole batch | 32.881 ms | 32.389 ms |
| Sum of op durations | 32.017 ms | 31.588 ms |
| Batch minus op sum | 0.864 ms | 0.801 ms |
| Steady op count | 1381 | 1264 |

Candidate operation totals: dense MUL_MAT 10.299 ms (includes output.weight 2.929 ms); routed gate/up MUL_MAT_ID_NX 4.599 ms; routed down MUL_MAT_ID 2.672 ms; MUL_MAT+ADD 2.913 ms; MUL_MAT_NX 1.467 ms; recurrent GATED_DELTA_NET+CPY 1.619 ms; SSM_CONV+WIN 1.399 ms; GET_ROWS 1.146 ms; router MOE+TOPK 0.610 ms; attention FLASH_ATTN_EXT 0.397 ms; ADD+NW 0.373 ms; ADD+SG 0.301 ms. The 560 ops under 8 us still consume 3.142 ms. Full per-op/per-weight breakdown is reproducible with `parse_last_batch.py`; `parse_profile.py` gives every batch. A negative cycle-stamp gap in candidate batch 2 is a wrap artifact, not negative latency; use batch duration/op sum for that batch.

New runner `run-decode.ps1 -Label <name>` uses seed 1234, temp 0, batch/ubatch 128, context 256, 128 tokens, and profile off. Add `-TopkAlias` for all-layer routing, or `-Tokens 4 -Profile 1` for profiling. It redirects native output directly to UTF-8 logs, avoiding PowerShell's UTF-16/wrapped native stderr conversion. Candidate and control profiling logs are `topk-alias-profile.log.err` and `topk-alias-control-profile.log.err`.

Next: force aliasing in the router oracle, test changing inputs and replay, and collect first divergent tensor evidence for the observed password mismatch. Broaden fixed-token prompt classes before accepting the fuse. MTP still needs exact target-equivalence and faster verification/catch-up; existing drifted fused-tail mode remains off. References consulted: https://neural.download/docs/research-workflow-playbook.html and https://neural.download/experiments/ornith-15-b70/notes/2026-08-23-ornith35b-shared-gate-residual-rms-positive.html (same-frozen-binary control and first-divergent-tensor discipline).

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

## 2026-09-26 19:13 — shared-expert down epilogue changed the INSERT

The 17:40 token's next concentrated pile after the expert-weight mul is the shared-expert down GEMV: 40 `MUL_MAT`s, 664 µs, about 16 µs each, already on the DMA band (Q8_0 or Q5_K, K=512, N=2048). Around it, the SwiGLU is 247 µs and the f32 gate dot is 274 µs. Both are almost all launch. `ADD+SG` is another 308 µs.

`HTP_OP_SHEXP_EP` keeps the down GEMV as `op_matmul` with that op's kernel params. It runs SwiGLU with the same HVX sigmoid and `hvx_mul_mul_f32_aa`, the gate dot with `vec_dot_f32_f32_aa_1x1`, then the same sigmoid-scale-add as `ADD+SG`. Decode only. `absorb_shexp_down` is the matcher. It is not called.

Same 128-token command (`GGML_HEXAGON_OPBATCH=3072`, `-n 128 -b 128 -ub 128 -p Hello --temp 0 --device HTP0 -ngl 99 -fit off --load-mode none`). Log `htp-qwen-shexpep.log`.

- the fuse line fired: `fused shared-expert down into SHEXP+EP (k 512 n 2048)`. `ADD+SG` logged first, then this op replaced it. No RMS line. No GDN-state line
- steady graph is 1261 ops (1381 minus 40 SwiGLUs, 40 downs, and 40 gate dots). Replay captured uid 20, then hit. Graphs reused 126/128
- decode: 4424.52 ms / 128 = **34.57 ms/token = 28.93 tok/s**. Seed 2302241197
- the text is PHP, but the INSERT is `username, password, email` and the run ends at `violation: 10`. It does not contain "You have an"

28.93 is faster than 28.4 and is not the rate to keep. The installed skel is the 844176-byte PHP file again. Its catalog is `npu_campaign/libggml-htp-php-844176.cat` (verifies that so). A copy of the so is `npu_campaign/libggml-htp-v81-php-844176.so`. The v81 prefix link that contains `SHEXP+EP` is 848432 bytes and is not installed. The so beside the build-tree catalog was overwritten with that 848432-byte file and the catalog there matches it. Do not copy that pair into `pkg-wos/lib`.

Confirm on the restored skel, log `htp-qwen-shexpep-off.log`: PHP through "You have an", no email column, seed 1241911316, **28.62 tok/s**, 34.94 ms/token, replay hit 1381 ops, graphs reused 126/128. No `SHEXP+EP` line. `ADD+NW` and `ADD+SG` still log. One run is not a pair. **28.4 stands.** `pkg-wos/bin/ggml-opencl.dll` is the stub, 102400 bytes. `HTP_OP_MAX_BUFS` is 64. The Q5_0 backup was not restored. Output weight was not requantized.

## 2026-09-26 19:30 — fused MTP verify tail drifted

Codex (read-only, gpt-5.6-sol, session `01a0e09d-e137-7262-95a2-69c81bc3d344`) on the one-submit splice:

- For one sequence, `h_shift[0]` is `pending_h` and `h_shift[k]` is `t_h_nextn[k-1]`. A one-token batch is only `pending_h`.
- The tail and the later bonus `draft()` have to use the same MTP KV. The target hybrid cache normally omits layer `n_layer()`. The draft context owns that layer today.
- A construction-time flag has to participate in `llm_graph_params::allow_reuse`. With the flag off, memory creation and the decoder graph stay on the previous path.
- `draft()` still decodes the bonus token. The tail's last row predicts the token sampled from the target, which is the token `draft()` then consumes.

`LLAMA_FUSE_MTP_VERIFY=1` is that flag. The target graph, after `t_h_nextn` is saved, shifts `pending_h` with `t_h_nextn` and runs blk.40 attention so its K/V land in the target cache. The MoE and the MTP LM head stay out of that graph. The draft context is created with `ctx_other` set and its layer-40 K/V tensors alias the target attention cache. `is_mem_shared` then skips the catch-up `llama_decode(ctx_dft)`. `process()` and `accept()` copy the accepted hidden row into the seed the next target graph reads.

Two temp-0 `Hello` runs, `--spec-draft-n-max 1`, logs `htp-qwen-mtp-fuse.log` and `htp-qwen-mtp-fuse2.log`:

- first reply keeps "a question" and the banner is Generation 22.2 t/s
- second reply says "questions" and the banner is Generation 21.3 t/s

The second sentence is the failure already seen when a draft-context write was kept in place of a recomputed row. `llama_kv_cache` views no-op `seq_rm` and `apply_ubatch` (`TAG_KV_CACHE_SHARE_CELLS`), so the shared layer does not commit the bonus step the way the separate draft cache did. Leave `LLAMA_FUSE_MTP_VERIFY` unset. `run-mtp-fuse.cmd` is the reproducer.

Flag off, same 128-token completion, log `htp-qwen-fuse-off.log`: PHP through "You have an", replay hit 1381 ops, **28.56 tok/s**, 35.01 ms/token. No fuse line. One run. The installed skel is still the 844176-byte PHP file. `absorb_shexp_down` stays uncalled.

## 2026-09-26 19:55 — top-k alias is overlap, and the PHP pair is 28.42

The 17:40 steady graph runs `MOE+TOPK` on layer 39 only. Layers 0–38 keep softmax, argsort, get_rows, and `MOE+WNORM` (about 44 µs). The matcher skip on the decode graph is `overlap`. The prompt graph skips on `logits` because the row is wider than one token. Logits are a dead 1 KB slot the allocator reuses for an output of the same chain.

`op_moe_topk` copies the 256 logits into scratch before it writes the index row or the 8 weights, so that alias is safe. The installed skel still rejects `src->data == dst->data` with `NO-SUPPORT`. Dropping the host overlap check fused all 40 rows (count 40) and the DSP aborted. Log `htp-qwen-topk-all.log.err`. No text.

The overlap check is restored. `argsort-ops.c` allows the alias for a later skel. Rebuilding v81 from that source produced another 848432-byte so (sha256 `52A4D490D48524236D300F53C2A8DFDB4DD81886ED04F71BB89C9901C2453F52`). That is not the older SHEXP link (`FD3AE079…`) and it was not copied into `pkg-wos/lib`. Installed skel remains 844176 bytes, sha256 `28F47160…`.

Same command as the 28.4 pair, restored host, no new fuse. Both reach "You have an". Neither INSERT has an email column. Replay hit 1381 ops. Graphs reused 126/128.

| log | tok/s | ms/token |
|---|---:|---:|
| `htp-qwen-topk-revert.log` | 28.46 | 35.14 |
| `htp-qwen-topk-revert2.log` | 28.42 | 35.19 |

35.19 and 35.14 beat 35.21 and 35.23. The quoted pair is **28.42 tok/s**. `pkg-wos/bin/ggml-opencl.dll` is the stub, 102400 bytes. `HTP_OP_MAX_BUFS` is 64. The Q5_0 backup was not restored. `absorb_shexp_down` is not called.
# Activation-low reuse: capture-level savings preserve sampled bits

Closed C202 capture 92078/native0: 70/70 selected tensor records bit-identical to 8DC1, bounded two logits/tokens identical to B090. Eight Q8_0 T9/T1 projection gates pass (NMSE range 4.7011909e-15 to 1.74528379e-12). Unchanged-threshold rotated-weight negatives reject ssm_out/down at 2.01364224/1.99739199, native1. Independent CPU bounded two-token gate passes with different logits; not full qualification. Baseline three hashes independently restored.

Prefill 217466.16ms versus prior capture 337671.14ms, 35.60% lower, is an ordered capture-level signal only; no isolated HMX/preparation attribution or independent benchmark yet. Two-token 12.38tok/s is not a qualified decode measurement. Marker reuses owned activation-low across synchronous column/nx jobs; weight-low reconstruction remains repeated. Full128 control/repeat and CPU gates, then stage profile, are required before broader claims. Diagnostic control remains OFF; no correctness or production promotion.
# Activation-low reuse full128 arm A: exact prior-control equivalence

33033 armA native0. Strict all128 logits/selected tokens bit-identical to B090, gate0. Independent CPU gate still first diverges93 (60239/3812, margins0.0516681671/0.104560852), gate1. Prefill217261.47ms/traced128 decode12.94tok/s; one ordered arm confirms output-preserving prefill signal, not independent benchmark or correctness promotion. ArmB remains live in same wrapper; final repeat and restoration gates pending. No code/binary mutation while matched pair active.
# Activation-low reuse pair closed: repeated prefill savings, no decode-gain claim

33033 terminal0/both native0. All128 A/B logits/tokens bit-identical; each versus corresponding B090 strict0. Both independent CPU gates first differ93 unchanged; baseline three hashes independently restored. Prefill217261.47/217255.66ms versus B090337350.42/337151.04ms (~35.60% lower). Traced128 decode12.94/12.83 versus12.83/12.89tok/s gives no convincing decode improvement. Ordered repetitions and exact tested-sequence preservation, not independent ABBA/production qualification. Next stage profile; weight-low prep and paired exports remain per tile, and whole HMX worker time must not be called pure HMX execution.
# Activation-low reuse stage profile closed

19443 terminal/native0, exact four-row B090 control gate0, bounded CPU four-token gate0 (logits different), baseline three hashes restored. Complete1384 decode batches74977.333us versus74856.667us: no meaningful decode gain. Complete1982 prefill batch217076716us, plain matrix178222170us, matrix+ADD24502221us, matrix NX13308123us; largest QKV operators ~3.41seconds each (T9). Whole-operator attribution only: scalar preparation, HMX and export not isolated. Control stays OFF; full CPU/coherence/MTP qualification remains incomplete.

B090 corresponding complete1982 batch337060257us; plain matrix277814883us, matrix+ADD37866117us, matrix NX20333615us. Their reductions99.593/13.364/7.025seconds track the~119.984second wall reduction. Gathered ID_NX827993->827643us and ID137026->136549us essentially unchanged, consistent with plain-only implementation scope. Next separate OFF-default capacity-safe operand-prep/HMX/export instrumentation; exact gates before interpreting timing. No runtime currently live at this closure.
# FARF timing return path is not observable in current host capture

1865 timing-candidate capture33818/native0 preserves all70 internal tensor records and complete2 logits/tokens exactly versus C202. Local ssm_out T9 NMSE1.74528379e-12 passes; wrong-weight ssm_out/down2.01364224/1.99739199 reject. Baseline three hashes independently restored. Completed detail/stderr contain zero HMX_RESIDUAL_STAGE records, matching prior retain-probe limitation. This is no returned stage timing, not zero stage cost. Do not qualify a performance claim from FARF-only instrumentation. Next explicitly tagged aggregate counters through verified profile response, preserving real PMU mode and normal layout, matching host decoder and guards; not implemented yet.
# Observable residual stages: scalar operand preparation is the diagnostic bottleneck

40585 terminal/native0; exact four-row C202 profile logits/tokens identical, bounded CPU4tokenspass, baseline3hashesrestored. Complete1982/T9 batch passesstrictparser:256records, prep214160734us/product_issue1232us/export1761807us/worker216332279us, jobs16390/products52280, batchwall217659742us. Prep accounts98.9962% ofworker and98.3924% ofbatch. Layer0QKV prep3.395605s outof3.427601sworker. Explicit tagged profile return reaches host; FARF-only route didnot. Product_issue is not completed/pureHMX time, completionmaylandexport, instrumentationperturbs timings.

Decode3complete1384batches74707.333us versusC20274977.333us essentiallyunchanged; no returned residualstage records inT1batches, parserrejectsabsence ratherthanclaimingzero. Prep optimizations address diagnosticprefill, notproven decodegain. Candidateidea: hoist32HFrawcolumnscales per tile, avoid1024repeatconversions withoutchanging arithmetic; separateOFFcontrol/exactgatesneeded. Then return to gatheredexpert precision/firstdivergence/productioncoherence andMTP, notsubstitutediagnosticspeedforfullgoal. PMU/overflowruntime andfull128instrumented-equivalence coveragepending.
# Expert-down affine sum must use its separately stored Q8_1 field

LocalGGUF layer0expert down isQ4_1[512,2048,256], unlikeQ4_0up/gate. ScalarQ8_1reference storesHFd andHFsum separately; sum usesoriginalF32d timesintegercodesum, notstoredHFd. Offlinechecker nowhas explicitq4_1_q8_1_ref algebra withrawQ4_1nibbles/d/m, F32blockscaleproducts andF64accumulation, notcompiledARMbitproof. Syntheticfixture distinguishes stored_s30.34375 fromstored_d*sum30.351318359375; zero/signedtie/affineoffset/codedot/malformedraw andframe/mappingtests passfive methods. Realcapture/runtimepacking/CPUroundingqualification stillpending. Longhoistcapture61732remainslive; noperformanceconclusion orreplacementinference.
# Q8_1 offline reference has native ARM64 byte evidence

Installed ggml-base.dll SHAAC592F1C55B02F44D0AC43ECB021D35E3A8929E69484C3370FF0A11CD85C47A4 exportsquantize_row_q8_1_ref. Pythonoraclecodes/HFd/HFs matcheverybyte over304blocks, zero/signedhalf-ties/seededmixedsign2^-24..2^8fixtures; sixunittestmethodspass. NotactualoptimizedCPUactivationbytes, ARMdot-reductionbitproof orNPUpacking; ggml-cpu directdotfunctions notexported. No additionalinference/hostDSPinstall; currenthoistcapture61732stilllive.
# Active CPU quantizer is not its scalar reference at half ties

Installed ARM64 ggml-cpu.dll SHA70C3D396D0127A675F92560533E1CC71E399C9305B679F4FC96625DD2831515D exposes CPUtraits. Q8_1from_float fixture withamax127 confirmscodes127/0/0/2/-2 for127/+.5/-.5/+1.5/-1.5, versus scalarreference127/1/-1/2/-2. Actualquantizerlane mustremainseparatefromreference; notproofthiscausesCPU93 andnotauthoritytopatchtierules. ViaQ4_1vec_dottrait,nrc1singleblockd0/m2/ones yields64 exactlymatchingaffineoracle; general reduction/nrc2/GEMM/actualcaptureunqualified. Sevenoracleunittestmethodspass. Originalhoistcapture61732stilllive, no otherinference.
