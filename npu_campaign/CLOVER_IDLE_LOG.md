# CLOVER_IDLE_LOG (append-only) - agent `idle-host`
Goal: the host main thread burns ~0.99 core during HTP0 generation in every configuration (port log 06:35). Make it sleep while the NPU works
(target < 0.15 busy cores on Qwen3.6 plain decode), without tok/s loss.
Worktree `llama.cpp-hexagon-clover-idle`, branch `clover-idle` (from `clover` e695295). Build `npu_campaign/build-clover-idle.cmd` (dir `build-clover-idle`),
packages `npu_campaign/pkg-clover-idle-<x>` via `pkg-clover-idle.ps1 -Name <x>`. Tools in `npu_campaign/clover-idle/`.

## 2026-10-30 10:15 Setup
- Worktree + build OK; `pkg-clover-idle-base` = clover e695295 unchanged (host ggml-hexagon.dll D7881219, skel 1AD13AF0).
- Tools: `clover-idle/PcSampler.cs` + `pcsamp.ps1` (under the NPU lock: starts the run, picks the busiest thread, SuspendThread/GetThreadContext
  every 2 ms, walks the AArch64 frame-record chain, symbolizes with dbghelp, prints module/function histograms + kernel/user split of that thread).
  `clover-idle/TimerProbe.cs` (Win32 sleep granularity, no NPU).
- Timer granularity on this machine (TimerProbe, 200 waits each): Sleep(1) = 15.6 ms (default 15.6 ms tick); high-resolution waitable timer
  (CREATE_WAITABLE_TIMER_HIGH_RESOLUTION) rounds every request up to the next 0.5 ms tick: 50-250 us -> median 505 us, 500 us -> 1023 us,
  1000 us -> 1512 us; SwitchToThread 0.1 us (no sleep). => a timed sleep costs >= 0.5 ms of latency; per-batch waits (~0.5 ms each) cannot just sleep.
- Host side facts (ggml-hexagon.cpp): batches go through an opqueue of depth 8 (GGML_HEXAGON_OPQUEUE); `flush_batch` waits for ONE response when
  8 are in flight, `flush_sync` (synchronize / get_tensor_async) waits for all. Every wait is `dspqueue_read(..., DSPQUEUE_TIMEOUT_NONE)` on Windows
  (OPPOLL=1: dspqueue_read_noblock spin). Responses carry the DSP-side batch time (`rsp.usecs`, always filled).
  Windows libcdsprpc.dll (qcnspmcdm8480, 2026-08-21) imports D3DKMTWaitForSynchronizationObjectFromCpu / WaitForSingleObject / QueryPerformanceCounter,
  no Sleep/SwitchToThread; it exports no dspsignal API.

## 2026-10-30 10:45 First measurements: the busy thread is in KERNEL mode and does not care how ggml waits
Code (branch clover-idle, uncommitted yet): `GGML_HEXAGON_WAIT` = 0 legacy dspqueue_read (default) | 1 dspqueue_read_noblock + 0.5 ms high-res
waitable-timer sleeps (optional spin tail `GGML_HEXAGON_WAIT_SPIN=<us>` driven by a per-batch DSP-time predictor) | 2 dspqueue packet callback ->
condition variable, explicit noblock reads. `GGML_HEXAGON_WAIT_STATS=1` prints counters.
A/B `clover-idle/idle-ab.ps1 -Label q1` (one lock hold, Qwen3.6-35B-A3B Q4_0 plain decode on HTP0, 600 tokens, -t 8; cores = process CPU over the
generation window, top-thread = busiest single thread over the same window, kernel = its privileged time):
| arm | tok/s | host cores | top thread | of which kernel | stdout hash |
| --- | ---: | ---: | ---: | ---: | --- |
| base (pkg-clover-idle-base) | 28.33 | 0.982 | 0.982 | 0.982 | 9ECA6290 |
| WAIT=0 (new pkg, legacy path) | 28.10 | 0.984 | 0.984 | 0.984 | 4E88DE86 |
| WAIT=1 (poll + 0.5 ms sleeps) | 28.60 | 0.984 | 0.984 | 0.984 | DCF010DC |
| WAIT=2 (packet callback + cv) | 28.34 | 0.983 | 0.983 | 0.983 | B20A915E |
=> (1) ~100% of the host CPU is ONE thread spinning in kernel mode (privileged time = total). (2) Replacing the main thread's blocking
dspqueue_read by sleeps (WAIT=1) or by a driver callback (WAIT=2) changes nothing: the burner is not our waiting thread, or it is a kernel path
that every dspqueue access triggers. Neither WAIT mode costs throughput (28.1-28.6 tok/s, within noise). (3) The stdout text differs between
every run, including two WAIT=0 runs: Qwen3.6 plain decode on HTP0 is not run-to-run deterministic (to be confirmed with a repeated arm),
so "bit-identical" has to be judged against that noise.
The first PC-sampling attempt sampled a CPU-threadpool worker by mistake (1-s census picked it; 88.7% ZwWaitForAlertByThreadId). The census
was rewritten to list every thread with user/kernel time and start address; re-run queued.

## 2026-10-30 11:05 ROOT CAUSE: the "0.99 busy core during generation" is a measurement artifact (process teardown), not a spin
- Census v2 (`pcsamp.ps1 -Label qwen-base2`, pkg-clover-idle-base, Qwen plain decode, per-thread user/kernel over 3 s, 3 s after `generate:`
  appears in stderr, i.e. truly inside the generation window): NO thread is busy. 7 threads at 0.109-0.130 core each, all USER time; the main
  thread is below 0.021 core (it sleeps fine inside the blocking dspqueue_read). PC samples of the top thread (4001 samples, 8 s):
  88% ntdll!ZwWaitForAlertByThreadId < SleepConditionVariableSRW < ggml-cpu worker loop, 12% in the ggml-cpu worker poll loop
  (ggml_graph_compute_poll_for_work, shown as ggml_threadpool_new+0x8xx: no PDB). => the generation-time host CPU is the ggml-cpu threadpool
  workers spin-polling (default `--poll 50`) after the small per-token CPU graph, ~0.12 core x 7 workers = ~0.85 core.
- Why every earlier number said "one thread, 0.98 core, kernel": clover-run.ps1 / clover-port/thr.ps1 / my idle-ab.ps1 v1 put the window at
  [last CPU sample - 0.5 s - gen_s, last sample - 0.5 s], assuming teardown < 1 s. Teardown is ~50-55 s at ~1 core, all kernel time in one thread
  (releasing ~20 GB of HTP-mapped buffers): q1 base stderr last write 10:32:28 (perf print), next arm's stderr created 10:33:22; the Clover run
  bf16-fl-ctrl.cpu.csv decodes until t=70.5 s and the process lives to t=123 s at 0.94-1.01 core the whole time. So the window landed on teardown
  in every configuration, which is why WAIT=1/2, OPPOLL, -t 1/2 and --poll 0 "changed nothing". The port log's 06:35 conclusion (driver busy-wait in
  blocking dspqueue_read) is wrong: blocking dspqueue_read on Windows sleeps (main thread < 0.02 core).
- Run-to-run determinism: Qwen plain decode stdout differs between two runs of the SAME package and env (q1 base 9ECA6290 vs q2 base A0E9C96D),
  so HTP0 Qwen decode is not bit-reproducible run to run; output-hash identity can't be used as the gate for host-only changes here.
- Fixed `idle-ab.ps1`: window from the stderr timestamp of the perf line (process-relative), per-arm `extra` args, prints the top threads, kills the
  process after the perf print (saves ~50 s of lock time per arm) unless `keep`; teardown duration is reported for kept arms.
- 11:03 lock incident (mine): I aborted the q2 QoS A/B (its CPU window was the broken one) by killing its PowerShell at ~11:02:58. The lock owner
  was then dead and owner.txt older than 2 min, so bf16-bundleC took the lock at 11:02:59 while my llama-completion child (pid 23080, loading the
  model since 11:01:58) was still terminating (process teardown takes tens of seconds). The overlap fell in bf16-bundleC's model-load phase
  (first ~70 s), not in its generation window, but bf16-bundleC "fin" load time may be inflated. Lesson: wait for the child to exit before the lock
  holder dies. q2 data kept: base 28.96 tok/s (hash A0E9C96D, differs from q1 base 9ECA6290 -> nondeterminism confirmed).

## 2026-10-30 11:15 Re-analysis of every Clover run's saved CPU trace (clover-logs/*.cpu.csv + stderr timestamps)
Window = [t(decoded line) - gen_s, t(decoded line)] from the stderr timestamp vs. the old window anchored at process exit:
| run (HTP0 unless noted) | tok/s | gen s | host cores, real generation window | old metric | teardown after gen |
| --- | ---: | ---: | ---: | ---: | ---: |
| ab2b-2-topkc (port A/B, topk-c) | 5.12 | 50.2 | 6.48 | 0.99 | 51.8 s |
| ab2b-1-all (port all) | 5.21 | 49.3 | 6.55 | 0.99 | 54.6 s |
| ab3a-1-allpoll (OPPOLL=1, 400 ms spin) | 5.73 | 44.9 | 7.39 | 0.98 | 54.5 s |
| ab3a-3-allpoll05 (OPPOLL=1, 0.5 ms) | 5.51 | 46.6 | 6.90 | 0.98 | 54.4 s |
| rb-hl2-rs6 (rollback, d5) | 6.69 | 38.4 | 6.88 | 0.99 | 51.0 s |
| gca-1-gc1 | 5.16 | 49.8 | 6.53 | 0.99 | 54.8 s |
| bf16-fin-A1 (bf16 control) | 5.03 | 51.1 | 6.46 | 0.98 | 53.7 s |
| bf16-fin-B1 (bf16 agent final pkg) | 8.79 | 29.2 | **0.18** | 0.98 | 53.4 s |
| rb-cmsw2-rs6b (CPU device, d5) | 33.06 | 7.8 | 11.95 | 10.26 | 1.7 s |
=> Clover draft-self on HTP0 with the current op coverage really uses ~6.5 host cores during generation (ggml-cpu threadpool doing the
CPU-fallback splits, 8 threads), not 0.99. OPPOLL=1 adds ~0.9 core (the main thread spinning in read_noblock). The bf16 agent's final package
(CPU fallbacks moved to HTP) brings it to 0.18 cores at +75% tok/s: that, not the wait path, is the idle-host fix for Clover.
Every HTP0 process spends 51-55 s at ~1 core (kernel) in teardown after generation (buffer release at exit); CPU-device runs 2 s.

## 2026-10-30 11:40 Fix measured on Qwen3.6 plain decode: `--poll 0` (or `-t 1`) -> 0.05 host cores, same tok/s
`clover-idle/idle-ab.ps1 -Label q3` (fixed window from stderr timestamps; one lock hold 11:27-11:38; pkg-clover-idle-base; Qwen3.6-35B-A3B Q4_0,
HTP0, 400 tokens, -t 8 -tb 8, A B C A B order):
| arm | tok/s | host cores (gen window) | busiest thread | main thread | teardown |
| --- | ---: | ---: | --- | ---: | ---: |
| base (default `--poll 50`) | 28.79 | 0.903 | 0.151 user (7 ggml-cpu workers at 0.108-0.151 each) | 0.031 | 52.3 s (kept) |
| `--poll 0` | 28.74 | **0.047** | 0.036 (main) | 0.036 | killed after perf print |
| `-t 1 -tb 1` | 28.76 | **0.041** | 0.033 (main) | 0.033 | killed |
| base repeat | 28.30 | 0.974 | 0.145 user (7 workers 0.120-0.145) | 0.031 | killed |
| `--poll 0` repeat | 27.96 | **0.049** | 0.038 (main) | 0.038 | killed |
Means: default 28.55 tok/s / 0.94 cores; `--poll 0` 28.35 tok/s / 0.048 cores (-0.7%, inside the default arm's own 28.30-28.79 spread);
`-t 1` 28.76 / 0.041. Target (< 0.15 busy cores on Qwen plain decode) met with a run-time flag, no code change needed.
Mechanism: each decode token still runs a tiny CPU graph (ggml-cpu backend), which wakes the 8-thread pool; with the default poll level the 7
workers then spin in ggml_graph_compute_poll_for_work before sleeping on their condition variable (~4 ms each per 35 ms token).
Output hashes differ in every run (also between identical arms), so bit-identity is not a usable check on HTP0 Qwen decode; the flags only change
CPU-threadpool idling, not numerics. Clover KLD gate NOT run (no time left in the lock queue); the only code change left on branch clover-idle
(7e2ae41) initializes the FastRPC QoS latency field (was uninitialized stack data) - host-only, no numeric path.

## Summary / root cause / next steps
- Root cause of "one host thread at 0.99 core": neither the driver nor ggml spinning during generation. It is a measurement artifact: every HTP0
  process spends ~51-55 s after generation releasing its HTP-mapped buffers at ~1 core in kernel mode, and clover-run.ps1 / thr.ps1 / idle-ab v1
  anchored the "generation window" at process exit, so they measured teardown. Blocking dspqueue_read on Windows sleeps properly (main thread
  0.02-0.04 core); OPPOLL=1 is the only wait path that spins (+0.9 core in the Clover traces).
- Real generation-time host load: Qwen plain decode 0.90-0.97 cores = ggml-cpu workers polling (fix: `--poll 0` or `-t 1`, 0.04-0.05 cores, no
  tok/s change). Clover draft-self with today's CPU fallbacks: ~6.5 cores (real CPU split work); the bf16 agent's final package (bf16-fin-B1)
  measures 0.18 cores at 8.79 tok/s, i.e. moving fallbacks to HTP is the Clover idle fix.
- clover-run.ps1 fixed (window from the decoded/eval-time stderr timestamp, prints teardown). clover-port/thr.ps1 still has the old window
  (port agent's tool, not touched). Old "HOST_CPU busy cores during generation" numbers in other logs are teardown numbers.
- The experimental host wait modes (GGML_HEXAGON_WAIT=1 timer-sleep poll, =2 dspqueue packet callback) worked and cost no tok/s
  (28.1-28.6 vs 28.3) but were reverted (main thread already idle); they remain in history as 82514d2.
- Next: (1) make the product default `--poll 0` (or poll 0 whenever every layer is offloaded) for llama-server/cli on NPU; (2) measure Clover
  with `--poll 0` (expected to cut the polling share of 6.5 cores, may cost wake latency per CPU split) and with the bf16 package + `--poll 0`;
  (3) teardown: 52 s of a core to unmap ~20 GB at exit (model unload) - check whether buffers can be released in bulk / without per-page work;
  (4) run the Clover KLD gate on 7e2ae41 for completeness (expected identical; host-only change).
