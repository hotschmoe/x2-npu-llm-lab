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
