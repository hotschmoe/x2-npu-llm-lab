# SERVE_LOG (agent `serve`, append-only)

## 2026-09-30 23:10-23:35
- Read coordination/overnight docs. llama-server (pkg-wos) has NO per-request speculation toggle: slot.can_speculate() = server-level `spec` object; task params only echo speculative.types. So `-Mode both` is not possible; it is an alias of `code`.
- Wrote serve\start-qwen.ps1/.cmd (detached hidden supervisor holds NPU lock for the server lifetime), stop-qwen, health.ps1, decisions_shim.py.
- Blocker: two lockless llama-cli (qwen4 pkg-qwen4-s12/s13, started 23:17, 70 s CPU) held ~40 GB (Available MBytes ~100). Not mine, not killed. Waiting for them.
- 23:45 fake-llama unit test of shim plumbing OK (serve\fake_llama.py). npu-lock.ps1 does not load under Windows PowerShell 5.1 ([NpuLockNative] type not found) -> locking is now serve\lockhold.py (lockrun protocol, 20 ms polling, owner-checked release), supervisor = that python process.
- 00:00 chat mode: 36.0 tok/s decode; shim test 12/12 PASS, 0.45-0.7 s/question, cache_n=0 always (hybrid model has no prefix reuse for the shim prompts). Found: llama-server process lingers ~15 s after kill holding 21 GB -> supervisor now waits for it to be gone before releasing the lock (a previous stop released the lock early and decode2 started during teardown).
- 00:08 code mode: 37.3 / 37.2 tok/s, acceptance 124/129 and 54/55. 00:16 smoke.ps1 PASS (3 PI cases), decode 33.5 avg, prefill 223 avg (metrics average over short turns).
- No per-request speculation toggle in this llama-server; -Mode both == code.
