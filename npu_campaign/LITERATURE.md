# Literature and upstream notes

Read 2026-09-26 while the signed Hexagon build was configuring. Citations are the public pages, not a re-run on this laptop.

## llama.cpp Hexagon on X2 Elite

- Matmul rework, 32x32 tiled weights, HMX-friendly layout. Drops v68/v69. Measured on X2 Elite. https://github.com/ggml-org/llama.cpp/pull/24954
- Flash-attention rework, merged 2026-07-01 as `13e6738`. X2 Elite tables are copied into `FINDINGS.md`. https://github.com/ggml-org/llama.cpp/pull/25085
- HMX gated-delta-net, merged 2026-09-21. This is the op for 30 of the 40 Qwen3.6 layers. Commit messages include a note that a related flash-attention change cuts DDR reads 20–30% during token generation. https://github.com/ggml-org/llama.cpp/pull/29199
- Session discovery. The author's 2026-08-27 note says Windows on X2 Elite does not support FastRPC domain discovery. Virtual sessions are still requested by name (`HTP0:0`, `HTP0:1`). https://github.com/ggml-org/llama.cpp/pull/27785
- Windows signing procedure matches what this campaign runs: Hexagon SDK 6.6.0.0, tools 19.0.07, `HEXAGON_HTP_CERT`, `inf2cat /os:10_25H2_ARM64`. https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/windows.md

## Bandwidth

- Hao, Wang, et al., "A Tale of Two Sides of Wafer", EuroSys 2026. arXiv:2509.23324. On Hexagon, HVX loads and `l2fetch` sit around 20–30 GB/s. A separate DMA engine reaches about 60 GB/s when the transfer is a large regular 1D or 2D block. TCM is about 8 MB. This is the paper that explains why the local shared-buffer ReLU stopped at 28 GB/s.
- HVX programmer's reference (v79, 80-N2040-61): DMA through the AXI slave is not coherent with the vector threads. The scalar core has to wait on the descriptor. A memcpy that forgets `dmwait` races the consumer. The local DMA skel already follows the llama.cpp descriptor pattern (`dmstart` / `dmlink` / `dmwait`).

## What not to treat as a measurement

- GitHub discussion 21702 (2026-04, Qwen3-4B on Snapdragon 8 Elite, 12 tok/s llama.cpp vs 26 tok/s Qualcomm's engine) is a real user report from before the matmul rework. The reply that llama.cpp "goes through the QNN SDK" does not match this backend. llama.cpp Hexagon opens `libggml-htp-v81.so` over FastRPC and runs its own HMX kernels.
- QHexRT's LFM2.5-230M numbers (prefill 12540 tok/s, decode 172 tok/s, CPU decode faster) are a 230M W8 model on a phone-class v81. They support the same split we already measured: NPU wins prefill, a bandwidth-bound decode can lose to a client that pulls more DRAM. They are not a 35B MoE result. https://www.runanywhere.ai/blog/qhexrt-npu-vs-cpu-lfm2.5-230m-qualcomm-hexagon
- Phone tables (8 Gen 3 / 8 Elite) where CPU decode beats NPU on sub-4B dense models are launch-overhead results. They predict that a tiny expert GEMV on this MoE will show a low implied GB/s even when a large DMA copy is fast.
