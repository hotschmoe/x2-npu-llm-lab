# NPU MoE: 20 tok/s

Date: 2026-09-25

Machine: ASUS Zenbook A16, Snapdragon X2 Elite Extreme, Hexagon v81, one NSP,
8 MB VTCM. Nominal DRAM pin rate 228 GB/s. Measured NPU paths so far: copied
QNN client buffers about 15 GB/s, `rpcmem` shared buffers about 28 GB/s.
A DMA-engine memcpy is built and not yet loaded (`AEE_EUNABLETOLOAD` until the
signed catalog is accepted after reboot).

llama.cpp is the measuring instrument. The native NVFP4 runtime stays the
serving path. A GGUF copy exists so the Hexagon backend can run; it is not a
second resident copy of the production checkpoint.

## Goal

Run the Ornith-class 35B mixture-of-experts model, 4-bit, entirely on the NPU,
at **20 tok/s or better** end to end on one warm decode stream.

"Entirely on the NPU" means router, expert GEMVs, shared expert, attention or
gated-delta, norms, and the output projection execute as Hexagon ops. Sampling
may stay on the CPU. The draft that produces the extra accepted tokens runs on
the NPU as well.

The 4-bit formats the fetched Hexagon backend can execute are Q4_0, Q4_K,
IQ4_NL, and MXFP4. There is no 3-bit op in that backend. A 3-bit model is a
later kernel, not the first target.

The model is Qwen3.6-35B-A3B, architecture `qwen35moe`: 40 layers, 30 gated
delta-net plus 10 full attention, 256 experts, top-8, about 3B active. That is
the architecture the Hexagon gated-delta and `MUL_MAT_ID` work was measured on,
including on X2 Elite. First file is
[bartowski/Qwen_Qwen3.6-35B-A3B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF)
`Q4_0` (20.8 GB). `IQ4_NL` (20.8 GB) is the alternate. Both are in the Hexagon
repack list. `Q4_K_M` (22.3 GB) and
[ggml-org/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF)
`Q4_K_M` (20.4 GB) are the quality check after Q4_0 runs. The draft head is the
1.06 GB `MTP Q4_0` in the ggml-org repo, used with llama.cpp `draft-mtp` and a
proposal length of 3. MXFP4_MOE (about 18.4–21 GB) is the smaller 4-bit to try
once Q4_0 has a trace, because MXFP4 is also a native Hexagon type.

Ornith-1.5-35B-A3B is the same `qwen35moe` graph. Its NVFP4 checkpoints,
including the GGUF NVFP4 and the MTP NVFP4, are the Adreno path. Upstream
Hexagon does not execute NVFP4. An Ornith Q4_K_M exists and would run, and it
does not get to 20 tok/s any sooner than Qwen3.6.

Gemma 4 has no 30B-A4B. The MoE in this size is
[gemma-4-26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it), about 3.8B
active. The QAT `Q4_0` file is about 14.4 GB and is a supported type. It is the
second model, after Qwen3.6, because its active set is larger and the Hexagon
MoE work was measured on Qwen, not on Gemma 4's expert layout.

## Why 20 tok/s is a latency goal

One token of this MoE activates about 2.3 GB. A plain decode that rereads those
bytes every token needs about 46 GB/s to hit 20 tok/s. The shared-buffer path
we measured is 28 GB/s, which is about 13 tok/s if every token pays for its own
read.

Accepting 1 to 2 draft tokens per main forward changes the denominator. The
main model reads the active weights once and verifies a short run:

| Draft tokens kept per main forward | Tokens produced per weight read | tok/s at 28 GB/s | tok/s at 45 GB/s |
|---:|---:|---:|---:|
| 0 | 1 | 12 | 20 |
| 1 | 2 | 24 | 39 |
| 2 | 3 | 37 | 59 |

Twenty tok/s is the row where one extra draft token is kept on the 28 GB/s
path, or where plain decode meets a bridge near 45 GB/s. Both are in scope.
The trace decides which one is real. A step whose implied GB/s is far below
the bridge is waiting, and a faster copy will not move it.

## What every step records

Same grain as the Adreno command trace in `native_nvfp4/trace_utils.py`: one
row per movement of bytes, then a rollup.

Outer row, one per graph op (`GGML_HEXAGON_PROFILE=1`):

- op name, tensor names, shape, dtype
- bytes read, bytes written, bytes that stayed in VTCM
- device microseconds and cycles
- implied GB/s = (bytes read + bytes written) / seconds / 1e9
- gap from the previous op's end to this op's start

Inner row, one per stage inside that op (`GGML_HEXAGON_PROFILE=3`):

- `DMA`, `HVX_A_QUANT`, `HVX_A_PREP`, `HVX_W_DEQUANT`, `HVX_W_PREP`,
  `HMX_COMP`, `HVX_O_PROC`, `L2FLUSH`
- cycles, and microseconds once the DSP clock in the profile line is known
- bytes that stage moved, when the op records them
- implied GB/s for that stage

A stage with a high implied GB/s is sitting on the bridge. A stage with a
low implied GB/s, or a gap larger than the stage itself, is latency: launch,
a tiny expert tile, a descriptor setup, a drain. Those get fused or deleted.
They do not get a wider DMA.

`GGML_HEXAGON_OPSTAGE=0x1` is the control. It queues the op and does not run
DMA or compute. The difference between `0x1` and `0x3` is device time.

## Steps

### 0. Boot

Reboot. Confirm `bcdedit /enum` shows `testsigning Yes` on the current entry.
The certificate `CN=GGML.HTP.v1`, thumbprint
`690E88679A281C052CD18243883064D94E256296`, is already in the machine Trusted
Root and Trusted Publishers stores. The `.pfx` is
`C:\Users\hotschmoe\Certs\ggml-htp-v1.pfx`.

### 1. llama.cpp, signed, one small model

Build the Hexagon backend with that certificate so `libggml-htp-v81.so` and
`libggml-htp.cat` land in the same directory. Point `ADSP_LIBRARY_PATH` at
that directory.

Smoke test is a small Q4_0 model that fits in one session, not the 35B.
`GGML_HEXAGON_VERBOSE=1` must show the matmul on `HTP0`. A load that returns
`0x80000406` is still the catalog, not the model.

Then sign `libdma_bw_skel.so` with the same `.pfx` and run
`island_bw/dma_bw_host.exe`. That number is the DMA-engine ceiling. It is one
input to the map, not a substitute for the per-op rows.

### 2. One MoE layer, fully traced

Convert one layer, then the full model, to Q4_0 or Q4_K. Run a single decode
token with `GGML_HEXAGON_PROFILE=1` and again with `GGML_HEXAGON_PROFILE=3`.

The layer row we already know from Adreno is the checklist: router, top-8
gather, expert gate, expert up, expert down, shared expert, residual, norm,
attention or gated-delta, conv state. Each of those becomes a profile line
with microseconds and implied GB/s. Roll the lines up the way
`trace_utils.py` rolls OpenCL events: time fraction, byte fraction, and the
ops whose implied GB/s is under half of the DMA ceiling measured in step 1.

One Hexagon domain maps about 3.5 GB. A 35B Q4 file is about 18–20 GB, so the
full model needs several sessions (`GGML_HEXAGON_NDEV`). Prove one session on
one layer before asking the driver for six.

### 3. Our ops, only where the map says

Changes wait on the table from step 2. The allowed kinds of change:

- fuse a run of ops whose gaps dominate their device time
- widen a DMA that the inner trace shows is short of the step-1 ceiling
- replace a HVX weight prep that is slower than the bytes require
- keep an op on the CPU when its NPU line is launch-bound and the CPU line is
  shorter end to end

A new kernel ships with the same row format: bytes in, bytes out, device
microseconds, implied GB/s, and the inner stage split. A kernel that raises
a microbenchmark and does not move the layer's slowest row is not a win.

Native NVFP4 stays on Adreno until a Hexagon op beats the traced Q4 line for
that same shape. The QHPI `NvFp4Linear` contract is the candidate, not the
starting point.

### 4. Draft sized for 1–2 accepted tokens

The draft is a small NPU model, hundreds of megabytes, not a second copy of
the 35B. It runs on the same Hexagon, then the main model verifies the run in
one weight read.

Target, measured over a few hundred tokens of real text:

- mean draft tokens kept per main forward: 1 to 2
- tokens produced per main forward: 2 to 3, counting the token the main model
  would have emitted anyway
- main-forward device time within 20% of the single-token trace, which is how
  we know the verify did not reread the experts

If the mean kept tokens stay below 1, fix the draft or the proposal length
before touching the main kernels again. If the main forward gets slower in
proportion to the proposal length, the verify is rereading weights and the
bandwidth math above does not apply.

### 5. Full model, warm stream

Resident 35B across however many sessions the driver actually allows. One
warm stream, batch 1. Record:

- tok/s over at least a few hundred tokens
- mean accepted draft tokens
- the same per-op table for one decoded token, so a regression has a row
- wall power if a meter is available; tok/s per watt is the laptop number,
  and it is recorded beside tok/s rather than instead of it

The goal is met when that stream holds 20 tok/s, the per-op table still
accounts for the token, and the experts stayed on the NPU.

## Out of scope until the table exists

A wider DMA memcpy, a custom NVFP4 kernel, and a bigger draft are all
unscoped until step 2 names the row they would move. UI-lag and heat are
reasons the NPU is the device. They are not a separate optimization pass.
