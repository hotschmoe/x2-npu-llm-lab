"""Patch 4: Q4_0 tile dequant via vlut16 (copy of the MXFP4 kernel structure), enabled with GGML_HEXAGON_MMID_PIPE.
Bit-identical expected: LUT yields exact fp16 ints -8..7, then the same qf16 scale multiply as before."""
import os, re
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'llama.cpp-hexagon', 'ggml', 'src', 'ggml-hexagon', 'htp')
p = os.path.join(root, 'hmx-mm-kernels-tiled.h')
s = open(p, newline='').read()
crlf = '\r\n' in s
s = s.replace('\r\n', '\n')
if 'q4_0_lut' in s:
    print('already applied'); raise SystemExit

a = s.index('static void dequantize_tiled_weight_to_fp16_task_mxfp4(')
b = s.index('\n}\n', a) + 3
fn = s[a:b]
fn = fn.replace('dequantize_tiled_weight_to_fp16_task_mxfp4(', 'dequantize_tiled_weight_to_fp16_task_q4_0_lut(', 1)
fn = fn.replace('hvx_vmem(mxfp4_to_fp16_lut)', 'hvx_vmem(q4_0_to_fp16_lut)')
old_scale = re.search(r'        HVX_Vector v = hvx_vmem\(tile_src \+ 512\);\n.*?HVX_Vector v_scale_duplicated = Q6_V_lo_W\(Q6_W_vshuff_VVR\(vh, vh, -2\)\);\n', fn, re.S)
assert old_scale, 'scale block not found'
new_scale = ('        HVX_Vector v_sc = hvx_vmem(tile_src + 512);\n'
             '        HVX_Vector v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));\n')
fn = fn.replace(old_scale.group(0), new_scale, 1)

block = '''#ifdef GGML_HEXAGON_MMID_PIPE
// Q4_0 nibble -> fp16 LUT: value (i - 8) at even halfword 2*i (odd halfwords zero), same convention as the MXFP4 LUT.
static const __fp16 q4_0_to_fp16_lut[64] __attribute__((aligned(VLEN))) = {
    -8, 0, -7, 0, -6, 0, -5, 0, -4, 0, -3, 0, -2, 0, -1, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0,
};

''' + fn + '''// Route the Q4_0 worker loop (DEQUANTIZE_WORKER_LOOP_IMPL(q4_0) pastes this name) to the LUT kernel.
#define dequantize_tiled_weight_to_fp16_task_q4_0 dequantize_tiled_weight_to_fp16_task_q4_0_lut
#endif  // GGML_HEXAGON_MMID_PIPE
'''
s = s[:b] + '\n' + block + s[b:]
if crlf: s = s.replace('\n', '\r\n')
open(p, 'w', newline='').write(s)
print('patched q4 lut')
