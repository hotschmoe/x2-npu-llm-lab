"""trace agent: write a copy of a GGUF where the dense (non-expert) mul_mat weights are dequantized to F32.

The weight VALUES are unchanged (Q4_0 -> F32 is exact); only the CPU arithmetic changes: ggml-cpu multiplies Q4_0
weights with activations quantized to Q8_0 (vec_dot_type), but F32 weights with F32 activations. Expert tensors,
token_embd and everything else are copied byte for byte. Used as a near-exact CPU reference for HTP KLD.
Usage: python make_f32dense_gguf.py in.gguf out.gguf
"""
import re
import sys

import numpy as np

sys.path.insert(0, r'C:\Users\hotschmoe\Documents\llm_inference\llama.cpp-hexagon-clover-trace\gguf-py')
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

DENSE = re.compile(r'^(blk\.\d+\.(attn_qkv|attn_gate|ssm_alpha|ssm_beta|ssm_out|attn_q|attn_k|attn_v|attn_output|'
                   r'ffn_gate_shexp|ffn_up_shexp|ffn_down_shexp)\.weight|output\.weight)$')

src, dst = sys.argv[1], sys.argv[2]
r = gguf.GGUFReader(src)
arch = r.fields[gguf.Keys.General.ARCHITECTURE].contents()
w = gguf.GGUFWriter(dst, arch=arch, endianess=r.endianess)
for field in r.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
        continue
    vt = field.types[0]
    st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(field.name, field.contents(), vt, sub_type=st)
conv = []
for t in r.tensors:
    if DENSE.match(t.name) and t.tensor_type != gguf.GGMLQuantizationType.F32:
        shape = (int(t.shape[1]), int(t.shape[0]))
        w.add_tensor_info(t.name, shape, np.dtype(np.float32), shape[0] * shape[1] * 4, gguf.GGMLQuantizationType.F32)
        conv.append(t.name)
    else:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
print(f'converting {len(conv)} tensors to F32')
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for t in r.tensors:
    if t.name in conv:
        a = dequantize(t.data, t.tensor_type).astype(np.float32).reshape(int(t.shape[1]), int(t.shape[0]))
        w.write_tensor_data(a, tensor_endianess=r.endianess)
    else:
        w.write_tensor_data(t.data, tensor_endianess=r.endianess)
w.close()
print('done', dst)
