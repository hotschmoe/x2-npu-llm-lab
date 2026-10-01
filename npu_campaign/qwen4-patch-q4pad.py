import os, re
W = r'C:\Users\hotschmoe\documents\llm_inference\llama.cpp-hexagon-qwen4\ggml\src\ggml-hexagon'
def rd(p):
    b = open(os.path.join(W, p), 'rb').read().decode()
    nl = '\r\n' if '\r\n' in b else '\n'
    return b.replace('\r\n', '\n'), nl
def wr(p, s, nl): open(os.path.join(W, p), 'wb').write(s.replace('\n', nl).encode())

# 1. tile size constants: Q4_0 / IQ4_NL tiles are stored 640 B (576 + 64 B pad) in DDR so the weight tile row is ONE contiguous 1D DMA with the original (aligned) dot kernels
p = 'htp/matmul-ops.h'
s, nl = rd(p)
assert '#define HTP_MM_WEIGHT_TILE_SIZE_Q4_0   576' in s
s = s.replace('#define HTP_MM_WEIGHT_TILE_SIZE_Q4_0   576', '#define HTP_MM_WEIGHT_TILE_SIZE_Q4_0   640   // qwen4: 576 B of quants+scales padded to 640 B in DDR (one contiguous 1D DMA per tile row, no VTCM re-padding)', 1)
s = s.replace('#define HTP_MM_WEIGHT_TILE_SIZE_IQ4_NL 576', '#define HTP_MM_WEIGHT_TILE_SIZE_IQ4_NL 640', 1)
wr(p, s, nl)

# 2. DSP instantiation literals
p = 'htp/matmul-ops.c'
s, nl = rd(p)
n = 0
for name in ('q4_0', 'iq4nl'):
    for macro in ('MATMUL_2D_REPACKED_IMPL', 'MATVEC_2D_REPACKED_IMPL', 'MATMUL_NX_2D_REPACKED_IMPL', 'MATMUL_4D_REPACKED_IMPL'):
        pat = re.compile(r'^(' + macro + r'\(' + name + r',\s*)576(,)', re.M)
        s, k = pat.subn(r'\1HTP_MM_WEIGHT_TILE_SIZE_Q4_0\2', s)
        n += k
print('instantiations patched', n)
assert n == 8, n
# the q4c variants are no longer used (the padded layout makes the original kernels contiguous-DMA friendly): neutralise the dispatch
s = s.replace("matmul_job_func = q4c ? hvx_mm_2d_repacked_q4_0c : hvx_mm_2d_repacked_q4_0;", "matmul_job_func = hvx_mm_2d_repacked_q4_0;")
s = s.replace("matmul_job_func = q4c ? hvx_mv_2d_repacked_q4_0c : hvx_mv_2d_repacked_q4_0;", "matmul_job_func = hvx_mv_2d_repacked_q4_0;")
s = s.replace("matmul_job_func = q4c ? hvx_mm_nx_2d_repacked_q4_0c : hvx_mm_nx_2d_repacked_q4_0;", "matmul_job_func = hvx_mm_nx_2d_repacked_q4_0;")
s = s.replace("    if (w->type == HTP_TYPE_Q4_0) { mmctx->vec_dot_32x1 = tiled_vec_dot_q4_0c_32x1; }\n", "    (void) mmctx; (void) w;   // qwen4: Q4_0 tiles are padded to 640 B in DDR now: the original dot kernel is used with the contiguous DMA\n")
wr(p, s, nl)

p = 'htp/mv-id-xpipe.h'
s, nl = rd(p)
a = "    if ((octx->flags & HTP_OPFLAGS_CONTIG) && wtype == HTP_TYPE_Q4_0 && mmctx->vec_dot_32x1 == tiled_vec_dot_q4_0c_32x1) return htp_mm_get_weight_tile_size(wtype);\n"
assert a in s
s = s.replace(a, "")
wr(p, s, nl)

# 3. host: alloc/nb row size + repack padding
p = 'ggml-hexagon.cpp'
s, nl = rd(p)
a = "    if (type == GGML_TYPE_Q4_K) {\n        return (size_t) (ne0 / 32) * (HTP_MM_WEIGHT_TILE_SIZE_Q4_1 / 32);\n    }\n"
assert a in s
s = s.replace(a, a + "    if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_IQ4_NL) {   // qwen4: tiles padded to 640 B\n        return (size_t) (ne0 / 32) * (HTP_MM_WEIGHT_TILE_SIZE_Q4_0 / 32);\n    }\n", 1)
# zero the pad of every repacked Q4_0 tile
a = "                uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;\n\n                uint8_t tile_quants[32][32];"
assert s.count(a) == 1
s = s.replace(a, "                uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;\n                memset(tile_dst + 576, 0, tile_size - 576);   // qwen4: 64 B tile padding\n\n                uint8_t tile_quants[32][32];", 1)
wr(p, s, nl)
print('q4pad ok')
