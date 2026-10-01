import os, re
W = r'C:\Users\hotschmoe\documents\llm_inference\llama.cpp-hexagon-qwen4\ggml\src\ggml-hexagon'
def rd(p):
    b = open(os.path.join(W, p), 'rb').read().decode()
    nl = '\r\n' if '\r\n' in b else '\n'
    return b.replace('\r\n', '\n'), nl
def wr(p, s, nl): open(os.path.join(W, p), 'wb').write(s.replace('\n', nl).encode())

# ---------------------------------------------------------------- 1. dot kernels for contiguous Q4_0 tiles (stride 576)
p = 'htp/hvx-mm-kernels-tiled.h'
s, nl = rd(p)
if 'tiled_vec_dot_q4_0c_32x1' not in s:
    marker = "static void tiled_vec_dot_q4_1_32x1("
    assert marker in s
    add = r'''// qwen4: Q4_0 tiles stored CONTIGUOUSLY (stride 576 B, no 640 B padding) so the weight DMA is one descriptor per tile row.
// Even tiles are 128 B aligned, odd tiles start 64 B past an aligned address: they are loaded with unaligned vector loads.
// Same arithmetic as tiled_vec_dot_q4_0_32x1/32x2 (bit-identical).
static inline void q4c_load_tile(HVX_Vector * w, const uint8_t * tile, uint32_t kt) {
    if ((kt & 1) == 0) {
        const HVX_Vector * p = (const HVX_Vector *) tile;
        w[0] = p[0]; w[1] = p[1]; w[2] = p[2]; w[3] = p[3]; w[4] = p[4];
    } else {
        w[0] = hvx_vmemu(tile);
        w[1] = hvx_vmemu(tile + 128);
        w[2] = hvx_vmemu(tile + 256);
        w[3] = hvx_vmemu(tile + 384);
        w[4] = hvx_vmemu(tile + 512);
    }
}

static void tiled_vec_dot_q4_0c_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows, const float * restrict sz) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();
    HVX_Vector i8 = Q6_Vb_vsplat_R(8);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        HVX_Vector w[5];
        q4c_load_tile(w, tile_ptr + (size_t) kt * 576, kt);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1152);

        HVX_Vector v_sum = accum_4bit_32x1(w, v_act, i8);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_w = w[4];
        HVX_Vector v_scale_a = v_act[8];
        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a);
        HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    if (sz) {
        hvx_vec_store_u(s, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } else {
        hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
    }
}

static void tiled_vec_dot_q4_0c_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows, const float * restrict sz0, const float * restrict sz1) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();
    HVX_Vector i8 = Q6_Vb_vsplat_R(8);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        HVX_Vector w[5];
        q4c_load_tile(w, tile_ptr + (size_t) kt * 576, kt);
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1152);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair v_sums = accum_4bit_32x2(w, v_act0, v_act1, i8);
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_w = w[4];
        HVX_Vector v_scale_a_c0 = v_act0[8];
        HVX_Vector v_scale_a_c1 = v_act1[8];

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a_c0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a_c1);

        HVX_Vector v_sum_scaled_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    if (sz0) {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
    } else {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);
    }
    if (sz1) {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
    } else {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);
    }
}

'''
    s = s.replace(marker, add + marker, 1)
    wr(p, s, nl)

# ---------------------------------------------------------------- 2. flag
p = 'htp/htp-ops.h'
s, nl = rd(p)
if 'HTP_OPFLAGS_CONTIG' not in s:
    a = "    HTP_OPFLAGS_MV_XPIPE "
    assert a in s
    s = s.replace(a, "    HTP_OPFLAGS_CONTIG         = (1U << 18), // qwen4: weight tile rows move with ONE contiguous DMA descriptor (Q4_0 tiles stored at stride 576, 64-row 2D descriptors cost ~12 ns/row)\n" + a, 1)
    wr(p, s, nl)

# ---------------------------------------------------------------- 3. matmul-ops.c
p = 'htp/matmul-ops.c'
s, nl = rd(p)

# 3a. MM_DMA_ARGS: contiguous when var1 spoof or (CONTIG flag and no padding)
old = [l for l in s.split('\n') if l.startswith('#define MM_DMA_ARGS(op, at, ts, nk)')]
assert len(old) == 1
new = ("#define MM_DMA_CONTIG(op, at, ts) (((op)->flags & HTP_OPFLAGS_SPOOF_DMAVAR1) || (((op)->flags & HTP_OPFLAGS_CONTIG) && (at) == (ts)))\n"
       "#define MM_DMA_ARGS(op, at, ts, nk) (MM_DMA_CONTIG(op, at, ts) ? (size_t)(ts)*(nk) : (size_t)(at)), (MM_DMA_CONTIG(op, at, ts) ? (size_t)(ts)*(nk) : (size_t)(ts)), (MM_DMA_CONTIG(op, at, ts) ? (size_t)(ts)*(nk) : (size_t)(ts)), (MM_DMA_CONTIG(op, at, ts) ? (size_t)1 : (size_t)(nk))")
s = s.replace(old[0], new, 1)

# 3b. tile alignment hook used by the macros (default: 128 B aligned; q4_0c overrides to the raw tile size)
s = s.replace("// Specialized repacked matmul macros", "#define MM_TILE_ALIGN(ts) hex_align_up((ts), 128)\n\n// Specialized repacked matmul macros", 1)
lines = s.split('\n')
cnt = 0
def region(start_pat):
    i = [k for k, l in enumerate(lines) if start_pat in l][0]
    j = i + 1
    while not (lines[j].startswith('#define') or lines[j].startswith('MAT')):
        j += 1
    return i, j
for pat in ['#define MATMUL_2D_REPACKED_IMPL', '#define MATVEC_2D_REPACKED_IMPL', '#define MATMUL_NX_2D_REPACKED_IMPL']:
    i, j = region(pat)
    for k in range(i, j):
        if 'const uint32_t aligned_tile_size = hex_align_up(tile_size, 128);' in lines[k]:
            lines[k] = lines[k].replace('hex_align_up(tile_size, 128)', 'MM_TILE_ALIGN(tile_size)'); cnt += 1
assert cnt == 3, cnt
s = '\n'.join(lines)
# the MATMUL_2D macro pushes use the raw argument list; convert to MM_DMA_ARGS there too
lines = s.split('\n')
i, j = region('#define MATMUL_2D_REPACKED_IMPL')
c2 = 0
for k in range(i, j):
    if 'aligned_tile_size, tile_size, tile_size, n_k_tiles_a)' in lines[k]:
        lines[k] = lines[k].replace('aligned_tile_size, tile_size, tile_size, n_k_tiles_a)', 'MM_DMA_ARGS(octx, aligned_tile_size, tile_size, n_k_tiles_a))'); c2 += 1
assert c2 == 2, c2
s = '\n'.join(lines)

# 3c. q4_0c instantiations after the existing q4_0 ones
a = "MATMUL_2D_REPACKED_IMPL(q4_0,       576,  tiled_vec_dot_q4_0_32x2,  tiled_vec_dot_q4_0_32x1)\n"
assert a in s
s = s.replace(a, a, 1)
# append c-variant instantiations right before hvx_mm_transfer_src1_dma (all three macros are defined by then? NO: only 2D here; the others later)
open(os.path.join(W, 'qwen4-q4c-stage.txt'), 'w').write('ok')
wr(p, s, nl)
print('stage1 ok')
