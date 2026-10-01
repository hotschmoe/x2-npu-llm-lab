import os
W = r'C:\Users\hotschmoe\documents\llm_inference\llama.cpp-hexagon-qwen4\ggml\src\ggml-hexagon'
def rd(p):
    b = open(os.path.join(W, p), 'rb').read().decode()
    nl = '\r\n' if '\r\n' in b else '\n'
    return b.replace('\r\n', '\n'), nl
def wr(p, s, nl): open(os.path.join(W, p), 'wb').write(s.replace('\n', nl).encode())

p = 'htp/hvx-mm-kernels-tiled.h'
s, nl = rd(p)
assert 'tiled_vec_dot_q8_0c_32x1' not in s
marker = "static inline HVX_Vector q5_k_weight_scale_product("
assert marker in s
add = r'''// qwen4: Q8_0 tiles stored contiguously (stride 1088 B); odd tiles are 64 B misaligned (unaligned vector loads). Bit-identical to the padded kernels.
static inline void q8c_load_tile(HVX_Vector * w, const uint8_t * tile, uint32_t kt) {
    if ((kt & 1) == 0) {
        const HVX_Vector * p = (const HVX_Vector *) tile;
        w[0] = p[0]; w[1] = p[1]; w[2] = p[2]; w[3] = p[3]; w[4] = p[4]; w[5] = p[5]; w[6] = p[6]; w[7] = p[7]; w[8] = p[8];
    } else {
        w[0] = hvx_vmemu(tile);       w[1] = hvx_vmemu(tile + 128); w[2] = hvx_vmemu(tile + 256); w[3] = hvx_vmemu(tile + 384);
        w[4] = hvx_vmemu(tile + 512); w[5] = hvx_vmemu(tile + 640); w[6] = hvx_vmemu(tile + 768); w[7] = hvx_vmemu(tile + 896);
        w[8] = hvx_vmemu(tile + 1024);
    }
}

static void tiled_vec_dot_q8_0c_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows, const float * restrict sz) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        HVX_Vector w[9];
        q8c_load_tile(w, tile_ptr + (size_t) kt * 1088, kt);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1152);

        HVX_Vector v_sum = accum_q8_0_32x1(w, v_act);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_w = w[8];
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

static void tiled_vec_dot_q8_0c_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows, const float * restrict sz0, const float * restrict sz1) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        HVX_Vector w[9];
        q8c_load_tile(w, tile_ptr + (size_t) kt * 1088, kt);
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1152);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair v_sums = accum_q8_0_32x2(w, v_act0, v_act1);
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_w = w[8];
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

p = 'htp/matmul-ops.c'
s, nl = rd(p)
assert 'q8_0c' not in s
def inst(s, last_line, text):
    assert last_line in s, last_line
    return s.replace(last_line, last_line + '\n' + text, 1)
over = "#undef MM_TILE_ALIGN\n#define MM_TILE_ALIGN(ts) (ts)\n"
back = "#undef MM_TILE_ALIGN\n#define MM_TILE_ALIGN(ts) hex_align_up((ts), 128)"
# place after the q4_0c instantiations
s = inst(s, "MATMUL_2D_REPACKED_IMPL(q4_0c,      576,  tiled_vec_dot_q4_0c_32x2, tiled_vec_dot_q4_0c_32x1)\n" + back,
         over + "MATMUL_2D_REPACKED_IMPL(q8_0c,      1088, tiled_vec_dot_q8_0c_32x2, tiled_vec_dot_q8_0c_32x1)\n" + back)
s = inst(s, "MATVEC_2D_REPACKED_IMPL(q4_0c,      576,  tiled_vec_dot_q4_0c_32x1)\n" + back,
         over + "MATVEC_2D_REPACKED_IMPL(q8_0c,      1088, tiled_vec_dot_q8_0c_32x1)\n" + back)
s = inst(s, "MATMUL_NX_2D_REPACKED_IMPL(q4_0c,   576,  tiled_vec_dot_q4_0c_32x2, tiled_vec_dot_q4_0c_32x1)\n" + back,
         over + "MATMUL_NX_2D_REPACKED_IMPL(q8_0c,   1088, tiled_vec_dot_q8_0c_32x2, tiled_vec_dot_q8_0c_32x1)\n" + back)
a = "                case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mm_2d_repacked_q8_0;   break;"
assert a in s
s = s.replace(a, "                case HTP_TYPE_Q8_0:   matmul_job_func = q8c ? hvx_mm_2d_repacked_q8_0c : hvx_mm_2d_repacked_q8_0;   break;", 1)
a = "                case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mv_2d_repacked_q8_0;   break;"
assert a in s
s = s.replace(a, "                case HTP_TYPE_Q8_0:   matmul_job_func = q8c ? hvx_mv_2d_repacked_q8_0c : hvx_mv_2d_repacked_q8_0;   break;", 1)
a = "            case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mm_nx_2d_repacked_q8_0;   break;"
assert a in s
s = s.replace(a, "            case HTP_TYPE_Q8_0:   matmul_job_func = q8c ? hvx_mm_nx_2d_repacked_q8_0c : hvx_mm_nx_2d_repacked_q8_0;   break;", 1)
Q8C = "    const bool q8c = ((octx->flags & HTP_OPFLAGS_CONTIG) != 0) && src0->type == HTP_TYPE_Q8_0 && octx->ctx->mdev.count <= 1;\n"
n = s.count("    const bool q4c = ((octx->flags & HTP_OPFLAGS_CONTIG) != 0) && src0->type == HTP_TYPE_Q4_0 && octx->ctx->mdev.count <= 1;\n")
assert n == 2, n
s = s.replace("    const bool q4c = ((octx->flags & HTP_OPFLAGS_CONTIG) != 0) && src0->type == HTP_TYPE_Q4_0 && octx->ctx->mdev.count <= 1;\n",
              "    const bool q4c = ((octx->flags & HTP_OPFLAGS_CONTIG) != 0) && src0->type == HTP_TYPE_Q4_0 && octx->ctx->mdev.count <= 1;\n" + Q8C)
wr(p, s, nl)
print('q8c ok')
