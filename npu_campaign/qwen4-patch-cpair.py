import os
W = r'C:\Users\hotschmoe\documents\llm_inference\llama.cpp-hexagon-qwen4\ggml\src\ggml-hexagon'
def rd(p):
    b = open(os.path.join(W, p), 'rb').read().decode()
    nl = '\r\n' if '\r\n' in b else '\n'
    return b.replace('\r\n', '\n'), nl
def wr(p, s, nl): open(os.path.join(W, p), 'wb').write(s.replace('\n', nl).encode())

p = 'htp/hvx-mm-kernels-tiled.h'
s, nl = rd(p)

# ---------------------------------------------------------------- Q4_0 contiguous (stride 576): process tiles in PAIRS (no branch, no unaligned loads: valign on aligned loads)
i0 = s.index('// qwen4: Q4_0 tiles stored CONTIGUOUSLY')
i1 = s.index('static void tiled_vec_dot_q4_1_32x1(')
q4 = r'''// qwen4: Q4_0 tiles stored CONTIGUOUSLY (stride 576 B, no 640 B padding) so the weight DMA is one descriptor per tile row.
// Two consecutive tiles are 1152 B = 9 aligned vectors p[0..8]: the even tile is p[0..3] + scale in the low half of p[4]; the odd tile starts 64 B into p[4]
// and is rebuilt with valign (4 data vectors + the scale from the high half of p[8]). Tiles are accumulated in ascending order with the same float
// operations as tiled_vec_dot_q4_0_32x1/32x2 (bit-identical).
static inline HVX_Vector q4c_tile_x1(const HVX_Vector * w, const HVX_Vector * v_act, HVX_Vector i8, HVX_Vector acc) {
    HVX_Vector v_sum = accum_4bit_32x1(w, v_act, i8);
    HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);
    HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(w[4], v_act[8]);
    HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);
    return hvx_vec_add_f32_f32(acc, v_sum_scaled);
}

static inline void q4c_tile_x2(const HVX_Vector * w, const HVX_Vector * v_act0, const HVX_Vector * v_act1, HVX_Vector i8, HVX_Vector * acc0, HVX_Vector * acc1) {
    HVX_VectorPair v_sums = accum_4bit_32x2(w, v_act0, v_act1, i8);
    HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(v_sums));
    HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(v_sums));
    HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(w[4], v_act0[8]);
    HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(w[4], v_act1[8]);
    *acc0 = hvx_vec_add_f32_f32(*acc0, hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0));
    *acc1 = hvx_vec_add_f32_f32(*acc1, hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1));
}

#define Q4C_ODD_FROM_PAIR(wo, p)                                      \
    HVX_Vector wo[5] = { Q6_V_valign_VVR((p)[5], (p)[4], 64), Q6_V_valign_VVR((p)[6], (p)[5], 64), \
                         Q6_V_valign_VVR((p)[7], (p)[6], 64), Q6_V_valign_VVR((p)[8], (p)[7], 64), \
                         Q6_V_vror_VR((p)[8], 64) }

static void tiled_vec_dot_q4_0c_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows, const float * restrict sz) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();
    HVX_Vector i8 = Q6_Vb_vsplat_R(8);

    const uint32_t n_k_tiles = n / 32;
    uint32_t kt = 0;
    for (; kt + 1 < n_k_tiles; kt += 2) {
        const HVX_Vector * restrict p = (const HVX_Vector *) (tile_ptr + (size_t) kt * 576);
        HVX_Vector we[5] = { p[0], p[1], p[2], p[3], p[4] };
        Q4C_ODD_FROM_PAIR(wo, p);
        v_sum_float = q4c_tile_x1(we, (const HVX_Vector *) (y_q + kt * 1152), i8, v_sum_float);
        v_sum_float = q4c_tile_x1(wo, (const HVX_Vector *) (y_q + (kt + 1) * 1152), i8, v_sum_float);
    }
    if (kt < n_k_tiles) {
        const HVX_Vector * restrict p = (const HVX_Vector *) (tile_ptr + (size_t) kt * 576);
        HVX_Vector we[5] = { p[0], p[1], p[2], p[3], p[4] };
        v_sum_float = q4c_tile_x1(we, (const HVX_Vector *) (y_q + kt * 1152), i8, v_sum_float);
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

    const uint32_t n_k_tiles = n / 32;
    uint32_t kt = 0;
    for (; kt + 1 < n_k_tiles; kt += 2) {
        const HVX_Vector * restrict p = (const HVX_Vector *) (tile_ptr + (size_t) kt * 576);
        HVX_Vector we[5] = { p[0], p[1], p[2], p[3], p[4] };
        Q4C_ODD_FROM_PAIR(wo, p);
        q4c_tile_x2(we, (const HVX_Vector *) (y0_q + kt * 1152), (const HVX_Vector *) (y1_q + kt * 1152), i8, &v_sum_float_c0, &v_sum_float_c1);
        q4c_tile_x2(wo, (const HVX_Vector *) (y0_q + (kt + 1) * 1152), (const HVX_Vector *) (y1_q + (kt + 1) * 1152), i8, &v_sum_float_c0, &v_sum_float_c1);
    }
    if (kt < n_k_tiles) {
        const HVX_Vector * restrict p = (const HVX_Vector *) (tile_ptr + (size_t) kt * 576);
        HVX_Vector we[5] = { p[0], p[1], p[2], p[3], p[4] };
        q4c_tile_x2(we, (const HVX_Vector *) (y0_q + kt * 1152), (const HVX_Vector *) (y1_q + kt * 1152), i8, &v_sum_float_c0, &v_sum_float_c1);
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
s = s[:i0] + q4 + s[i1:]

# ---------------------------------------------------------------- Q8_0 contiguous (stride 1088): pairs = 17 aligned vectors
i0 = s.index('// qwen4: Q8_0 tiles stored contiguously')
i1 = s.index('static inline HVX_Vector q5_k_weight_scale_product(')
q8 = r'''// qwen4: Q8_0 tiles stored contiguously (stride 1088 B). Two tiles = 2176 B = 17 aligned vectors p[0..16]; the odd tile starts 64 B into p[8] (rebuilt with valign,
// scale = high half of p[16]). Bit-identical to the padded kernels.
static inline HVX_Vector q8c_tile_x1(const HVX_Vector * w, const HVX_Vector * v_act, HVX_Vector acc) {
    HVX_Vector v_sum = accum_q8_0_32x1(w, v_act);
    HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);
    HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(w[8], v_act[8]);
    HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);
    return hvx_vec_add_f32_f32(acc, v_sum_scaled);
}

static inline void q8c_tile_x2(const HVX_Vector * w, const HVX_Vector * v_act0, const HVX_Vector * v_act1, HVX_Vector * acc0, HVX_Vector * acc1) {
    HVX_VectorPair v_sums = accum_q8_0_32x2(w, v_act0, v_act1);
    HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(v_sums));
    HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(v_sums));
    HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(w[8], v_act0[8]);
    HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(w[8], v_act1[8]);
    *acc0 = hvx_vec_add_f32_f32(*acc0, hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0));
    *acc1 = hvx_vec_add_f32_f32(*acc1, hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1));
}

#define Q8C_EVEN_FROM_PAIR(we, p) \
    HVX_Vector we[9] = { (p)[0], (p)[1], (p)[2], (p)[3], (p)[4], (p)[5], (p)[6], (p)[7], (p)[8] }
#define Q8C_ODD_FROM_PAIR(wo, p)                                                                                  \
    HVX_Vector wo[9] = { Q6_V_valign_VVR((p)[9],  (p)[8],  64), Q6_V_valign_VVR((p)[10], (p)[9],  64),            \
                         Q6_V_valign_VVR((p)[11], (p)[10], 64), Q6_V_valign_VVR((p)[12], (p)[11], 64),            \
                         Q6_V_valign_VVR((p)[13], (p)[12], 64), Q6_V_valign_VVR((p)[14], (p)[13], 64),            \
                         Q6_V_valign_VVR((p)[15], (p)[14], 64), Q6_V_valign_VVR((p)[16], (p)[15], 64),            \
                         Q6_V_vror_VR((p)[16], 64) }

static void tiled_vec_dot_q8_0c_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows, const float * restrict sz) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();

    const uint32_t n_k_tiles = n / 32;
    uint32_t kt = 0;
    for (; kt + 1 < n_k_tiles; kt += 2) {
        const HVX_Vector * restrict p = (const HVX_Vector *) (tile_ptr + (size_t) kt * 1088);
        Q8C_EVEN_FROM_PAIR(we, p);
        Q8C_ODD_FROM_PAIR(wo, p);
        v_sum_float = q8c_tile_x1(we, (const HVX_Vector *) (y_q + kt * 1152), v_sum_float);
        v_sum_float = q8c_tile_x1(wo, (const HVX_Vector *) (y_q + (kt + 1) * 1152), v_sum_float);
    }
    if (kt < n_k_tiles) {
        const HVX_Vector * restrict p = (const HVX_Vector *) (tile_ptr + (size_t) kt * 1088);
        Q8C_EVEN_FROM_PAIR(we, p);
        v_sum_float = q8c_tile_x1(we, (const HVX_Vector *) (y_q + kt * 1152), v_sum_float);
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

    const uint32_t n_k_tiles = n / 32;
    uint32_t kt = 0;
    for (; kt + 1 < n_k_tiles; kt += 2) {
        const HVX_Vector * restrict p = (const HVX_Vector *) (tile_ptr + (size_t) kt * 1088);
        Q8C_EVEN_FROM_PAIR(we, p);
        Q8C_ODD_FROM_PAIR(wo, p);
        q8c_tile_x2(we, (const HVX_Vector *) (y0_q + kt * 1152), (const HVX_Vector *) (y1_q + kt * 1152), &v_sum_float_c0, &v_sum_float_c1);
        q8c_tile_x2(wo, (const HVX_Vector *) (y0_q + (kt + 1) * 1152), (const HVX_Vector *) (y1_q + (kt + 1) * 1152), &v_sum_float_c0, &v_sum_float_c1);
    }
    if (kt < n_k_tiles) {
        const HVX_Vector * restrict p = (const HVX_Vector *) (tile_ptr + (size_t) kt * 1088);
        Q8C_EVEN_FROM_PAIR(we, p);
        q8c_tile_x2(we, (const HVX_Vector *) (y0_q + kt * 1152), (const HVX_Vector *) (y1_q + kt * 1152), &v_sum_float_c0, &v_sum_float_c1);
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
s = s[:i0] + q8 + s[i1:]
wr(p, s, nl)
print('cpair ok')
