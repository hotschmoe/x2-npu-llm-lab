import sys,os; sys.path.insert(0,r'C:/Users/HOTSCH~1/AppData/Local/Temp/claude/C--Users-hotschmoe-documents-llm-inference/12944359-adec-4afb-a6e9-134244e56374/scratchpad')
from sub import sub
sub('layout-q4.h','// decode diagnostics:',r'''// L1x: nibbles stored as (n ^ 8) = 4-bit two's complement of (n-8): sign extension replaces and/lsr + vsub
static inline void lq4_dequant_tile_L1x(const uint8_t * restrict nib, HVX_Vector vsd, __fp16 * restrict dst) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        HVX_Vector v  = hvx_vmem(nib + i * 128);
        HVX_Vector x0 = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(v, 12), 12);
        HVX_Vector x1 = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(v, 8), 12);
        HVX_Vector x2 = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(v, 4), 12);
        HVX_Vector x3 = Q6_Vh_vasr_VhR(v, 12);
        hvx_vmem(dst + (4 * i + 0) * 64) = lq4_cvt_scale(x0, vsd);
        hvx_vmem(dst + (4 * i + 1) * 64) = lq4_cvt_scale(x1, vsd);
        hvx_vmem(dst + (4 * i + 2) * 64) = lq4_cvt_scale(x2, vsd);
        hvx_vmem(dst + (4 * i + 3) * 64) = lq4_cvt_scale(x3, vsd);
    }
}
// decode with (n^8)-stored nibbles: xor 8 then vsub 8 (= 4-bit sign extension)
static inline HVX_Vector lq4_accum_32x1_x(const HVX_Vector * restrict vptr, const HVX_Vector * restrict v_act) {
    HVX_Vector v_sum0 = Q6_V_vzero(), v_sum1 = Q6_V_vzero();
    const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector i8 = Q6_Vb_vsplat_R(8);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        HVX_Vector v = vptr[i];
        HVX_Vector lo = Q6_V_vand_VV(v, mask_h4);
        HVX_Vector hi = Q6_Vub_vlsr_VubR(v, 4);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, Q6_Vb_vsub_VbVb(Q6_V_vxor_VV(lo, i8), i8), v_act[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, Q6_Vb_vsub_VbVb(Q6_V_vxor_VV(hi, i8), i8), v_act[i * 2 + 1]);
    }
    return Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
}
static void lq4_diag_dot_x(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    const uint8_t * restrict wp = vx; const uint8_t * restrict y_q = vy;
    HVX_Vector acc = Q6_V_vzero();
    const HVX_VectorPred q_ev = Q6_Q_vand_VR(Q6_V_vsplat_R(0x0000FFFF), 0x01010101);
    for (uint32_t pt = 0; pt < n / 64; pt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (wp + pt * 1152);
        const HVX_Vector * restrict a0 = (const HVX_Vector *) (y_q + (2 * pt) * 1152);
        const HVX_Vector * restrict a1 = a0 + 9;
        HVX_Vector se = lq4_accum_32x1_x(vptr, a0), so = lq4_accum_32x1_x(vptr + 4, a1);
        HVX_Vector a_comb = Q6_V_vmux_QVV(q_ev, a0[8], a1[8]);
        HVX_VectorPair sc = Q6_Wsf_vmpy_VhfVhf(vptr[8], a_comb);
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(se), Q6_V_lo_W(sc)));
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(so), Q6_V_hi_W(sc)));
    }
    hvx_vec_store_u(s, 128, acc);
}

// decode diagnostics:''')
sub('layout-bench.h','    // old dequant again with the old kernel',r'''    // L1x: (n^8) stored nibbles, sign-extension dequant (checked against the same reference), and the matching decode kernel (pair packaging)
    for (int t = 0; t < LB_NT; t++) for (int i = 0; i < 512; i++) w_l1[t * 640 + i] ^= 0x88;
    for (int pt = 0; pt < LB_NT / 2; pt++) for (int i = 0; i < 1024; i++) w_pl1[pt * 1152 + i] ^= 0x88;
    memset(d_new, 0, LB_NT * 2048);
    LB_TIME(({ for (int t = 0; t < LB_NT; t++) lq4_dequant_tile_L1x(w_l1 + t * 640, lq4_scale_dup_640(w_l1 + t * 640), d_new + t * 1024); }), cy, pk);
    r[nr++] = cy; r[nr++] = pk; LB_CHECK_DQ(L1x640);
    memset(d_new, 0, LB_NT * 2048);
    LB_TIME(({ for (int t = 0; t < LB_NT; t++) { const uint8_t * blk = w_pl1 + (t >> 1) * 1152; lq4_dequant_tile_L1x(blk + (t & 1) * 512, lq4_scale_dup_pair(blk, t & 1), d_new + t * 1024); } }), cy, pk);
    r[nr++] = cy; r[nr++] = pk; LB_CHECK_DQ(L1xpair);
    // decode with xor-stored nibbles: pair packaging, swizzled act (same operands as L1 pair)
    LB_TIME(lq4_diag_dot_x(LB_NT * 32, o_new, w_pl1, y_sw), cy, pk);
    r[nr++] = cy; r[nr++] = pk;
    { float t1[32]; lq4_diag_dot_x(LB_NT * 32, o_new, w_pl1, y_sw); tiled_vec_dot_q4_0_32x1(LB_NT * 32, o_ref, w_old, y_nat, 32, NULL); r[nr++] = (uint32_t) memcmp(o_ref, o_new, 128) != 0; (void) t1; r[nr++] = 0; }
    // old dequant again with the old kernel''')
