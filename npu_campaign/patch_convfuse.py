"""Prefill conv-window fusion: CONCAT(transpose)+CPY(state update)+SSM_CONV(+SILU) -> one token-major DSP op.
Edits: htp/ssm-conv.h, htp/ssm-conv.c, ggml-hexagon.cpp, htp/htp-opnode.h (log name)."""
import os, re
base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'llama.cpp-hexagon', 'ggml', 'src', 'ggml-hexagon')

def load(rel):
    p = os.path.join(base, rel)
    s = open(p, newline='').read()
    return p, s.replace('\r\n', '\n'), '\r\n' in s

def save(p, s, crlf):
    if crlf: s = s.replace('\n', '\r\n')
    open(p, 'w', newline='').write(s)

# ---------------------------------------------------------------- ssm-conv.h
p, s, crlf = load('htp/ssm-conv.h')
if 'htp_ssm_pf_layout' not in s:
    s = s.replace('''    uint32_t fused_window;
};''', '''    uint32_t fused_window;   // 2: prefill token-major window (see below)

    // fused_window == 2: prefill. src0 = packed 3-token history [3, d_inner], src1 = kernel, src2 = token-major
    // qkv (ne0 = n_t tokens, ne1 = d_inner channels, nb1 == 4), dst[0] = conv (optionally SiLU'd) [d_inner, n_t],
    // dst[1] = new history. The CONCAT and the 2 DDR transposes around it are not materialized.
    uint32_t silu;
};''')
    s = s.replace('''#if defined(__cplusplus)
static_assert(sizeof(struct htp_ssm_conv_kernel_params)''', '''// Per-thread VTCM block for the fused prefill window: 128-channel tiles, 128-token blocks, double buffered.
#define HTP_SSM_PF_TILE 128u
#define HTP_SSM_PF_B    128u
struct htp_ssm_pf_layout {
    uint32_t off_xin[2];
    uint32_t off_y[2];
    uint32_t off_sig;
    uint32_t off_wraw;
    uint32_t off_wT;
    uint32_t off_st;
    uint32_t off_ns;
    uint32_t per_thread;
};

static inline struct htp_ssm_pf_layout htp_ssm_pf_layout(void) {
    struct htp_ssm_pf_layout L;
    const uint32_t xin_bytes = htp_ssm_win_align128((HTP_SSM_PF_B + 3u) * HTP_SSM_PF_TILE * 4u);
    const uint32_t y_bytes   = htp_ssm_win_align128(HTP_SSM_PF_B * HTP_SSM_PF_TILE * 4u);
    uint32_t o = 0;
    L.off_xin[0] = o; o += xin_bytes;
    L.off_xin[1] = o; o += xin_bytes;
    L.off_y[0]   = o; o += y_bytes;
    L.off_y[1]   = o; o += y_bytes;
    L.off_sig    = o; o += y_bytes;
    L.off_wraw   = o; o += htp_ssm_win_align128(HTP_SSM_PF_TILE * 4u * 4u);
    L.off_wT     = o; o += htp_ssm_win_align128(HTP_SSM_PF_TILE * 4u * 4u);
    L.off_st     = o; o += htp_ssm_win_align128(HTP_SSM_PF_TILE * 3u * 4u);
    L.off_ns     = o; o += htp_ssm_win_align128(HTP_SSM_PF_TILE * 3u * 4u);
    L.per_thread = o;
    return L;
}

#if defined(__cplusplus)
static_assert(sizeof(struct htp_ssm_conv_kernel_params)''', 1)
    save(p, s, crlf)

# ---------------------------------------------------------------- ssm-conv.c
p, s, crlf = load('htp/ssm-conv.c')
if 'ssm_conv_prefill_tm_thread' not in s:
    kernel = r'''
// ---- fused prefill window (fused_window == 2) -------------------------------------------------------------
// Token-major in/out: rows are DMA'd straight from the matmul output (row = token, contiguous channels), the four
// conv taps are four row-shifted vectors, so no DDR or in-register transposes are needed. Same accumulation order
// (qf32 add of x*w for j = 0..3) as the generic prefill kernel; SiLU uses the same sigmoid + mul helpers as the SILU op.

// rows 0..2 of the block-0 input = the three history tokens, taken from the [c][3] packed state.
static inline void ssm_pf_fill_state_rows(float * xin, const float * st_raw, uint32_t tile_n, uint32_t rowfl) {
    __attribute__((aligned(128))) unsigned char scratch[768];
    for (uint32_t c = 0; c < tile_n; c += 32) {
        const HVX_Vector * sp = (const HVX_Vector *) (st_raw + c * 3);
        ((HVX_Vector *) scratch)[0] = sp[0];
        ((HVX_Vector *) scratch)[1] = sp[1];
        ((HVX_Vector *) scratch)[2] = sp[2];
        asm volatile("barrier" ::: "memory");
        for (unsigned i = 0; i < 384; i += 32) {
            Q6_dcinva_A(scratch + i);
        }
        asm volatile("barrier" ::: "memory");
        const float * st = (const float *) scratch;
        float * rows = (float *) (scratch + 384);
        for (unsigned ch = 0; ch < 32; ++ch) {
            rows[0 * 32 + ch] = st[ch * 3 + 0];
            rows[1 * 32 + ch] = st[ch * 3 + 1];
            rows[2 * 32 + ch] = st[ch * 3 + 2];
        }
        for (unsigned j = 0; j < 3; ++j) {
            for (unsigned ch = 0; ch < 32; ++ch) {
                xin[j * rowfl + c + ch] = rows[j * 32 + ch];
            }
        }
    }
    asm volatile("barrier" ::: "memory");
}

// new packed history [c][3] from the last three input rows (rows r0, r0+1, r0+2 of the block buffer).
static inline void ssm_pf_extract_state(const float * last3, float * ns, uint32_t tile_n, uint32_t rowfl) {
    __attribute__((aligned(128))) unsigned char scratch[768];
    for (uint32_t c = 0; c < tile_n; c += 32) {
        ((HVX_Vector *) scratch)[0] = *(const HVX_Vector *) (last3 + 0 * rowfl + c);
        ((HVX_Vector *) scratch)[1] = *(const HVX_Vector *) (last3 + 1 * rowfl + c);
        ((HVX_Vector *) scratch)[2] = *(const HVX_Vector *) (last3 + 2 * rowfl + c);
        asm volatile("barrier" ::: "memory");
        for (unsigned i = 0; i < 384; i += 32) {
            Q6_dcinva_A(scratch + i);
        }
        asm volatile("barrier" ::: "memory");
        const float * r0 = (const float *) scratch;
        const float * r1 = r0 + 32;
        const float * r2 = r0 + 64;
        float * out = (float *) (scratch + 384);
        for (unsigned ch = 0; ch < 32; ++ch) {
            out[ch * 3 + 0] = r0[ch];
            out[ch * 3 + 1] = r1[ch];
            out[ch * 3 + 2] = r2[ch];
        }
        for (unsigned i = 0; i < 96; ++i) {
            ns[c * 3 + i] = out[i];
        }
    }
    asm volatile("barrier" ::: "memory");
}

static void ssm_conv_prefill_tm_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_ssm_conv_context *             scctx   = (struct htp_ssm_conv_context *) data;
    struct htp_ops_context *                  octx    = scctx->octx;
    const struct htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    const struct htp_tensor * restrict state = octx->src[0];
    const struct htp_tensor * restrict kern  = octx->src[1];
    const struct htp_tensor * restrict tok   = octx->src[2];
    const struct htp_tensor * restrict ydst  = octx->dsts[0];
    const struct htp_tensor * restrict ndst  = octx->dsts[1];

    const uint32_t dr  = scctx->nrows_per_thread;
    const uint32_t ir0 = scctx->row_start + dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, scctx->row_start + scctx->nrows);
    if (ir0 >= ir1) {
        return;
    }

    const uint32_t T    = kparams->n_t;
    const uint32_t B    = HTP_SSM_PF_B;
    const uint32_t nblk = (T + B - 1) / B;
    const bool     silu = kparams->silu != 0;
    const size_t   tok_tstride = tok->nb[0];
    const size_t   y_tstride   = ydst->nb[1];

    dma_queue * q = octx->ctx->dma[ith];
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    uint8_t * base = octx->src0_spad.data + (size_t) ith * octx->src0_spad.size_per_thread;
    const struct htp_ssm_pf_layout L = htp_ssm_pf_layout();
    float * xin[2] = { (float *) (base + L.off_xin[0]), (float *) (base + L.off_xin[1]) };
    float * yb[2]  = { (float *) (base + L.off_y[0]),   (float *) (base + L.off_y[1]) };
    float * sigb   = (float *) (base + L.off_sig);
    float * w_raw  = (float *) (base + L.off_wraw);
    float * w_T    = (float *) (base + L.off_wT);
    float * st_raw = (float *) (base + L.off_st);
    float * ns_buf = (float *) (base + L.off_ns);

    uint32_t in_seq[64], out_seq[64];   // T <= 64 * 128 = 8192 tokens per op (host checks)

    for (uint32_t c0 = ir0; c0 < ir1; c0 += HTP_SSM_PF_TILE) {
        const uint32_t tile_n = MIN(HTP_SSM_PF_TILE, ir1 - c0);
        const uint32_t rowfl  = tile_n;                       // floats per VTCM row
        const size_t   rowbytes = (size_t) tile_n * sizeof(float);

        uint32_t sp = 0, pp = 0;
#define PF_WAIT(target) do { while (pp <= (target)) { dma_queue_pop(q); pp++; } } while (0)

        // weights + packed state for this channel tile
        const size_t w_bytes  = (size_t) tile_n * 4u * sizeof(float);
        const size_t st_bytes = (size_t) tile_n * 3u * sizeof(float);
        dma_queue_push(q, dma_make_data((uint8_t *) w_raw,  kern->data  + (size_t) c0 * kern->nb[1]),  w_bytes,  w_bytes,  w_bytes,  1);
        dma_queue_push(q, dma_make_data((uint8_t *) st_raw, state->data + (size_t) c0 * state->nb[1]), st_bytes, st_bytes, st_bytes, 1);
        sp += 2;
        PF_WAIT(1);
        hvx_ssm_conv_unpack_to_T_4(w_raw, w_T, tile_n, tile_n);

        // block 0 input: rows 3.. from DDR, rows 0..2 from the packed state
        {
            const uint32_t bn = MIN(B, T);
            in_seq[0] = sp++;
            dma_queue_push(q, dma_make_data((uint8_t *) (xin[0] + 3u * rowfl), tok->data + (size_t) c0 * sizeof(float)),
                           rowbytes, tok_tstride, rowbytes, bn);
        }
        ssm_pf_fill_state_rows(xin[0], st_raw, tile_n, rowfl);
        PF_WAIT(in_seq[0]);

        for (uint32_t b = 0; b < nblk; ++b) {
            const uint32_t t0 = b * B;
            const uint32_t bn = MIN(B, T - t0);
            float * xr = xin[b & 1];
            float * yo = yb[b & 1];

            if (b + 1 < nblk) {
                const uint32_t nt0 = t0 + B;
                const uint32_t nbn = MIN(B, T - nt0);
                in_seq[b + 1] = sp++;
                dma_queue_push(q, dma_make_data((uint8_t *) xin[(b + 1) & 1], tok->data + (size_t) (nt0 - 3u) * tok_tstride + (size_t) c0 * sizeof(float)),
                               rowbytes, tok_tstride, rowbytes, nbn + 3u);
            }
            if (b >= 2) {
                PF_WAIT(out_seq[b - 2]);
            }

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) b);
            for (uint32_t t = 0; t < bn; ++t) {
                for (uint32_t cb = 0; cb < tile_n; cb += VLEN_FP32) {
                    HVX_Vector acc = hvx_vec_splat_f32(0.0f);
                    for (uint32_t j = 0; j < 4; ++j) {
                        HVX_Vector x = *(const HVX_Vector *) (xr + (t + j) * rowfl + cb);
                        HVX_Vector w = *(const HVX_Vector *) (w_T + j * tile_n + cb);
                        acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
                    }
                    *(HVX_Vector *) (yo + t * rowfl + cb) = Q6_Vsf_equals_Vqf32(acc);
                }
            }
            if (silu) {
                const uint32_t n = bn * rowfl;
                hvx_sigmoid_f32_aa((uint8_t *) sigb, (const uint8_t *) yo, n);
                hvx_mul_f32_aaa((uint8_t *) yo, (const uint8_t *) yo, (const uint8_t *) sigb, n);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) b);

            if (b + 1 == nblk) {
                // last three input rows of the last block (rows bn, bn+1, bn+2) -> new packed history
                ssm_pf_extract_state(xr + (size_t) bn * rowfl, ns_buf, tile_n, rowfl);
            }

            out_seq[b] = sp++;
            dma_queue_push(q, dma_make_data(ydst->data + (size_t) t0 * y_tstride + (size_t) c0 * sizeof(float), (uint8_t *) yo),
                           y_tstride, rowbytes, rowbytes, bn);
            if (b + 1 < nblk) {
                PF_WAIT(in_seq[b + 1]);
            }
        }

        const uint32_t ns_seq = sp++;
        dma_queue_push(q, dma_make_data(ndst->data + (size_t) c0 * 3u * sizeof(float), (uint8_t *) ns_buf), st_bytes, st_bytes, st_bytes, 1);
        PF_WAIT(ns_seq);
#undef PF_WAIT
    }
    (void) nth;
}

static int op_ssm_conv_prefill_tm(struct htp_ops_context * octx) {
    const struct htp_tensor * state = octx->src[0];
    const struct htp_tensor * kern  = octx->src[1];
    const struct htp_tensor * tok   = octx->src[2];
    const struct htp_tensor * ydst  = octx->dsts[0];
    const struct htp_tensor * ndst  = octx->dsts[1];
    if (!state || !kern || !tok || !ydst || !ndst) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (state->type != HTP_TYPE_F32 || kern->type != HTP_TYPE_F32 || tok->type != HTP_TYPE_F32 ||
        ydst->type != HTP_TYPE_F32 || ndst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }
    const struct htp_ssm_conv_kernel_params * kparams = (const struct htp_ssm_conv_kernel_params *) octx->kernel_params;
    if (kparams->d_conv != 4 || kparams->n_t < 4 || kparams->n_t > 64u * HTP_SSM_PF_B || kparams->n_s != 1) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (octx->ctx->mdev.count > 1) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    const struct htp_ssm_pf_layout L = htp_ssm_pf_layout();
    const uint32_t vtcm_total = L.per_thread * octx->n_threads;
    if (vtcm_total > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }
    octx->src0_spad.size_per_thread = L.per_thread;
    octx->src0_spad.size            = vtcm_total;
    octx->src0_spad.data            = octx->ctx->vtcm_base;
    octx->src0_spad.src             = NULL;
    octx->src1_spad.size            = 0;
    octx->dst_spad.size             = 0;

    struct htp_ssm_conv_context scctx = {
        .octx             = octx,
        .kparams          = kparams,
        .nrows_per_thread = kparams->d_inner_per_thread,
        .d_inner_tile     = HTP_SSM_PF_TILE,
        .row_start        = 0,
        .nrows            = kparams->d_inner,
    };
    work_queue_run(octx->ctx->work_queue, ssm_conv_prefill_tm_thread, &scctx, octx->n_threads);
    return HTP_STATUS_OK;
}

'''
    anchor = 'int op_ssm_conv_f32(struct htp_ops_context * octx) {'
    assert anchor in s
    s = s.replace(anchor, kernel + anchor, 1)
    old = '''    if (kparams->fused_window) {
        return op_ssm_conv_window(octx);
    }
'''
    assert old in s
    s = s.replace(old, '''    if (kparams->fused_window == 2) {
        return op_ssm_conv_prefill_tm(octx);
    }
    if (kparams->fused_window) {
        return op_ssm_conv_window(octx);
    }
''', 1)
    save(p, s, crlf)

# ---------------------------------------------------------------- htp-opnode.h log name
p, s, crlf = load('htp-opnode.h')
old = 'kparams->fused_window ? "decode-win"'
if old in s and 'prefill-win' not in s:
    s = s.replace(old, 'kparams->fused_window == 2 ? "prefill-win" : kparams->fused_window ? "decode-win"', 1)
    save(p, s, crlf)

# ---------------------------------------------------------------- host matcher
p, s, crlf = load('ggml-hexagon.cpp')
if 'try_fuse_ssm_conv_prefill' not in s:
    host = r'''    // Prefill (T > 1) conv-state window: CONCAT(state, transposed qkv) + CPY(new state) + SSM_CONV become one
    // token-major op that reads the matmul output directly. SiLU is folded in by try_fuse_ssm_conv_silu.
    bool try_fuse_ssm_conv_prefill(const htp_opnode & node) {
        if (node.opcode != HTP_OP_SSM_CONV || n_ops < 2) return false;
        if (sess->mdev.count > 1) return false;
        const ggml_tensor * conv = node.dst();
        if (!conv || conv->ne[1] < 4 || conv->ne[2] != 1 || conv->ne[3] != 1) return false;
        const int64_t T = conv->ne[1];
        if (T > 64 * (int64_t) HTP_SSM_PF_B) return false;

        const ggml_tensor * window = node.src0();
        const ggml_tensor * kernel = node.src1();
        if (!window || !kernel || window->op != GGML_OP_CONCAT) return false;
        if ((window->flags & GGML_TENSOR_FLAG_OUTPUT) || (conv->flags & GGML_TENSOR_FLAG_OUTPUT)) return false;
        if (kernel->type != GGML_TYPE_F32 || conv->type != GGML_TYPE_F32) return false;
        if (kernel->ne[0] != 4 || kernel->nb[0] != sizeof(float) || kernel->nb[1] != 4 * sizeof(float)) return false;
        const int64_t d_inner = kernel->ne[1];
        if (d_inner <= 0 || (d_inner % 32) != 0 || d_inner > UINT32_MAX) return false;
        if (conv->ne[0] != d_inner || conv->nb[0] != sizeof(float)) return false;
        if (((const int32_t *) window->op_params)[0] != 0) return false;

        const ggml_tensor * state = window->src[0];
        const ggml_tensor * token = window->src[1];
        if (!state || !token || state->type != GGML_TYPE_F32 || token->type != GGML_TYPE_F32) return false;
        if (state->ne[0] != 3 || state->ne[1] != d_inner || state->ne[2] != 1 || state->ne[3] != 1) return false;
        if (state->nb[0] != sizeof(float) || state->nb[1] != 3 * sizeof(float)) return false;
        if (token->ne[0] != T || token->ne[1] != d_inner || token->ne[2] != 1 || token->ne[3] != 1) return false;
        if (token->nb[1] != sizeof(float) || token->nb[0] < (size_t) d_inner * sizeof(float) || (token->nb[0] % 4) != 0) return false;
        if (window->ne[0] != T + 3) return false;

        int concat_i = -1;
        for (unsigned int i = 0; i < n_ops; i++) {
            if (ops[i].opcode == HTP_OP_CONCAT && ops[i].node == window) { concat_i = (int) i; break; }
        }
        if (concat_i < 0) return false;

        int cpy_i = -1;
        for (unsigned int i = (unsigned int) concat_i + 1; i < n_ops; i++) {
            if (ops[i].opcode != HTP_OP_CPY) continue;
            const ggml_tensor * src = ops[i].src0();
            if (!src || src->op != GGML_OP_VIEW || src->view_src != window) continue;
            if (src->view_offs != (size_t) T * sizeof(float)) continue;
            if (src->ne[0] != 3 || src->ne[1] != d_inner || src->nb[0] != sizeof(float) || src->nb[1] != (size_t) (T + 3) * sizeof(float)) continue;
            cpy_i = (int) i;
            break;
        }
        if (cpy_i < 0) return false;
        const ggml_tensor * state_dst = ops[cpy_i].node;
        if (!state_dst || state_dst->type != GGML_TYPE_F32 || (state_dst->flags & GGML_TENSOR_FLAG_OUTPUT)) return false;
        if (state_dst->ne[0] != 3 * d_inner || state_dst->ne[1] != 1 || state_dst->ne[2] != 1 || state_dst->nb[0] != sizeof(float)) return false;

        for (unsigned int i = 0; i < n_ops; i++) {
            if ((int) i == concat_i || (int) i == cpy_i) continue;
            for (const ggml_tensor * in : ops[i].inputs) {
                if (in == window) return false;
            }
        }
        auto backed = [](const ggml_tensor * t) { return t && t->buffer && t->extra; };
        if (!backed(state) || !backed(token) || !backed(kernel) || !backed(conv) || !backed(state_dst)) return false;

        const uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, (uint32_t) ((d_inner + 31) / 32));
        if (n_threads == 0) return false;
        const uint32_t raw_rpt = (uint32_t) ((d_inner + n_threads - 1) / n_threads);
        const uint32_t dpt     = (raw_rpt + 31u) & ~31u;
        for (uint32_t t = 0; t < n_threads; t++) {
            const uint32_t ir0 = dpt * t;
            if (ir0 >= (uint32_t) d_inner) break;
            if ((std::min)(dpt, (uint32_t) d_inner - ir0) % 32u) return false;
        }
        const struct htp_ssm_pf_layout pfl = htp_ssm_pf_layout();
        const uint32_t vtcm_total = pfl.per_thread * n_threads;
        if ((size_t) vtcm_total > sess->vtcm_size) return false;
        if (!try_fuse_common({state, token, kernel, conv, state_dst})) return false;

        struct htp_ssm_conv_kernel_params kparams;
        memset(&kparams, 0, sizeof(kparams));
        kparams.n_threads          = n_threads;
        kparams.d_conv             = 4;
        kparams.d_inner            = (uint32_t) d_inner;
        kparams.n_t                = (uint32_t) T;
        kparams.n_s                = 1;
        kparams.d_inner_per_thread = dpt;
        kparams.d_inner_tile       = HTP_SSM_PF_TILE;
        kparams.vtcm_size          = vtcm_total;
        kparams.vtcm_src0_size     = vtcm_total;
        kparams.div_n_threads      = init_fastdiv_values(n_threads);
        kparams.fused_window       = 2;
        kparams.silu               = 0;

        htp_opnode & fused = ops[concat_i];
        fused.node   = node.node;
        fused.opcode = HTP_OP_SSM_CONV;
        fused.name   = "SSM_CONV+PF";
        fused.inputs.clear();
        fused.inputs.push_back(state);
        fused.inputs.push_back(kernel);
        fused.inputs.push_back(token);
        fused.outputs.clear();
        fused.outputs.push_back(conv);
        fused.fused.clear();
        memset(fused.kernel_params, 0, sizeof(fused.kernel_params));
        memcpy(fused.kernel_params, &kparams, sizeof(kparams));

        htp_op_desc & o = h_ops[concat_i];
        memset(&o, 0, sizeof(o));
        o.opcode = HTP_OP_SSM_CONV;
        memcpy(o.kernel_params, &kparams, sizeof(kparams));
        o.src[0] = (uint16_t) add_tensor(state);
        o.src[1] = (uint16_t) add_tensor(kernel);
        o.src[2] = (uint16_t) add_tensor(token);
        for (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) o.src[s] = 0xffff;
        o.dst[0] = (uint16_t) add_tensor(conv);
        o.dst[1] = (uint16_t) add_tensor(state_dst);
        for (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) o.dst[d] = 0xffff;

        drop_op((unsigned int) cpy_i);

        static int logged = 0;
        if (logged++ == 0) {
            GGML_LOG_INFO("ggml-hex: %s fused prefill conv-state window into SSM_CONV (d_inner %u T %u)\n",
                          sess->c_name(), (unsigned) d_inner, (unsigned) T);
        }
        HEX_VERBOSE("ggml-hex: %s fused SSM_CONV+PF (#%d)\n", sess->c_name(), concat_i);
        return true;
    }

    // SILU right after the prefill conv is folded into it (same sigmoid+mul helpers on the DSP).
    bool try_fuse_ssm_conv_silu(const htp_opnode & node) {
        if (node.opcode != HTP_OP_UNARY_SILU || n_ops < 2) return false;
        const ggml_tensor * src = node.src0();
        const ggml_tensor * dst = node.dst();
        if (!src || !dst || src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
        if (dst->ne[0] != src->ne[0] || dst->ne[1] != src->ne[1] || dst->ne[2] != 1 || dst->ne[3] != 1) return false;
        if (dst->nb[0] != sizeof(float) || (dst->flags & GGML_TENSOR_FLAG_OUTPUT) || (src->flags & GGML_TENSOR_FLAG_OUTPUT)) return false;
        if (!dst->buffer || !dst->extra) return false;

        int ci = -1;
        for (int i = (int) n_ops - 1; i >= 0 && i >= (int) n_ops - 8; i--) {
            if (ops[i].opcode == HTP_OP_SSM_CONV && ops[i].name == "SSM_CONV+PF" &&
                ops[i].outputs.size() == 1 && ops[i].outputs[0] == src) {
                ci = i;
                break;
            }
        }
        if (ci < 0) return false;
        for (unsigned int i = 0; i < n_ops; i++) {
            if ((int) i == ci) continue;
            for (const ggml_tensor * in : ops[i].inputs) {
                if (in == src) return false;
            }
        }
        if (!try_fuse_common({dst})) return false;

        struct htp_ssm_conv_kernel_params kparams;
        memcpy(&kparams, ops[ci].kernel_params, sizeof(kparams));
        kparams.silu = 1;
        memcpy(ops[ci].kernel_params, &kparams, sizeof(kparams));
        ops[ci].name = "SSM_CONV+PF+SILU";
        ops[ci].outputs[0] = dst;
        ops[ci].fused.push_back(node.node);

        htp_op_desc & o = h_ops[ci];
        memcpy(o.kernel_params, &kparams, sizeof(kparams));
        o.dst[0] = (uint16_t) add_tensor(dst);

        static int logged = 0;
        if (logged++ == 0) {
            GGML_LOG_INFO("ggml-hex: %s fused SILU into prefill SSM_CONV\n", sess->c_name());
        }
        HEX_VERBOSE("ggml-hex: %s fused SSM_CONV+PF+SILU (#%d)\n", sess->c_name(), ci);
        return true;
    }

'''
    anchor = '    // Eight expert rows are summed by seven serial ADDs.'
    assert anchor in s
    s = s.replace(anchor, host + anchor, 1)
    old = '        if (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_SSM_CONV_WIN)  && try_fuse_ssm_conv_window(node)) return true;'
    assert old in s
    s = s.replace(old, old + '\n        if (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_SSM_CONV_WIN)  && try_fuse_ssm_conv_prefill(node)) return true;\n        if (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_SSM_CONV_WIN)  && try_fuse_ssm_conv_silu(node))    return true;', 1)
    save(p, s, crlf)
print('patched convfuse')
