"""One-shot patch: adds hmx_mm_id_pipelined (GGML_HEXAGON_MMID_PIPE) to htp/matmul-ops.c and the CMake option.
Idempotent: skips if already applied."""
import os
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'llama.cpp-hexagon', 'ggml', 'src', 'ggml-hexagon', 'htp')
p = os.path.join(root, 'matmul-ops.c')
s = open(p, newline='').read()
crlf = '\r\n' in s
s = s.replace('\r\n', '\n')
if 'hmx_mm_id_pipelined' in s:
    print('already applied'); raise SystemExit

NEW = r'''
#ifdef GGML_HEXAGON_MMID_PIPE
static worker_callback_t hmx_mm_id_dequant_fn(int weight_type) {
    switch (weight_type) {
        case HTP_TYPE_Q4_0:   return dequantize_tiled_worker_loop_q4_0;
        case HTP_TYPE_IQ4_NL: return dequantize_tiled_worker_loop_iq4_nl;
        case HTP_TYPE_Q4_1:
        case HTP_TYPE_Q4_K:   return dequantize_tiled_worker_loop_q4_1;
        case HTP_TYPE_MXFP4:  return dequantize_tiled_worker_loop_mxfp4;
        case HTP_TYPE_Q8_0:   return dequantize_tiled_worker_loop_q8_0;
        case HTP_TYPE_Q5_K:   return dequantize_tiled_worker_loop_q5_k;
        case HTP_TYPE_Q6_K:   return dequantize_tiled_worker_loop_q6_k;
        default:              return NULL;
    }
}

// Cross-expert software pipeline for prefill MoE (MUL_MAT_ID / _NX).
// Work items are (expert, weight) pairs; the per-expert F32->F16 activation gather is shared by all weights of the
// expert. Weight DMA for item i+1, HMX for item i and output scatter for item i-1 overlap dequant of item i.
// Only experts with 0 < rows <= *m_max_out are handled here; the caller runs the legacy path for the rest.
// Returns 0 on success, 1 if not applicable (no work done), <0 on error.
static int hmx_mm_id_pipelined(
        struct htp_ops_context * octx, struct htp_mm_context * mmctx,
        const struct htp_tensor * const * srcw, const struct htp_tensor * const * dstw, int nw,
        const struct htp_tensor * act, int * m_max_out) {
    *m_max_out = 0;
    struct htp_context * ctx = octx->ctx;
    if (ctx->mdev.count > 1 || nw < 1 || nw > 4 || !mmctx->matrix_rows) return 1;

    const int k = (int) srcw[0]->ne[0];
    const int n = (int) srcw[0]->ne[1];
    const int n_as = (int) srcw[0]->ne[2];
    if (k % 32 != 0 || n % 32 != 0 || !hex_is_aligned((const void *) act->data, VLEN)) return 1;

    worker_callback_t dq_fn[4];
    size_t row_stride[4];
    size_t raw_bytes = 0;
    const int n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    for (int p = 0; p < nw; ++p) {
        if (!srcw[p] || !dstw[p]) return 1;
        if ((int) srcw[p]->ne[0] != k || (int) srcw[p]->ne[1] != n || (int) srcw[p]->ne[2] != n_as) return 1;
        if (!hex_is_aligned((const void *) dstw[p]->data, VLEN)) return 1;
        dq_fn[p] = hmx_mm_id_dequant_fn((int) srcw[p]->type);
        if (!dq_fn[p]) return 1;
        row_stride[p] = htp_mm_get_tiled_row_stride((int) srcw[p]->type, k);
        if (row_stride[p] == 0) return 1;
        const size_t rb = (size_t)(n / 32) * n_k_tiles * htp_mm_get_weight_aligned_tile_size((int) srcw[p]->type);
        if (rb > raw_bytes) raw_bytes = rb;
    }
    raw_bytes = hex_align_up(raw_bytes, HTP_MM_HMX_TILE_SIZE);
    const size_t wf_bytes = hex_align_up((size_t) n * k * sizeof(__fp16), HTP_MM_HMX_TILE_SIZE);
    const size_t fixed = 2 * raw_bytes + 2 * wf_bytes + 8 * HTP_MM_HMX_TILE_SIZE;
    const size_t per_row = 2 * (size_t) k * sizeof(__fp16) + 2 * (size_t) n * sizeof(__fp16);
    if (fixed >= ctx->vtcm_size) return 1;
    int m_max = (int) ((ctx->vtcm_size - fixed) / per_row) & ~31;
    if (m_max > 256) m_max = 256;
    if (m_max < 32) return 1;

    const uint32_t * counts = mmctx->matrix_row_counts;
    uint16_t experts[512];
    int ne = 0;
    for (int e = 0; e < n_as && ne < 512; ++e) {
        if (counts[e] > 0 && (int) counts[e] <= m_max) experts[ne++] = (uint16_t) e;
    }
    *m_max_out = m_max;
    if (ne == 0) return 0;

    uint8_t * vp = (uint8_t *) ctx->vtcm_base;
    void   * raw[2]  = { vtcm_seq_alloc(&vp, raw_bytes), vtcm_seq_alloc(&vp, raw_bytes) };
    __fp16 * wf[2]   = { (__fp16 *) vtcm_seq_alloc(&vp, wf_bytes), (__fp16 *) vtcm_seq_alloc(&vp, wf_bytes) };
    const size_t act_bytes = hex_align_up((size_t) m_max * k * sizeof(__fp16), HTP_MM_HMX_TILE_SIZE);
    const size_t out_bytes = hex_align_up((size_t) m_max * n * sizeof(__fp16), HTP_MM_HMX_TILE_SIZE);
    __fp16 * actb[2] = { (__fp16 *) vtcm_seq_alloc(&vp, act_bytes), (__fp16 *) vtcm_seq_alloc(&vp, act_bytes) };
    __fp16 * outb[2] = { (__fp16 *) vtcm_seq_alloc(&vp, out_bytes), (__fp16 *) vtcm_seq_alloc(&vp, out_bytes) };
    __fp16 * scales  = (__fp16 *) vtcm_seq_alloc(&vp, 256);
    if ((size_t) (vp - (uint8_t *) ctx->vtcm_base) > ctx->vtcm_size) { *m_max_out = 0; return 1; }
    hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));

    dma_queue * dma = ctx->dma[0];
    const struct fastdiv_values n_k_tiles_div = init_fastdiv_values(n_k_tiles);
    const int N = ne * nw;
    hmx_matmul_job_t jobs[2];

#define MMID_PIPE_PUSH_DMA(idx) do {                                                                        \
        const int ei_ = (idx) / nw, p_ = (idx) % nw;                                                        \
        const int ty_ = (int) srcw[p_]->type;                                                               \
        const uint32_t ts_ = htp_mm_get_weight_tile_size(ty_);                                              \
        const uint32_t as_ = htp_mm_get_weight_aligned_tile_size(ty_);                                      \
        dma_queue_push(dma, dma_make_data(raw[(idx) & 1], srcw[p_]->data + (size_t) experts[ei_] * srcw[p_]->nb[2]), \
                       as_, ts_, ts_, (size_t)(n / 32) * n_k_tiles);                                        \
    } while (0)

    MMID_PIPE_PUSH_DMA(0);
    for (int i = 0; i < N; ++i) {
        const int ei = i / nw, p = i % nw;
        const int cur_a = experts[ei];
        const int cne1  = (int) counts[cur_a];
        const int m_pad = hex_align_up(cne1, 32);
        const int ty    = (int) srcw[p]->type;

        void * curr_raw = (void *) dma_queue_pop(dma).dst;
        if (i + 1 < N) MMID_PIPE_PUSH_DMA(i + 1);

        dequantize_tiled_weight_chunk_to_fp16_tiles(ctx, wf[i & 1], curr_raw, n, k, row_stride[p], ty,
                                                    n_k_tiles, n_k_tiles_div, dq_fn[p], (int) octx->n_threads);
        if (p == 0) {
            transfer_activation_chunk_gathered_threaded(
                ctx, actb[ei & 1], (const float *) act->data, 0, m_pad, k,
                mmctx->matrix_rows, cur_a, mmctx->mapping_stride, (int) act->ne[1],
                act->nb[1], act->nb[2], cne1, (int) octx->n_threads, (int) act->ne[0],
                hmx_use_q8_0_activation_reference(ty));
        }

        hmx_matmul_job_init(&jobs[i & 1], outb[i & 1], actb[ei & 1], wf[i & 1], scales,
                            m_pad / HTP_MM_HMX_TILE_N_ROWS, n / HTP_MM_HMX_TILE_N_COLS, k / HTP_MM_HMX_TILE_N_ROWS);
        hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &jobs[i & 1]));

        if (i > 0) {
            hmx_queue_pop(ctx->hmx_queue);
            const int j = i - 1, ej = j / nw, pj = j % nw, aj = experts[ej], cj = (int) counts[aj];
            transfer_output_chunk_scattered_threaded(
                ctx, (float *) dstw[pj]->data, outb[j & 1], 0, hex_align_up(cj, 32), n,
                mmctx->matrix_rows, aj, mmctx->mapping_stride, dstw[pj]->nb[1], dstw[pj]->nb[2], cj,
                (int) octx->n_threads);
        }
    }
    {
        hmx_queue_pop(ctx->hmx_queue);
        const int j = N - 1, ej = j / nw, pj = j % nw, aj = experts[ej], cj = (int) counts[aj];
        transfer_output_chunk_scattered_threaded(
            ctx, (float *) dstw[pj]->data, outb[j & 1], 0, hex_align_up(cj, 32), n,
            mmctx->matrix_rows, aj, mmctx->mapping_stride, dstw[pj]->nb[1], dstw[pj]->nb[2], cj,
            (int) octx->n_threads);
    }
#undef MMID_PIPE_PUSH_DMA
    return 0;
}
#endif  // GGML_HEXAGON_MMID_PIPE

static int hmx_mm_op_matmul_id(
'''
anchor = '\nstatic int hmx_mm_op_matmul_id(\n'
assert anchor in s
s = s.replace(anchor, NEW, 1)

old1 = '''    const int n_ids = octx->src[2]->ne[0];
    const int n_as  = ne02;

    for (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        const int32_t cne1 = matrix_row_counts[cur_a];
        if (cne1 == 0) continue;
'''
new1 = '''    const int n_ids = octx->src[2]->ne[0];
    const int n_as  = ne02;

    int pipe_mmax = 0;
#ifdef GGML_HEXAGON_MMID_PIPE
    {
        const struct htp_tensor * sw[1] = { src0 };
        const struct htp_tensor * dw[1] = { dst };
        if (hmx_mm_id_pipelined(octx, mmctx, sw, dw, 1, src1, &pipe_mmax) < 0) return HTP_STATUS_INTERNAL_ERR;
    }
#endif

    for (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        const int32_t cne1 = matrix_row_counts[cur_a];
        if (cne1 == 0) continue;
        if (pipe_mmax > 0 && cne1 <= pipe_mmax) continue;
'''
assert old1 in s; s = s.replace(old1, new1, 1)

old2 = '''    const int n_as = src0->ne[2];

    for (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        const int32_t cne1 = matrix_row_counts[cur_a];
        if (cne1 == 0) continue;
'''
new2 = '''    const int n_as = src0->ne[2];

    int pipe_mmax = 0;
#ifdef GGML_HEXAGON_MMID_PIPE
    if (n_weights <= 4) {
        const struct htp_tensor * sw[4] = { 0 };
        const struct htp_tensor * dw[4] = { 0 };
        for (uint32_t p = 0; p < n_weights; ++p) { sw[p] = octx->src[p]; dw[p] = octx->dsts[p]; }
        if (hmx_mm_id_pipelined(octx, mmctx, sw, dw, (int) n_weights, act, &pipe_mmax) < 0) return HTP_STATUS_INTERNAL_ERR;
    }
#endif

    for (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        const int32_t cne1 = matrix_row_counts[cur_a];
        if (cne1 == 0) continue;
        if (pipe_mmax > 0 && cne1 <= pipe_mmax) continue;
'''
assert old2 in s; s = s.replace(old2, new2, 1)
if crlf: s = s.replace('\n', '\r\n')
open(p, 'w', newline='').write(s)

p = os.path.join(root, 'CMakeLists.txt')
c = open(p, newline='').read()
nl = '\r\n' if '\r\n' in c else '\n'
a = 'option(GGML_HEXAGON_RMS_REGISTER_BARRIER_REFERENCE'
assert a in c
c = c.replace(a, 'option(GGML_HEXAGON_MMID_PIPE "Cross-expert software-pipelined HMX MoE prefill kernel" OFF)' + nl + a, 1)
b = '    $<$<BOOL:${GGML_HEXAGON_RMS_REGISTER_BARRIER_REFERENCE}>:GGML_HEXAGON_RMS_REGISTER_BARRIER_REFERENCE=1>'
assert b in c
c = c.replace(b, b + nl + '    $<$<BOOL:${GGML_HEXAGON_MMID_PIPE}>:GGML_HEXAGON_MMID_PIPE=1>', 1)
open(p, 'w', newline='').write(c)
print('patched')
