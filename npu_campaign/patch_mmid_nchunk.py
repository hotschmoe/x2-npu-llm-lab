"""Generalize hmx_mm_id_pipelined to column chunks: items are (expert row-chunk, weight, n-chunk). Shapes whose whole n x k fp16 tile
fits VTCM (Qwen3.6) keep a single chunk (identical behavior); larger ones (Gemma 4 fused gate_up 1408x2816, down 2816x704) are
split into 32-column-multiple chunks so the pipeline applies instead of returning 'not applicable'."""
import os
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'llama.cpp-hexagon', 'ggml', 'src', 'ggml-hexagon', 'htp')
p = os.path.join(root, 'matmul-ops.c')
s = open(p, newline='').read()
crlf = '\r\n' in s
s = s.replace('\r\n', '\n')
if 'next_tot' in s:
    print('already applied'); raise SystemExit

# ---- worker state + worker: next-item partition is passed explicitly
old = '''    uint32_t                 next_tile;
    uint32_t                 next_aligned;
    int                      n_scatter;'''
assert old in s
s = s.replace(old, '''    uint32_t                 next_tile;
    uint32_t                 next_aligned;
    int                      next_tot;      // tiles in the next item
    int                      next_tpt;      // tiles per task of the next item
    int                      n_scatter;''', 1)
old = '''    if (start < end && st->push_next) {
        dma_queue_push(q, dma_make_data(st->next_raw + (size_t) start * st->next_aligned,
                                        st->next_src + (dma_addr_t) start * st->next_tile),
                       st->next_aligned, st->next_tile, st->next_tile, (size_t) (end - start));
    }'''
assert old in s
s = s.replace(old, '''    if (st->push_next) {
        const int nstart = (int) ith * st->next_tpt;
        const int nend   = hex_smin(nstart + st->next_tpt, st->next_tot);
        if (nstart < nend) {
            dma_queue_push(q, dma_make_data(st->next_raw + (size_t) nstart * st->next_aligned,
                                            st->next_src + (dma_addr_t) nstart * st->next_tile),
                           st->next_aligned, st->next_tile, st->next_tile, (size_t) (nend - nstart));
        }
    }''', 1)

# ---- new function body
a = s.index('static int hmx_mm_id_pipelined(')
b = s.index('#endif  // GGML_HEXAGON_MMID_PIPE', a)
new_fn = r'''static int hmx_mm_id_pipelined(
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
    uint32_t tile_sz[4], tile_al[4];
    const int n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    uint32_t max_al = 0;
    for (int p = 0; p < nw; ++p) {
        if (!srcw[p] || !dstw[p]) return 1;
        if ((int) srcw[p]->ne[0] != k || (int) srcw[p]->ne[1] != n || (int) srcw[p]->ne[2] != n_as) return 1;
        if (!hex_is_aligned((const void *) dstw[p]->data, VLEN)) return 1;
        dq_fn[p] = hmx_mm_id_dequant_fn((int) srcw[p]->type);
        if (!dq_fn[p]) return 1;
        row_stride[p] = htp_mm_get_tiled_row_stride((int) srcw[p]->type, k);
        if (row_stride[p] == 0) return 1;
        tile_sz[p] = htp_mm_get_weight_tile_size((int) srcw[p]->type);
        tile_al[p] = htp_mm_get_weight_aligned_tile_size((int) srcw[p]->type);
        if (tile_al[p] > max_al) max_al = tile_al[p];
    }

    // Column chunk: the whole n if the double-buffered raw+fp16 weight tiles leave room for >= 32 rows (unchanged behavior
    // for shapes that already fit), otherwise the largest 32-multiple that leaves room for >= 64 rows.
    int nch = n;
    int m_max = 0;
    size_t raw_bytes = 0, wf_bytes = 0;
    for (int cand = n; cand >= 32; cand -= 32) {
        const size_t rb = hex_align_up((size_t) (cand / 32) * n_k_tiles * max_al, HTP_MM_HMX_TILE_SIZE);
        const size_t wb = hex_align_up((size_t) cand * k * sizeof(__fp16), HTP_MM_HMX_TILE_SIZE);
        const size_t fixed = 2 * rb + 2 * wb + 8 * HTP_MM_HMX_TILE_SIZE;
        const size_t per_row = 2 * (size_t) k * sizeof(__fp16) + 2 * (size_t) cand * sizeof(__fp16);
        if (fixed >= ctx->vtcm_size) continue;
        int mm = (int) ((ctx->vtcm_size - fixed) / per_row) & ~31;
        if (mm > 256) mm = 256;
        const int need = (cand == n) ? 32 : 64;
        if (mm >= need) { nch = cand; m_max = mm; raw_bytes = rb; wf_bytes = wb; break; }
    }
    if (m_max < 32) return 1;
    const int nc = (n + nch - 1) / nch;

    const uint32_t * counts = mmctx->matrix_row_counts;
    uint16_t it_e[1024], it_r0[1024], it_rows[1024];
    int ne = 0;
    for (int e = 0; e < n_as; ++e) {
        const int c = (int) counts[e];
        for (int r0 = 0; r0 < c; r0 += m_max) {
            if (ne >= 1024) return 1;   // not applicable, no work done yet
            it_e[ne] = (uint16_t) e; it_r0[ne] = (uint16_t) r0; it_rows[ne] = (uint16_t) hex_smin(m_max, c - r0);
            ne++;
        }
    }
    *m_max_out = 0x7fffffff;             // every expert handled here
    if (ne == 0) return 0;

    uint8_t * vp = (uint8_t *) ctx->vtcm_base;
    void   * raw[2]  = { vtcm_seq_alloc(&vp, raw_bytes), vtcm_seq_alloc(&vp, raw_bytes) };
    __fp16 * wf[2]   = { (__fp16 *) vtcm_seq_alloc(&vp, wf_bytes), (__fp16 *) vtcm_seq_alloc(&vp, wf_bytes) };
    const size_t act_bytes = hex_align_up((size_t) m_max * k * sizeof(__fp16), HTP_MM_HMX_TILE_SIZE);
    const size_t out_bytes = hex_align_up((size_t) m_max * nch * sizeof(__fp16), HTP_MM_HMX_TILE_SIZE);
    __fp16 * actb[2] = { (__fp16 *) vtcm_seq_alloc(&vp, act_bytes), (__fp16 *) vtcm_seq_alloc(&vp, act_bytes) };
    __fp16 * outb[2] = { (__fp16 *) vtcm_seq_alloc(&vp, out_bytes), (__fp16 *) vtcm_seq_alloc(&vp, out_bytes) };
    __fp16 * scales  = (__fp16 *) vtcm_seq_alloc(&vp, 256);
    if ((size_t) (vp - (uint8_t *) ctx->vtcm_base) > ctx->vtcm_size) { *m_max_out = 0; return 1; }
    hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));

    const struct fastdiv_values n_k_tiles_div = init_fastdiv_values(n_k_tiles);
    const int per_e = nw * nc;           // items per expert row-chunk
    const int N = ne * per_e;
    const int nthr = (int) octx->n_threads;
    hmx_matmul_job_t jobs[2];

#define PF_CHUNK_COLS(c_) hex_smin(nch, n - (c_) * nch)
#define PF_ITEM_TILES(c_) ((PF_CHUNK_COLS(c_) / HTP_MM_HMX_TILE_N_COLS) * n_k_tiles)
#define PF_TPT(tiles_)    (nthr == 1 ? (tiles_) : hmx_ceil_div((tiles_), nthr))
#define PF_SRC(i_, out_p_, out_c_) ({                                                                            \
        const int ei__ = (i_) / per_e, rr__ = (i_) % per_e; (out_p_) = rr__ / nc; (out_c_) = rr__ % nc;         \
        srcw[(out_p_)]->data + (size_t) it_e[ei__] * srcw[(out_p_)]->nb[2] +                                    \
            (size_t) (((out_c_) * nch) / HTP_MM_HMX_TILE_N_COLS) * n_k_tiles * tile_sz[(out_p_)]; })

    struct mmid_pipe_dq_state dqs;
    memset(&dqs, 0, sizeof(dqs));
    dqs.dq.k_block = k;
    dqs.dq.n_k_tiles = n_k_tiles;
    dqs.dq.n_k_tiles_div = n_k_tiles_div;
    dqs.dq.traces = ctx->trace;
    dqs.dq.ctx = ctx;

    // Prologue: every thread pushes its slice of item 0.
    {
        struct mmid_pipe_dq_state pro = dqs;
        int p0, c0;
        pro.push_next = true;
        pro.next_raw = (uint8_t *) raw[0];
        pro.next_src = PF_SRC(0, p0, c0);
        pro.next_tile = tile_sz[p0];
        pro.next_aligned = tile_al[p0];
        pro.next_tot = PF_ITEM_TILES(c0);
        pro.next_tpt = PF_TPT(pro.next_tot);
        pro.dq.n_tot_tiles = pro.next_tot;
        pro.dq.n_tiles_per_task = pro.next_tpt;
        const int n_tasks0 = (pro.next_tot + pro.next_tpt - 1) / pro.next_tpt;
        pro.dq.n_tasks = n_tasks0;
        if (n_tasks0 == 1) mmid_pipe_dq_worker(1, 0, &pro);
        else worker_pool_run_func(ctx->worker_pool, mmid_pipe_dq_worker, &pro, n_tasks0);
    }
    for (int i = 0; i < N; ++i) {
        const int ei = i / per_e;
        int p, c;
        (void) PF_SRC(i, p, c);
        const int cur_a = it_e[ei];
        const int cne1  = (int) it_rows[ei];
        const int r0    = (int) it_r0[ei];
        const int m_pad = hex_align_up(cne1, 32);
        const int ty    = (int) srcw[p]->type;
        const int cols  = PF_CHUNK_COLS(c);
        const int tiles = PF_ITEM_TILES(c);
        const int tpt   = PF_TPT(tiles);

        dqs.dq.dst = wf[i & 1];
        dqs.dq.src = (const uint8_t *) raw[i & 1];
        dqs.dq.row_stride = row_stride[p];
        dqs.dq.weight_type = ty;
        dqs.dq.tile_size = tile_sz[p];
        dqs.dq.aligned_tile_size = tile_al[p];
        dqs.dq.n_cols = cols;
        dqs.dq.n_tot_tiles = tiles;
        dqs.dq.n_tiles_per_task = tpt;
        dqs.dq.n_tasks = (tiles + tpt - 1) / tpt;
        dqs.fn = dq_fn[p];
        dqs.pop_cur = true;
        dqs.push_next = (i + 1 < N);
        if (dqs.push_next) {
            int pn, cn;
            dqs.next_src = PF_SRC(i + 1, pn, cn);
            dqs.next_raw = (uint8_t *) raw[(i + 1) & 1];
            dqs.next_tile = tile_sz[pn];
            dqs.next_aligned = tile_al[pn];
            dqs.next_tot = PF_ITEM_TILES(cn);
            dqs.next_tpt = PF_TPT(dqs.next_tot);
        }
        dqs.n_scatter = 0;
        if (i >= 2) {   // HMX(i-2) was popped at the end of iteration i-1
            const int j = i - 2, ej = j / per_e;
            int pj, cj;
            (void) PF_SRC(j, pj, cj);
            mmid_pipe_scatter_state(&dqs.os[0], ctx, (float *) dstw[pj]->data + (size_t) cj * nch, outb[j & 1], (int) it_rows[ej], (int) it_r0[ej],
                                    PF_CHUNK_COLS(cj), mmctx->matrix_rows, it_e[ej], mmctx->mapping_stride, dstw[pj]->nb[1], dstw[pj]->nb[2], nthr);
            dqs.n_scatter = 1;
        }
        dqs.do_gather = (p == 0 && c == 0);
        if (dqs.do_gather) {
            dqs.gfn = mmid_pipe_gather_state(&dqs.gs, ctx, actb[ei & 1], (const float *) act->data, cne1, r0, k,
                                             mmctx->matrix_rows, cur_a, mmctx->mapping_stride, (int) act->ne[1],
                                             act->nb[1], act->nb[2], (int) act->ne[0],
                                             hmx_use_q8_0_activation_reference(ty), nthr);
        }
        if (nthr == 1) mmid_pipe_dq_worker(1, 0, &dqs);
        else worker_pool_run_func(ctx->worker_pool, mmid_pipe_dq_worker, &dqs, nthr);

        hmx_matmul_job_init(&jobs[i & 1], outb[i & 1], actb[ei & 1], wf[i & 1], scales,
                            m_pad / HTP_MM_HMX_TILE_N_ROWS, cols / HTP_MM_HMX_TILE_N_COLS, k / HTP_MM_HMX_TILE_N_ROWS);
        hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &jobs[i & 1]));
        if (i > 0) hmx_queue_pop(ctx->hmx_queue);   // HMX(i-1) done
    }
    hmx_queue_pop(ctx->hmx_queue);                   // HMX(N-1) done
    {
        // Drain: scatter the last two items in one dispatch.
        dqs.fn = NULL; dqs.pop_cur = false; dqs.push_next = false; dqs.do_gather = false; dqs.n_scatter = 0;
        dqs.dq.n_tot_tiles = 1; dqs.dq.n_tiles_per_task = 1;
        for (int j = (N >= 2 ? N - 2 : 0); j < N; ++j) {
            const int ej = j / per_e;
            int pj, cj;
            (void) PF_SRC(j, pj, cj);
            mmid_pipe_scatter_state(&dqs.os[dqs.n_scatter++], ctx, (float *) dstw[pj]->data + (size_t) cj * nch, outb[j & 1], (int) it_rows[ej], (int) it_r0[ej],
                                    PF_CHUNK_COLS(cj), mmctx->matrix_rows, it_e[ej], mmctx->mapping_stride, dstw[pj]->nb[1], dstw[pj]->nb[2], nthr);
        }
        if (nthr == 1) mmid_pipe_dq_worker(1, 0, &dqs);
        else worker_pool_run_func(ctx->worker_pool, mmid_pipe_dq_worker, &dqs, nthr);
    }
#undef PF_SRC
#undef PF_TPT
#undef PF_ITEM_TILES
#undef PF_CHUNK_COLS
    return 0;
}
'''
s = s[:a] + new_fn + s[b:]
if crlf: s = s.replace('\n', '\r\n')
open(p, 'w', newline='').write(s)
print('patched nchunk')
