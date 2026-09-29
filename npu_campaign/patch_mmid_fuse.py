"""Patch 2: fuse scatter(i-2) + gather + DMA + dequant into ONE worker dispatch per (expert, weight) item."""
import os, re
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'llama.cpp-hexagon', 'ggml', 'src', 'ggml-hexagon', 'htp')
p = os.path.join(root, 'matmul-ops.c')
s = open(p, newline='').read().replace('\r\n', '\n')
if 'mmid_pipe_scatter_state' in s:
    print('already applied'); raise SystemExit

# ---- 1. replace struct + worker
a = s.index('struct mmid_pipe_dq_state {')
b = s.index('// Cross-expert software pipeline for prefill MoE (MUL_MAT_ID / _NX).')
new_worker = r'''struct mmid_pipe_dq_state {
    tiled_dequantize_state_t dq;
    worker_callback_t        fn;        // NULL: prologue (push only)
    bool                     pop_cur;
    bool                     push_next;
    uint8_t *                next_raw;
    dma_addr_t               next_src;
    uint32_t                 next_tile;
    uint32_t                 next_aligned;
    int                      n_scatter;
    output_transfer_scattered_task_state_t os[2];
    bool                     do_gather;
    worker_callback_t        gfn;
    activation_transfer_gathered_task_state_t gs;
};

static void mmid_pipe_dq_worker(unsigned int n, unsigned int ith, void * data) {
    struct mmid_pipe_dq_state * st = (struct mmid_pipe_dq_state *) data;
    const int start = (int) ith * st->dq.n_tiles_per_task;
    const int end   = hex_smin(start + st->dq.n_tiles_per_task, st->dq.n_tot_tiles);
    dma_queue * q = st->dq.ctx->dma[ith];
    if (start < end && st->push_next) {
        dma_queue_push(q, dma_make_data(st->next_raw + (size_t) start * st->next_aligned,
                                        st->next_src + (dma_addr_t) start * st->next_tile),
                       st->next_aligned, st->next_tile, st->next_tile, (size_t) (end - start));
    }
    for (int j = 0; j < st->n_scatter; ++j) transfer_output_chunk_scattered_worker_fn(n, ith, &st->os[j]);
    if (st->do_gather) st->gfn(n, ith, &st->gs);
    if (start < end && st->pop_cur) dma_queue_pop(q);
    if (st->fn) st->fn(n, ith, &st->dq);
}

static inline void mmid_pipe_scatter_state(output_transfer_scattered_task_state_t * os, struct htp_context * ctx,
        float * dst, const __fp16 * vtcm_src, int cne1, int n_cols, const struct mmid_row_mapping * rows,
        int cur_a, int stride, size_t nb1, size_t nb2, int n_threads) {
    const int n_rows = hex_align_up(cne1, 32);
    const int cpt    = hex_align_up(hmx_ceil_div(n_rows, n_threads), 2);
    memset(os, 0, sizeof(*os));
    os->traces = ctx->trace; os->matrix_rows = rows; os->vtcm_src = vtcm_src; os->dst = dst;
    os->n_tasks = hmx_ceil_div(n_rows, cpt); os->n_tot_chunks = n_rows; os->n_chunks_per_task = cpt;
    os->n_cols = n_cols; os->cur_a = cur_a; os->mapping_stride = stride; os->dst_nb1 = nb1; os->dst_nb2 = nb2;
    os->start_row = 0; os->cne1 = cne1;
}

static inline worker_callback_t mmid_pipe_gather_state(activation_transfer_gathered_task_state_t * gs, struct htp_context * ctx,
        __fp16 * dst, const float * src, int cne1, int k, const struct mmid_row_mapping * rows, int cur_a, int stride,
        int ne11, size_t nb11, size_t nb12, int k_valid, bool ref_q8, int n_threads) {
    const int n_rows = hex_align_up(cne1, 32);
    const int cpt    = hex_align_up(hmx_ceil_div(n_rows, n_threads), 2);
    memset(gs, 0, sizeof(*gs));
    gs->traces = ctx->trace; gs->matrix_rows = rows; gs->dst = dst; gs->src = src;
    gs->n_tasks = hmx_ceil_div(n_rows, cpt); gs->n_tot_chunks = n_rows; gs->n_chunks_per_task = cpt;
    gs->k_block = k; gs->cur_a = cur_a; gs->mapping_stride = stride; gs->ne11 = ne11;
    gs->ne11_div = ne11 > 1 ? init_fastdiv_values(ne11) : (struct fastdiv_values){0, 0};
    gs->reference_q8_0 = ref_q8; gs->nb11 = nb11; gs->nb12 = nb12; gs->start_row = 0; gs->cne1 = cne1; gs->k_valid = k_valid;
    return ne11 == 1 ? transfer_activation_chunk_gathered_worker_flat_fn : transfer_activation_chunk_gathered_worker_fn;
}

'''
s = s[:a] + new_worker + s[b:]

# ---- 2. replace the main loop (from item loop to end of function)
c = s.index('    for (int i = 0; i < N; ++i) {\n        const int ei = i / nw, p = i % nw;')
d = s.index('    return 0;\n}\n#endif  // GGML_HEXAGON_MMID_PIPE')
new_loop = r'''    const int nthr = (int) octx->n_threads;
    for (int i = 0; i < N; ++i) {
        const int ei = i / nw, p = i % nw;
        const int cur_a = experts[ei];
        const int cne1  = (int) counts[cur_a];
        const int m_pad = hex_align_up(cne1, 32);
        const int ty    = (int) srcw[p]->type;

        dqs.dq.dst = wf[i & 1];
        dqs.dq.src = (const uint8_t *) raw[i & 1];
        dqs.dq.row_stride = row_stride[p];
        dqs.dq.weight_type = ty;
        dqs.dq.tile_size = htp_mm_get_weight_tile_size(ty);
        dqs.dq.aligned_tile_size = htp_mm_get_weight_aligned_tile_size(ty);
        dqs.fn = dq_fn[p];
        dqs.pop_cur = true;
        dqs.push_next = (i + 1 < N);
        if (dqs.push_next) {
            const int pn = (i + 1) % nw, en = (i + 1) / nw, tyn = (int) srcw[pn]->type;
            dqs.next_raw = (uint8_t *) raw[(i + 1) & 1];
            dqs.next_src = srcw[pn]->data + (size_t) experts[en] * srcw[pn]->nb[2];
            dqs.next_tile = htp_mm_get_weight_tile_size(tyn);
            dqs.next_aligned = htp_mm_get_weight_aligned_tile_size(tyn);
        }
        dqs.n_scatter = 0;
        if (i >= 2) {   // HMX(i-2) was popped at the end of iteration i-1
            const int j = i - 2, ej = j / nw, pj = j % nw, aj = experts[ej];
            mmid_pipe_scatter_state(&dqs.os[0], ctx, (float *) dstw[pj]->data, outb[j & 1], (int) counts[aj], n,
                                    mmctx->matrix_rows, aj, mmctx->mapping_stride, dstw[pj]->nb[1], dstw[pj]->nb[2], nthr);
            dqs.n_scatter = 1;
        }
        dqs.do_gather = (p == 0);
        if (dqs.do_gather) {
            dqs.gfn = mmid_pipe_gather_state(&dqs.gs, ctx, actb[ei & 1], (const float *) act->data, cne1, k,
                                             mmctx->matrix_rows, cur_a, mmctx->mapping_stride, (int) act->ne[1],
                                             act->nb[1], act->nb[2], (int) act->ne[0],
                                             hmx_use_q8_0_activation_reference(ty), nthr);
        }
        if (nthr == 1) mmid_pipe_dq_worker(1, 0, &dqs);
        else worker_pool_run_func(ctx->worker_pool, mmid_pipe_dq_worker, &dqs, nthr);

        hmx_matmul_job_init(&jobs[i & 1], outb[i & 1], actb[ei & 1], wf[i & 1], scales,
                            m_pad / HTP_MM_HMX_TILE_N_ROWS, n / HTP_MM_HMX_TILE_N_COLS, k / HTP_MM_HMX_TILE_N_ROWS);
        hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &jobs[i & 1]));
        if (i > 0) hmx_queue_pop(ctx->hmx_queue);   // HMX(i-1) done
    }
    hmx_queue_pop(ctx->hmx_queue);                   // HMX(N-1) done
    {
        // Drain: scatter the last two items in one dispatch.
        dqs.fn = NULL; dqs.pop_cur = false; dqs.push_next = false; dqs.do_gather = false; dqs.n_scatter = 0;
        for (int j = (N >= 2 ? N - 2 : 0); j < N; ++j) {
            const int ej = j / nw, pj = j % nw, aj = experts[ej];
            mmid_pipe_scatter_state(&dqs.os[dqs.n_scatter++], ctx, (float *) dstw[pj]->data, outb[j & 1], (int) counts[aj], n,
                                    mmctx->matrix_rows, aj, mmctx->mapping_stride, dstw[pj]->nb[1], dstw[pj]->nb[2], nthr);
        }
        if (nthr == 1) mmid_pipe_dq_worker(1, 0, &dqs);
        else worker_pool_run_func(ctx->worker_pool, mmid_pipe_dq_worker, &dqs, nthr);
    }

'''
s = s[:c] + new_loop + s[d:]
open(p, 'w', newline='').write(s)
print('patched fuse')
