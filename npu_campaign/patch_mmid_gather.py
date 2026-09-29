"""Patch 5 (after patch_mmid_chunk.py): exact-range activation gather for the MoE pipeline.
The stock gathered worker pads each thread's n_rows to 32 and converts every valid row from its start onward, so with
~16 rows/expert thread t converts (16-2t) rows (~4.5x redundant). This worker converts each row exactly once and
zero-fills tile padding rows itself."""
import os
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'llama.cpp-hexagon', 'ggml', 'src', 'ggml-hexagon', 'htp')
p = os.path.join(root, 'matmul-ops.c')
s = open(p, newline='').read().replace('\r\n', '\n')
if 'mmid_gather_worker_exact' in s:
    print('already applied'); raise SystemExit
assert 'mmid_pipe_gather_state' in s and 'it_r0[' in s, 'apply fuse+chunk patches first'

worker = r'''// Exact-range gather: thread ith owns local rows [ith*cpt, (ith+1)*cpt) of the m_pad-row tile block.
static void mmid_gather_worker_exact(unsigned int n, unsigned int ith, void * data) {
    activation_transfer_gathered_task_state_t * st = (activation_transfer_gathered_task_state_t *) data;
    const uint32_t cpt = st->n_chunks_per_task;
    const uint32_t lr0 = ith * cpt;
    if (lr0 >= st->n_tot_chunks) return;
    const uint32_t lr1 = hex_smin(lr0 + cpt, st->n_tot_chunks);
    struct htp_thread_trace * tr = &st->traces[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, ith);

    const uint32_t k_block = st->k_block, k_valid = st->k_valid;
    const uint32_t n_k_tiles = k_block / HTP_MM_HMX_TILE_N_COLS;
    const bool flat = st->ne11 <= 1;
    for (uint32_t lr = lr0; lr < lr1; lr += 2) {
        const uint32_t g0 = st->start_row + lr, g1 = g0 + 1;
        const bool v0 = g0 < st->cne1, v1 = g1 < st->cne1;
        const float * p0 = NULL, * p1 = NULL;
        if (v0) {
            const struct mmid_row_mapping m = st->matrix_rows[st->cur_a * st->mapping_stride + g0];
            const uint32_t i11 = flat ? 0 : fastmodulo(m.i1, st->ne11, &st->ne11_div);
            p0 = (const float *) ((const uint8_t *) st->src + i11 * st->nb11 + m.i2 * st->nb12);
        }
        if (v1) {
            const struct mmid_row_mapping m = st->matrix_rows[st->cur_a * st->mapping_stride + g1];
            const uint32_t i11 = flat ? 0 : fastmodulo(m.i1, st->ne11, &st->ne11_div);
            p1 = (const float *) ((const uint8_t *) st->src + i11 * st->nb11 + m.i2 * st->nb12);
        }
        const uint32_t tr0 = lr / HTP_MM_HMX_TILE_N_ROWS;   // tile row index
        const uint32_t tr1 = lr % HTP_MM_HMX_TILE_N_ROWS;   // intra-tile row index
        uint32_t c = 0;
        for (; c + 32 <= k_valid; c += 32) {
            HVX_Vector a = v0 ? hmx_reference_activation_load(p0 + c, st->reference_q8_0) : Q6_V_vzero();
            HVX_Vector b = v1 ? hmx_reference_activation_load(p1 + c, st->reference_q8_0) : Q6_V_vzero();
            HVX_Vector * tile = (HVX_Vector *) (st->dst + (tr0 * n_k_tiles + c / HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_N_ELMS);
            tile[tr1 / 2] = hvx_vec_f32_to_f16_shuff(a, b);
        }
        if (c < k_block) {
            HVX_Vector a = v0 ? hmx_reference_activation_load(p0 + c, st->reference_q8_0) : Q6_V_vzero();
            HVX_Vector b = v1 ? hmx_reference_activation_load(p1 + c, st->reference_q8_0) : Q6_V_vzero();
            const uint32_t rem = k_valid - c;
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem > 0 ? rem * sizeof(float) : 0);
            a = Q6_V_vmux_QVV(mask, a, Q6_V_vzero());
            b = Q6_V_vmux_QVV(mask, b, Q6_V_vzero());
            HVX_Vector * tile = (HVX_Vector *) (st->dst + (tr0 * n_k_tiles + c / HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_N_ELMS);
            tile[tr1 / 2] = hvx_vec_f32_to_f16_shuff(a, b);
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, ith);
}

'''
anchor = 'static inline void mmid_pipe_scatter_state('
assert anchor in s
s = s.replace(anchor, worker + anchor, 1)
old = '    return ne11 == 1 ? transfer_activation_chunk_gathered_worker_flat_fn : transfer_activation_chunk_gathered_worker_fn;'
assert old in s
s = s.replace(old, '    return mmid_gather_worker_exact;', 1)
open(p, 'w', newline='').write(s)
print('patched exact gather')
