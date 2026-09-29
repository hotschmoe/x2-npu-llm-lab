"""Patch 3 (apply after patch_mmid_fuse.py): split experts with > m_max rows into row-chunk items inside the
pipeline instead of falling back to the legacy serial path. Items are (expert, row0, rows)."""
import os
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'llama.cpp-hexagon', 'ggml', 'src', 'ggml-hexagon', 'htp')
p = os.path.join(root, 'matmul-ops.c')
s = open(p, newline='').read().replace('\r\n', '\n')
if 'it_r0[' in s:
    print('already applied'); raise SystemExit
assert 'mmid_pipe_scatter_state' in s, 'apply patch_mmid_fuse.py first'

def rep(old, new, count=1):
    global s
    assert s.count(old) >= 1, 'missing: ' + old[:70]
    s = s.replace(old, new) if count == 0 else s.replace(old, new, count)

# helper signatures: add r0 after cne1 (cne1 becomes the chunk row count)
rep('''        float * dst, const __fp16 * vtcm_src, int cne1, int n_cols, const struct mmid_row_mapping * rows,''',
    '''        float * dst, const __fp16 * vtcm_src, int cne1, int r0, int n_cols, const struct mmid_row_mapping * rows,''')
rep('''    os->start_row = 0; os->cne1 = cne1;''', '''    os->start_row = r0; os->cne1 = r0 + cne1;''')
rep('''        __fp16 * dst, const float * src, int cne1, int k, const struct mmid_row_mapping * rows, int cur_a, int stride,''',
    '''        __fp16 * dst, const float * src, int cne1, int r0, int k, const struct mmid_row_mapping * rows, int cur_a, int stride,''')
rep('''gs->nb12 = nb12; gs->start_row = 0; gs->cne1 = cne1; gs->k_valid = k_valid;''',
    '''gs->nb12 = nb12; gs->start_row = r0; gs->cne1 = r0 + cne1; gs->k_valid = k_valid;''')

# item list instead of expert list
rep('''    uint16_t experts[512];
    int ne = 0;
    for (int e = 0; e < n_as && ne < 512; ++e) {
        if (counts[e] > 0 && (int) counts[e] <= m_max) experts[ne++] = (uint16_t) e;
    }
    *m_max_out = m_max;
    if (ne == 0) return 0;
''', '''    uint16_t it_e[1024], it_r0[1024], it_rows[1024];
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
''')
rep('experts[0] * srcw[0]->nb[2]', 'it_e[0] * srcw[0]->nb[2]')
rep('experts[en] * srcw[pn]->nb[2]', 'it_e[en] * srcw[pn]->nb[2]')
rep('''        const int cur_a = experts[ei];
        const int cne1  = (int) counts[cur_a];''', '''        const int cur_a = it_e[ei];
        const int cne1  = (int) it_rows[ei];
        const int r0    = (int) it_r0[ei];''')
rep('''            const int j = i - 2, ej = j / nw, pj = j % nw, aj = experts[ej];
            mmid_pipe_scatter_state(&dqs.os[0], ctx, (float *) dstw[pj]->data, outb[j & 1], (int) counts[aj], n,''',
    '''            const int j = i - 2, ej = j / nw, pj = j % nw, aj = it_e[ej];
            mmid_pipe_scatter_state(&dqs.os[0], ctx, (float *) dstw[pj]->data, outb[j & 1], (int) it_rows[ej], (int) it_r0[ej], n,''')
rep('''gfn = mmid_pipe_gather_state(&dqs.gs, ctx, actb[ei & 1], (const float *) act->data, cne1, k,''',
    '''gfn = mmid_pipe_gather_state(&dqs.gs, ctx, actb[ei & 1], (const float *) act->data, cne1, r0, k,''')
rep('''            const int ej = j / nw, pj = j % nw, aj = experts[ej];
            mmid_pipe_scatter_state(&dqs.os[dqs.n_scatter++], ctx, (float *) dstw[pj]->data, outb[j & 1], (int) counts[aj], n,''',
    '''            const int ej = j / nw, pj = j % nw, aj = it_e[ej];
            mmid_pipe_scatter_state(&dqs.os[dqs.n_scatter++], ctx, (float *) dstw[pj]->data, outb[j & 1], (int) it_rows[ej], (int) it_r0[ej], n,''')
assert 'experts[' not in s[s.index('static int hmx_mm_id_pipelined'):s.index('#endif  // GGML_HEXAGON_MMID_PIPE')], 'leftover experts[]'
open(p, 'w', newline='').write(s)
print('patched chunk')
