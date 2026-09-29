"""DSP-side flatten of contiguous F32 elementwise unary ops with tiny rows (SIGMOID [1,32,T], SCALE [128,..], SILU [128,..], ...):
the row-oriented executor pays DMA + dispatch per row. Present the tensors as rows of 1024 elements for the duration of the op,
recompute the tiling params locally (port of ggml_hexagon_precompute_unary_params), restore afterwards. Guarded by
GGML_HEXAGON_UNARY_FLATTEN (CMake option, default ON)."""
import os
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'llama.cpp-hexagon', 'ggml', 'src', 'ggml-hexagon', 'htp')
p = os.path.join(root, 'unary-ops.c')
s = open(p, newline='').read()
crlf = '\r\n' in s
s = s.replace('\r\n', '\n')
if 'execute_op_unary_impl' in s:
    print('already applied'); raise SystemExit

old_sig = 'static int execute_op_unary(struct htp_ops_context * octx) {'
assert s.count(old_sig) == 1
s = s.replace(old_sig, 'static int execute_op_unary_impl(struct htp_ops_context * octx) {', 1)

wrapper = r'''
#ifdef GGML_HEXAGON_UNARY_FLATTEN
#define UNARY_FLAT_ROW 1024u

static bool unary_flat_op_ok(uint32_t op) {
    switch (op) {
        case HTP_OP_SCALE:
        case HTP_OP_CLAMP:
        case HTP_OP_LEAKY_RELU:
        case HTP_OP_SQR:
        case HTP_OP_SQRT:
        case HTP_OP_UNARY_NEG:
        case HTP_OP_UNARY_EXP:
        case HTP_OP_UNARY_SIGMOID:
        case HTP_OP_UNARY_SILU:
        case HTP_OP_UNARY_GELU:
        case HTP_OP_UNARY_SOFTPLUS:
        case HTP_OP_UNARY_TANH:
        case HTP_OP_UNARY_ABS:
        case HTP_OP_UNARY_LOG:
        case HTP_OP_UNARY_RELU:
            return true;
        default:
            return false;
    }
}

static bool unary_flat_contig(const struct htp_tensor * t) {
    return t->nb[0] == sizeof(float) && t->nb[1] == t->ne[0] * t->nb[0] &&
           t->nb[2] == t->nb[1] * t->ne[1] && t->nb[3] == t->nb[2] * t->ne[2];
}

static int execute_op_unary(struct htp_ops_context * octx) {
    struct htp_tensor * src0 = (struct htp_tensor *) octx->src[0];
    struct htp_tensor * dst  = (struct htp_tensor *) octx->dst;

    if (!unary_flat_op_ok(octx->op) || !src0 || !dst || octx->src[1] || octx->ctx->mdev.count > 1 ||
        src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32 ||
        src0->ne[0] >= UNARY_FLAT_ROW / 2 || dst->ne[0] != src0->ne[0] || dst->ne[1] != src0->ne[1] ||
        dst->ne[2] != src0->ne[2] || dst->ne[3] != src0->ne[3] ||
        !unary_flat_contig(src0) || !unary_flat_contig(dst)) {
        return execute_op_unary_impl(octx);
    }
    const uint32_t total = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    if (total == 0 || (total % UNARY_FLAT_ROW) != 0 || (uintptr_t) src0->data % 128 != 0 || (uintptr_t) dst->data % 128 != 0) {
        return execute_op_unary_impl(octx);
    }

    // recompute the tiling params for the flat shape (port of the host precompute)
    struct htp_unary_kernel_params kp;
    memset(&kp, 0, sizeof(kp));
    const uint32_t nrows = total / UNARY_FLAT_ROW;
    const uint32_t nthr  = MIN((uint32_t) octx->ctx->n_threads, nrows);
    const size_t   esz   = sizeof(float);
    const size_t   rowb  = UNARY_FLAT_ROW * esz;
    kp.n_threads              = nthr;
    kp.src0_row_size_aligned  = hex_round_up(rowb, 128);
    kp.dst_row_size_aligned   = hex_round_up(rowb, 128);
    struct htp_unary_vtcm_layout L;
    uint32_t col_tile = 0, vtcm_row_per_thread = 0;
    htp_unary_vtcm_layout_build(&L, octx->op, UNARY_FLAT_ROW, UNARY_FLAT_ROW, 0, false, nthr, octx->ctx->vtcm_size, esz,
                                &col_tile, &vtcm_row_per_thread);
    kp.col_tile                  = col_tile;
    kp.vtcm_row_per_thread       = vtcm_row_per_thread;
    kp.vtcm_size                 = L.total_bytes;
    kp.vtcm_src0_size_per_thread = L.src0_bytes;
    kp.vtcm_src1_size_per_thread = L.src1_bytes;
    kp.vtcm_dst_size_per_thread  = L.dst_bytes;
    kp.vtcm_src0_size            = L.src0_bytes * nthr;
    kp.vtcm_src1_size            = L.src1_bytes * nthr;
    kp.vtcm_dst_size             = L.dst_bytes * nthr;
    kp.block                     = col_tile ? 0 : ((L.src0_bytes / 2) / kp.src0_row_size_aligned);
    const uint32_t tiles_per_row = col_tile > 0 ? (UNARY_FLAT_ROW + col_tile - 1) / col_tile : 1;
    kp.div_ne01  = init_fastdiv_values(nrows);
    kp.div_ne02  = init_fastdiv_values(1);
    kp.div_ne012 = init_fastdiv_values(nrows);
    kp.div_tpr   = init_fastdiv_values(tiles_per_row);
    if (kp.block == 0 && col_tile == 0) {
        return execute_op_unary_impl(octx);
    }

    uint32_t s_ne[4], d_ne[4], s_nb[4], d_nb[4];
    memcpy(s_ne, src0->ne, sizeof(s_ne)); memcpy(d_ne, dst->ne, sizeof(d_ne));
    memcpy(s_nb, src0->nb, sizeof(s_nb)); memcpy(d_nb, dst->nb, sizeof(d_nb));
    int32_t saved_kp[HTP_OP_MAX_KERN_PARAMS];
    memcpy(saved_kp, octx->kernel_params, sizeof(saved_kp));

    struct htp_tensor * ts[2] = { src0, dst };
    for (int i = 0; i < 2; ++i) {
        ts[i]->ne[0] = UNARY_FLAT_ROW; ts[i]->ne[1] = nrows; ts[i]->ne[2] = 1; ts[i]->ne[3] = 1;
        ts[i]->nb[0] = (uint32_t) esz;  ts[i]->nb[1] = (uint32_t) rowb;
        ts[i]->nb[2] = (uint32_t) rowb * nrows; ts[i]->nb[3] = ts[i]->nb[2];
    }
    memcpy(octx->kernel_params, &kp, sizeof(kp));

    const int st = execute_op_unary_impl(octx);

    memcpy(src0->ne, s_ne, sizeof(s_ne)); memcpy(dst->ne, d_ne, sizeof(d_ne));
    memcpy(src0->nb, s_nb, sizeof(s_nb)); memcpy(dst->nb, d_nb, sizeof(d_nb));
    memcpy(octx->kernel_params, saved_kp, sizeof(saved_kp));
    return st;
}
#else
static int execute_op_unary(struct htp_ops_context * octx) {
    return execute_op_unary_impl(octx);
}
#endif  // GGML_HEXAGON_UNARY_FLATTEN
'''
# insert the wrapper right after the end of execute_op_unary_impl
i0 = s.index('static int execute_op_unary_impl(struct htp_ops_context * octx) {')
i1 = s.index('\n}\n', i0) + 3
s = s[:i1] + wrapper + s[i1:]
if crlf: s = s.replace('\n', '\r\n')
open(p, 'w', newline='').write(s)

p = os.path.join(root, 'CMakeLists.txt')
c = open(p, newline='').read()
nl = '\r\n' if '\r\n' in c else '\n'
a = 'option(GGML_HEXAGON_MMID_PIPE '
assert a in c
c = c.replace(a, 'option(GGML_HEXAGON_UNARY_FLATTEN "Present contiguous tiny-row elementwise unary ops as 1024-wide rows" ON)' + nl + a, 1)
b = '    $<$<BOOL:${GGML_HEXAGON_MMID_PIPE}>:GGML_HEXAGON_MMID_PIPE=1>'
assert b in c
c = c.replace(b, b + nl + '    $<$<BOOL:${GGML_HEXAGON_UNARY_FLATTEN}>:GGML_HEXAGON_UNARY_FLATTEN=1>', 1)
open(p, 'w', newline='').write(c)
print('patched unary flatten')
