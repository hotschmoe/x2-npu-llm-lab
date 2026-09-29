"""[prefill2 variant: 2x1 dot for the bit-exact reduction tree]
Router refine (GGML_HEXAGON_ROUTER_REFINE): after the HMX (fp16-operand) MUL_MAT for the F32 MoE router
[K=2048 -> N=256], recompute the top-16 candidates of every token exactly in F32 on HVX with the SAME dot helper the HVX router
uses (bit-identical logits for those entries). Non-candidate logits stay fp16-accurate; they only enter the softmax denominator,
which cancels in the normalized top-8 weights. Result: HMX speed with the exact HVX expert selection."""
import os
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'llama.cpp-hexagon-prefill2', 'ggml', 'src', 'ggml-hexagon', 'htp')
p = os.path.join(root, 'matmul-ops.c')
s = open(p, newline='').read()
crlf = '\r\n' in s
s = s.replace('\r\n', '\n')
if 'router_refine_fn' in s:
    print('already applied'); raise SystemExit

code = r'''
#ifdef GGML_HEXAGON_ROUTER_REFINE
#define ROUTER_RF_K 16u
#define ROUTER_RF_KDIM 2048u
#define ROUTER_RF_NEXP 256u

static bool router_refine_applicable(const struct htp_ops_context * octx) {
    const struct htp_tensor * w = octx->src[0];
    const struct htp_tensor * x = octx->src[1];
    const struct htp_tensor * d = octx->dst;
    return w && x && d && !octx->src[2] &&
           w->type == HTP_TYPE_F32 && x->type == HTP_TYPE_F32 && d->type == HTP_TYPE_F32 &&
           w->ne[0] == ROUTER_RF_KDIM && w->ne[1] == ROUTER_RF_NEXP && w->ne[2] == 1 && w->ne[3] == 1 &&
           w->nb[0] == sizeof(float) && w->nb[1] == ROUTER_RF_KDIM * sizeof(float) &&
           x->ne[0] == ROUTER_RF_KDIM && x->ne[1] >= 32 && x->ne[2] == 1 && x->ne[3] == 1 &&
           x->nb[0] == sizeof(float) && x->nb[1] == ROUTER_RF_KDIM * sizeof(float) &&
           d->ne[0] == ROUTER_RF_NEXP && d->ne[1] == x->ne[1] && d->ne[2] == 1 && d->ne[3] == 1 &&
           d->nb[0] == sizeof(float);
}

struct router_refine_ctx {
    struct htp_ops_context * octx;
};

static uint8_t * router_refine_wregion(const struct htp_ops_context * octx) {
    return (uint8_t *) hex_align_up((uintptr_t) octx->ctx->vtcm_base, 128);
}

static void router_refine_wload(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    if (ith != 0) return;
    struct router_refine_ctx * rc = (struct router_refine_ctx *) data;
    struct htp_ops_context * octx = rc->octx;
    const struct htp_tensor * w = octx->src[0];
    uint8_t * dstv = router_refine_wregion(octx);
    dma_queue * q = octx->ctx->dma[0];
    const uint32_t total = ROUTER_RF_NEXP * ROUTER_RF_KDIM * (uint32_t) sizeof(float);
    for (uint32_t off = 0; off < total; off += 65536u) {
        dma_queue_push(q, dma_make_data(dstv + off, w->data + off), 65536u, 65536u, 65536u, 1);
        dma_queue_pop(q);
    }
}

static void router_refine_fn(unsigned int nth, unsigned int ith, void * data) {
    struct router_refine_ctx * rc = (struct router_refine_ctx *) data;
    struct htp_ops_context * octx = rc->octx;
    const struct htp_tensor * x = octx->src[1];
    const struct htp_tensor * d = octx->dst;
    const uint32_t T = (uint32_t) x->ne[1];

    const uint32_t xrow = ROUTER_RF_KDIM * (uint32_t) sizeof(float);
    const uint8_t * W = router_refine_wregion(octx);
    uint8_t * xb[2];
    xb[0] = (uint8_t *) W + ROUTER_RF_NEXP * xrow + (size_t) ith * 2u * xrow;
    xb[1] = xb[0] + xrow;
    dma_queue * q = octx->ctx->dma[ith];

    uint32_t t = ith;
    uint32_t i = 0;
    if (t < T) {
        dma_queue_push(q, dma_make_data(xb[0], x->data + (size_t) t * x->nb[1]), xrow, xrow, xrow, 1);
    }
    for (; t < T; t += nth, ++i) {
        const int b = (int) (i & 1);
        const uint32_t tn = t + nth;
        if (tn < T) {
            dma_queue_push(q, dma_make_data(xb[b ^ 1], x->data + (size_t) tn * x->nb[1]), xrow, xrow, xrow, 1);
        }
        dma_queue_pop(q);                    // x_t ready (the prefetch of x_tn stays queued behind it)

        float * L = (float *) (uintptr_t) (d->data + (size_t) t * d->nb[1]);
        float __attribute__((aligned(128))) lg[ROUTER_RF_NEXP];
        for (uint32_t v = 0; v < ROUTER_RF_NEXP / 32u; ++v) {
            hvx_vec_store_a(lg + v * 32u, 128, *(const HVX_UVector *) (L + v * 32u));
        }

        uint32_t cand[ROUTER_RF_K];
        float    cval[ROUTER_RF_K];
        uint32_t mi = 0;
        for (uint32_t k = 0; k < ROUTER_RF_K; ++k) { cand[k] = k; cval[k] = lg[k]; if (cval[k] < cval[mi]) mi = k; }
        for (uint32_t e = ROUTER_RF_K; e < ROUTER_RF_NEXP; ++e) {
            if (lg[e] > cval[mi]) {
                cand[mi] = e; cval[mi] = lg[e];
                mi = 0;
                for (uint32_t k = 1; k < ROUTER_RF_K; ++k) { if (cval[k] < cval[mi]) mi = k; }
            }
        }

        // Use the 2x1 dot: its lane reduction (hvx_vec_reduce_sum_f32x2 tree) is the one the 2x2 router kernel uses, so the
        // recomputed logits are bit-identical to the HVX router's. (The 1x1 kernel reduces in a different order.)
        for (uint32_t k = 0; k < ROUTER_RF_K; k += 2) {
            float r[2] __attribute__((aligned(8)));
            vec_dot_f32_f32_aa_2x1(ROUTER_RF_KDIM, r, W + (size_t) cand[k] * xrow, W + (size_t) cand[k + 1] * xrow, xb[b]);
            L[cand[k]] = r[0];
            L[cand[k + 1]] = r[1];
        }
    }
    while (dma_queue_depth(q) > 0) { dma_queue_pop(q); }
}

static int router_refine(struct htp_ops_context * octx) {
    const size_t xrow = ROUTER_RF_KDIM * sizeof(float);
    const size_t need = (size_t) ROUTER_RF_NEXP * xrow + (size_t) octx->n_threads * 2u * xrow + 256u;
    if (octx->ctx->vtcm_size < need) {
        return HTP_STATUS_OK;                // not enough VTCM: keep the approximate HMX logits
    }
    struct router_refine_ctx rc = { octx };
    work_queue_run(octx->ctx->work_queue, router_refine_wload, &rc, 1);
    work_queue_run(octx->ctx->work_queue, router_refine_fn, &rc, octx->n_threads);
    return HTP_STATUS_OK;
}
#endif  // GGML_HEXAGON_ROUTER_REFINE

int op_matmul(struct htp_ops_context * octx) {'''
anchor = '\nint op_matmul(struct htp_ops_context * octx) {'
assert anchor in s
s = s.replace(anchor, code, 1)

old = '''    if (kparams->n_hmx) {
        return hmx_mm_op_matmul(octx, kparams);
    }

    return hvx_mm_matmul(octx);'''
assert old in s
new = '''    if (kparams->n_hmx) {
#ifdef GGML_HEXAGON_ROUTER_REFINE
        int st = hmx_mm_op_matmul(octx, kparams);
        if (st == HTP_STATUS_OK && router_refine_applicable(octx)) {
            st = router_refine(octx);
        }
        return st;
#else
        return hmx_mm_op_matmul(octx, kparams);
#endif
    }

    return hvx_mm_matmul(octx);'''
s = s.replace(old, new, 1)
if crlf: s = s.replace('\n', '\r\n')
open(p, 'w', newline='').write(s)

p = os.path.join(root, 'CMakeLists.txt')
c = open(p, newline='').read()
nl = '\r\n' if '\r\n' in c else '\n'
a = 'option(GGML_HEXAGON_MMID_PIPE '
assert a in c
c = c.replace(a, 'option(GGML_HEXAGON_ROUTER_REFINE "Exact F32 recompute of the top-16 router candidates after the HMX router matmul" ON)' + nl + a, 1)
b = '    $<$<BOOL:${GGML_HEXAGON_MMID_PIPE}>:GGML_HEXAGON_MMID_PIPE=1>'
assert b in c
c = c.replace(b, b + nl + '    $<$<BOOL:${GGML_HEXAGON_ROUTER_REFINE}>:GGML_HEXAGON_ROUTER_REFINE=1>', 1)
open(p, 'w', newline='').write(c)
print('patched router refine')
