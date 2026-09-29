"""Prefill MoE weighted-sum fusion: MUL(down, weights) + 7 chained ADDs of the per-slot views -> one multi-threaded ADD_NW.
Same per-row math as the decode ADD_NW kernel (hvx_mul_scalar_f32_aa, then hvx_add_f32_aaa left fold), just T tokens.
Edits: htp/binary-ops.c (DSP), ggml-hexagon.cpp (host matcher)."""
import os
base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'llama.cpp-hexagon', 'ggml', 'src', 'ggml-hexagon')

def load(rel):
    p = os.path.join(base, rel)
    s = open(p, newline='').read()
    return p, s.replace('\r\n', '\n'), '\r\n' in s

def save(p, s, crlf):
    if crlf: s = s.replace('\n', '\r\n')
    open(p, 'w', newline='').write(s)

# ------------------------------------------------------------------ DSP
p, s, crlf = load('htp/binary-ops.c')
if 'add_nw_prefill_worker' not in s:
    dsp = r'''// Prefill variant: T tokens, kernel_params[2] = T. src[0] = down [n_elem, 8, T] (contiguous per token), src[1] = weights
// [1, 8, T], dst = [n_elem, T]. Token t is handled by thread t % nth; DMA in/out double buffered per thread.
static void add_nw_prefill_worker(unsigned int nth, unsigned int ith, void * data) {
    struct htp_add_nw_args * args = (struct htp_add_nw_args *) data;
    struct htp_ops_context * octx = args->octx;
    const uint32_t n_src  = (uint32_t) octx->kernel_params[0];
    const uint32_t n_elem = (uint32_t) octx->kernel_params[1];
    const uint32_t T      = (uint32_t) octx->kernel_params[2];
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * w   = octx->src[1];
    const struct htp_tensor * dst = octx->dst;

    const uint32_t row  = n_elem * (uint32_t) sizeof(float);
    const uint32_t span = n_src * row;
    const uint32_t per  = hex_align_up(span, 128) + 2u * hex_align_up(row, 128) + 128u;   // block + acc + tmp + weights
    uint8_t * base = (uint8_t *) hex_align_up((uintptr_t) octx->ctx->vtcm_base, 128) + (size_t) ith * 2u * per;
    uint8_t * blk[2] = { base, base + per };
    uint8_t * acc[2], * tmp[2], * wv[2];
    for (int k = 0; k < 2; ++k) {
        acc[k] = blk[k] + hex_align_up(span, 128);
        tmp[k] = acc[k] + hex_align_up(row, 128);
        wv[k]  = tmp[k] + hex_align_up(row, 128);
    }

    dma_queue * q = octx->ctx->dma[ith];
    uint32_t sp = 0, pp = 0;
    uint32_t in_seq[2] = { 0, 0 }, out_seq[2] = { 0, 0 };
    bool     out_pending[2] = { false, false };
#define NW_WAIT(target) do { while (pp <= (target)) { dma_queue_pop(q); pp++; } } while (0)
#define NW_PUSH_IN(buf, tok) do {                                                                                        \
        dma_queue_push(q, dma_make_data(blk[buf], src->data + (size_t) (tok) * src->nb[2]), span, span, span, 1);        \
        dma_queue_push(q, dma_make_data(wv[buf], w->data + (size_t) (tok) * w->nb[2]), 32, 32, 32, 1);                   \
        in_seq[buf] = sp + 1; sp += 2;                                                                                   \
    } while (0)

    uint32_t i = 0;
    uint32_t t = ith;
    if (t < T) { NW_PUSH_IN(0, t); }
    for (; t < T; t += nth, ++i) {
        const int b = (int) (i & 1);
        const uint32_t tn = t + nth;
        if (tn < T) {
            if (out_pending[b ^ 1]) { NW_WAIT(out_seq[b ^ 1]); out_pending[b ^ 1] = false; }
            NW_PUSH_IN(b ^ 1, tn);
        }
        NW_WAIT(in_seq[b]);
        if (out_pending[b]) { NW_WAIT(out_seq[b]); out_pending[b] = false; }

        float __attribute__((aligned(128))) wt[32];
        hvx_vec_store_a(wt, 32, *(const HVX_Vector *) wv[b]);
        hvx_mul_scalar_f32_aa(acc[b], blk[b], wt[0], n_elem);
        for (uint32_t s = 1; s < n_src; s++) {
            hvx_mul_scalar_f32_aa(tmp[b], blk[b] + (size_t) s * row, wt[s], n_elem);
            hvx_add_f32_aaa(acc[b], acc[b], tmp[b], n_elem);
        }
        dma_queue_push(q, dma_make_data(dst->data + (size_t) t * dst->nb[1], acc[b]), row, row, row, 1);
        out_seq[b] = sp++;
        out_pending[b] = true;
    }
    while (pp < sp) { dma_queue_pop(q); pp++; }
#undef NW_PUSH_IN
#undef NW_WAIT
    (void) nth;
}

'''
    anchor = 'int op_add_nw(struct htp_ops_context * octx) {'
    assert anchor in s
    s = s.replace(anchor, dsp + anchor, 1)
    i0 = s.index('int op_add_nw(struct htp_ops_context * octx) {')
    i1 = s.index('\n}\n', i0) + 3
    new_fn = r'''int op_add_nw(struct htp_ops_context * octx) {
    struct htp_add_nw_args args;
    args.octx   = octx;
    args.status = HTP_STATUS_INTERNAL_ERR;
    const uint32_t n_src  = (uint32_t) octx->kernel_params[0];
    const uint32_t n_elem = (uint32_t) octx->kernel_params[1];
    const uint32_t T      = (uint32_t) octx->kernel_params[2];
    if (T > 1) {
        const struct htp_tensor * src = octx->src[0];
        const struct htp_tensor * w   = octx->src[1];
        const struct htp_tensor * dst = octx->dst;
        if (n_src != 8 || n_elem == 0 || (n_elem % 32u) != 0 || !src || !w || !dst ||
            src->type != HTP_TYPE_F32 || w->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32 ||
            src->ne[0] != n_elem || src->ne[1] != n_src || src->ne[2] != T ||
            src->nb[0] != sizeof(float) || src->nb[1] != n_elem * (uint32_t) sizeof(float) ||
            src->nb[2] != (size_t) n_src * n_elem * sizeof(float) ||
            w->ne[0] != 1 || w->ne[1] != n_src || w->ne[2] != T || w->nb[0] != sizeof(float) ||
            w->nb[1] != sizeof(float) || w->nb[2] != (size_t) n_src * sizeof(float) ||
            dst->ne[0] != n_elem || dst->ne[1] != T || dst->nb[0] != sizeof(float) ||
            src->data == 0 || w->data == 0 || dst->data == 0) {
            return HTP_STATUS_NO_SUPPORT;
        }
        const uint32_t row  = n_elem * (uint32_t) sizeof(float);
        const uint32_t span = n_src * row;
        const size_t   per  = (size_t) hex_align_up(span, 128) + 2u * hex_align_up(row, 128) + 128u;
        const uint32_t nth  = octx->n_threads;
        if ((size_t) octx->ctx->vtcm_size < (size_t) nth * 2u * per + 256u) {
            return HTP_STATUS_VTCM_TOO_SMALL;
        }
        work_queue_run(octx->ctx->work_queue, add_nw_prefill_worker, &args, nth);
        return HTP_STATUS_OK;
    }
    work_queue_run(octx->ctx->work_queue, add_nw_worker, &args, 1);
    return args.status;
}
'''
    s = s[:i0] + new_fn + s[i1:]
    save(p, s, crlf)

# ------------------------------------------------------------------ host
p, s, crlf = load('ggml-hexagon.cpp')
if 'try_fuse_moe_wsum_prefill' not in s:
    host = r'''    // Prefill expert combine: MUL(down[C,8,T], w[1,8,T]) then seven chained ADDs of the eight per-slot views [C,T].
    // One multi-threaded ADD_NW per token block; same per-row math as the decode ADD_NW.
    bool try_fuse_moe_wsum_prefill(const htp_opnode & node) {
        if (node.opcode != HTP_OP_ADD || n_ops < 7) return false;
        if (sess->mdev.count > 1) return false;
        const ggml_tensor * dst = node.dst();
        if (!dst || dst->type != GGML_TYPE_F32 || dst->ne[2] != 1 || dst->ne[3] != 1 || (dst->flags & GGML_TENSOR_FLAG_OUTPUT)) return false;
        const int64_t n_elem = dst->ne[0];
        const int64_t T      = dst->ne[1];
        if (T < 2 || n_elem <= 0 || (n_elem % 32) != 0 || n_elem > 8192 || dst->nb[0] != sizeof(float)) return false;
        if (T > INT32_MAX) return false;

        // walk the left-fold chain backwards: node = ADD(prev, view7), ops[n_ops-1] = ADD(prev', view6), ..., ops[n_ops-6] = ADD(view0, view1)
        const ggml_tensor * parent = nullptr;
        const ggml_tensor * expect = node.src0();
        const ggml_tensor * slot7  = node.src1();
        if (!expect || !slot7) return false;
        const ggml_tensor * views[8];
        views[7] = slot7;
        for (int k = 1; k <= 6; k++) {
            const htp_opnode & a = ops[n_ops - k];
            if (a.opcode != HTP_OP_ADD || a.outputs.size() != 1 || !a.fused.empty() || a.dst() != expect) return false;
            if (a.dst()->flags & GGML_TENSOR_FLAG_OUTPUT) return false;
            if (k < 6) {
                views[7 - k] = a.src1();
                expect = a.src0();
            } else {
                views[0] = a.src0();
                views[1] = a.src1();
            }
        }
        for (int e = 0; e < 8; e++) {
            const ggml_tensor * v = views[e];
            if (!v || v->op != GGML_OP_VIEW || !v->view_src || v->type != GGML_TYPE_F32) return false;
            if (!parent) parent = v->view_src;
            if (v->view_src != parent) return false;
            if (v->view_offs != (size_t) e * n_elem * sizeof(float)) return false;
            if (v->ne[0] != n_elem || v->ne[1] != T || v->nb[0] != sizeof(float) || v->nb[1] != (size_t) 8 * n_elem * sizeof(float)) return false;
        }
        if (parent->ne[0] != n_elem || parent->ne[1] != 8 || parent->ne[2] != T || parent->ne[3] != 1) return false;
        if (parent->nb[1] != (size_t) n_elem * sizeof(float) || parent->nb[2] != (size_t) 8 * n_elem * sizeof(float)) return false;
        if (parent->flags & GGML_TENSOR_FLAG_OUTPUT) return false;

        int mul_i = -1;
        for (unsigned int i = 0; i + 6 < n_ops; i++) {
            if (ops[i].opcode == HTP_OP_MUL && ops[i].fused.empty() && ops[i].node == parent) { mul_i = (int) i; break; }
        }
        if (mul_i < 0) return false;
        const htp_opnode & mul = ops[mul_i];
        if (mul.outputs.size() != 1 || mul.inputs.size() < 2) return false;
        const ggml_tensor * down = mul.src0();
        const ggml_tensor * weights = mul.src1();
        if (!down || !weights) return false;
        if (down->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32) return false;
        if (down->ne[0] != n_elem || down->ne[1] != 8 || down->ne[2] != T || down->ne[3] != 1) return false;
        if (down->nb[0] != sizeof(float) || down->nb[1] != (size_t) n_elem * sizeof(float) || down->nb[2] != (size_t) 8 * n_elem * sizeof(float)) return false;
        if (weights->ne[0] != 1 || weights->ne[1] != 8 || weights->ne[2] != T || weights->ne[3] != 1) return false;
        if (weights->nb[0] != sizeof(float) || weights->nb[1] != sizeof(float) || weights->nb[2] != 8 * sizeof(float)) return false;
        if ((down->flags | weights->flags) & GGML_TENSOR_FLAG_OUTPUT) return false;
        auto backed = [](const ggml_tensor * t) { return t && t->data && t->buffer && t->extra; };
        if (!backed(down) || !backed(weights) || !dst->data || !dst->buffer) return false;
        auto extended = [](const ggml_tensor * t) {
            auto * sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
            return !sbuf || sbuf->extended;
        };
        if (extended(down) || extended(weights) || extended(dst)) return false;
        // the output is written while inputs are still being read: it must not alias them
        static int ov_logged = 0;
        if (ggml_hexagon_tensors_overlap(dst, down) || ggml_hexagon_tensors_overlap(dst, weights)) {
            if (ov_logged++ == 0) GGML_LOG_INFO("ggml-hex: %s prefill expert combine skipped: output aliases an input
", sess->c_name());
            return false;
        }

        // nothing else may read the weighted rows or the partial sums
        for (unsigned int i = 0; i < n_ops; i++) {
            if ((int) i == mul_i || i >= n_ops - 6) continue;
            for (const ggml_tensor * in : ops[i].inputs) {
                if (in && (in == parent || in->view_src == parent)) return false;
            }
        }
        if (!try_fuse_common({down, weights, dst})) return false;

        std::vector<decltype(ops[0].node)> absorbed;
        absorbed.push_back(ops[mul_i].node);
        for (int k = 6; k >= 1; k--) absorbed.push_back(ops[n_ops - k].node);

        htp_opnode & fused = ops[mul_i];
        fused.node   = node.node;
        fused.opcode = HTP_OP_ADD_NW;
        fused.name   = "ADD+NW+T";
        fused.inputs.clear();
        fused.inputs.push_back(down);
        fused.inputs.push_back(weights);
        fused.outputs.clear();
        fused.outputs.push_back(dst);
        fused.fused.clear();
        for (auto n : absorbed) fused.fused.push_back(n);
        memset(fused.kernel_params, 0, sizeof(fused.kernel_params));
        fused.kernel_params[0] = 8;
        fused.kernel_params[1] = (int32_t) n_elem;
        fused.kernel_params[2] = (int32_t) T;

        htp_op_desc & o = h_ops[mul_i];
        memset(&o, 0, sizeof(o));
        o.opcode = HTP_OP_ADD_NW;
        memcpy(o.kernel_params, fused.kernel_params, sizeof(o.kernel_params));
        o.src[0] = (uint16_t) add_tensor(down);
        o.src[1] = (uint16_t) add_tensor(weights);
        for (uint32_t s = 2; s < HTP_OP_MAX_INPUTS; s++) o.src[s] = 0xffff;
        o.dst[0] = (uint16_t) add_tensor(dst);
        for (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) o.dst[d] = 0xffff;

        for (int k = 0; k < 6; k++) drop_op(n_ops - 1);   // the six earlier ADDs; the current node is simply not added

        static int logged = 0;
        if (logged++ == 0) {
            GGML_LOG_INFO("ggml-hex: %s fused prefill expert combine into ADD+NW (n_elem %d T %d)\n", sess->c_name(), (int) n_elem, (int) T);
        }
        HEX_VERBOSE("ggml-hex: %s fused ADD+NW+T (#%d)\n", sess->c_name(), mul_i);
        return true;
    }

'''
    anchor = '    // Eight expert rows are summed by seven serial ADDs.'
    assert anchor in s
    s = s.replace(anchor, host + anchor, 1)
    old = '        if (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MOE_ADD)       && try_fuse_moe_add(node))       return true;'
    assert old in s
    s = s.replace(old, '        if (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MOE_ADD)       && try_fuse_moe_wsum_prefill(node)) return true;\n' + old, 1)
    save(p, s, crlf)
print('patched moewsum')
