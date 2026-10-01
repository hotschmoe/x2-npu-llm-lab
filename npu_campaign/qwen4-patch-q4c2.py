import os
W = r'C:\Users\hotschmoe\documents\llm_inference\llama.cpp-hexagon-qwen4\ggml\src\ggml-hexagon'
def rd(p):
    b = open(os.path.join(W, p), 'rb').read().decode()
    nl = '\r\n' if '\r\n' in b else '\n'
    return b.replace('\r\n', '\n'), nl
def wr(p, s, nl): open(os.path.join(W, p), 'wb').write(s.replace('\n', nl).encode())

p = 'htp/matmul-ops.c'
s, nl = rd(p)
assert 'q4_0c' not in s, 'already applied'

def inst(s, last_line, text):
    assert last_line in s, last_line
    return s.replace(last_line, last_line + '\n' + text, 1)
over = "#undef MM_TILE_ALIGN\n#define MM_TILE_ALIGN(ts) (ts)\n"
back = "#undef MM_TILE_ALIGN\n#define MM_TILE_ALIGN(ts) hex_align_up((ts), 128)\n"
s = inst(s, "MATMUL_2D_REPACKED_IMPL(mxfp4,      544,  tiled_vec_dot_mxfp4_32x2, tiled_vec_dot_mxfp4_32x1)",
         over + "MATMUL_2D_REPACKED_IMPL(q4_0c,      576,  tiled_vec_dot_q4_0c_32x2, tiled_vec_dot_q4_0c_32x1)\n" + back.rstrip('\n'))
s = inst(s, "MATVEC_2D_REPACKED_IMPL(mxfp4,      544,  tiled_vec_dot_mxfp4_32x1)",
         over + "MATVEC_2D_REPACKED_IMPL(q4_0c,      576,  tiled_vec_dot_q4_0c_32x1)\n" + back.rstrip('\n'))
s = inst(s, "MATMUL_NX_2D_REPACKED_IMPL(mxfp4,   544,  tiled_vec_dot_mxfp4_32x2, tiled_vec_dot_mxfp4_32x1)",
         over + "MATMUL_NX_2D_REPACKED_IMPL(q4_0c,   576,  tiled_vec_dot_q4_0c_32x2, tiled_vec_dot_q4_0c_32x1)\n" + back.rstrip('\n'))

a = "                case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mm_2d_repacked_q4_0;   break;"
assert a in s
s = s.replace(a, "                case HTP_TYPE_Q4_0:   matmul_job_func = q4c ? hvx_mm_2d_repacked_q4_0c : hvx_mm_2d_repacked_q4_0;   break;", 1)
a = "                case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mv_2d_repacked_q4_0;   break;"
assert a in s
s = s.replace(a, "                case HTP_TYPE_Q4_0:   matmul_job_func = q4c ? hvx_mv_2d_repacked_q4_0c : hvx_mv_2d_repacked_q4_0;   break;", 1)
a = "            case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mm_nx_2d_repacked_q4_0;   break;"
assert a in s
s = s.replace(a, "            case HTP_TYPE_Q4_0:   matmul_job_func = q4c ? hvx_mm_nx_2d_repacked_q4_0c : hvx_mm_nx_2d_repacked_q4_0;   break;", 1)

Q4C = "    const bool q4c = ((octx->flags & HTP_OPFLAGS_CONTIG) != 0) && src0->type == HTP_TYPE_Q4_0 && octx->ctx->mdev.count <= 1;\n"
a = "    // Compute src0_nrows_per_thread\n"
assert a in s
s = s.replace(a, Q4C + a, 1)
a = "    struct htp_mm_context mmctx_struct = {0};\n    struct htp_mm_context * mmctx = &mmctx_struct;\n    mmctx->octx = octx;\n    mmctx->act  = act;\n"
assert a in s
s = s.replace(a, a + Q4C, 1)

a = "int op_matmul_id(struct htp_ops_context * octx) {"
assert a in s
helper = ("static inline void qwen4_use_q4c_dot(struct htp_mm_context * mmctx, const struct htp_tensor * w) {\n"
          "    if (w->type == HTP_TYPE_Q4_0) { mmctx->vec_dot_32x1 = tiled_vec_dot_q4_0c_32x1; }\n"
          "}\n\n")
s = s.replace(a, helper + a, 1)

a = "            const bool xp = ((octx->flags & HTP_OPFLAGS_MV_XPIPE) != 0) && octx->ctx->mdev.count <= 1;\n"
assert a in s
s = s.replace(a, a + "            if (xp && (octx->flags & HTP_OPFLAGS_CONTIG) && src1_nrows <= 32) { qwen4_use_q4c_dot(mmctx, src0); }\n", 1)

a = "            s = hvx_mm_matmul_id_nx(octx, mmctx, act_nrows > 1 ?"
assert a in s
s = s.replace(a, ("            if ((octx->flags & HTP_OPFLAGS_MV_XPIPE) && (octx->flags & HTP_OPFLAGS_CONTIG) && octx->ctx->mdev.count <= 1 &&\n"
                  "                (act_nrows > 1 ? (act_nrows * ids->ne[0] <= 32) : (ids->ne[0] * kparams->n_weights <= MVID_XP_MAX_ITEMS))) { qwen4_use_q4c_dot(mmctx, src0); }\n") + a, 1)
wr(p, s, nl)

# host: env + flag on matmul-like ops
p = 'ggml-hexagon.cpp'
s, nl = rd(p)
a = "        if (opt_gdn_rb && node.opcode == HTP_OP_GATED_DELTA_NET) { o.flags |= HTP_OPFLAGS_GDN_RB; }"
assert a in s
s = s.replace(a, a + "\n        if (opt_contig && (node.opcode == HTP_OP_MUL_MAT || node.opcode == HTP_OP_MUL_MAT_ADD || node.opcode == HTP_OP_MUL_MAT_NX || node.opcode == HTP_OP_MUL_MAT_ID || node.opcode == HTP_OP_MUL_MAT_ID_NX)) { o.flags |= HTP_OPFLAGS_CONTIG; }", 1)
a = "static int opt_gdn_rb = 0;"
assert a in s
s = s.replace(a, a + "\n// qwen4: GGML_HEXAGON_Q4C=1 -> weight tile rows by ONE contiguous DMA descriptor (Q4_0 contiguous dot kernels, Q4_1/Q6_K descriptor only). Needs MV_XPIPE for the expert ops.\nstatic int opt_contig = 0;", 1)
a = '    if (const char * sgr = getenv("GGML_HEXAGON_GDN_RB")) { opt_gdn_rb = atoi(sgr); }'
assert a in s
s = s.replace(a, a + '\n    if (const char * sq4c = getenv("GGML_HEXAGON_Q4C")) { opt_contig = atoi(sq4c); }', 1)
wr(p, s, nl)
print('stage2 ok')
