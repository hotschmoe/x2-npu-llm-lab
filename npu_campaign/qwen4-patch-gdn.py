import os
W = r'C:\Users\hotschmoe\documents\llm_inference\llama.cpp-hexagon-qwen4\ggml\src\ggml-hexagon'
def rd(p): return open(os.path.join(W, p), 'rb').read().decode()
def wr(p, s): open(os.path.join(W, p), 'wb').write(s.encode())

p = 'htp/gated-delta-net-ops.c'
s = rd(p)
nl = '\r\n' if '\r\n' in s else '\n'
s = s.replace('\r\n', '\n')
a = "static void gated_delta_net_f32_tg_thread("
assert a in s
s = s.replace(a, '#include "gdn-rb.h"\n\n' + a, 1)
a = "    if (n_tokens == 1) {\n        work_queue_run(octx->ctx->work_queue, gated_delta_net_f32_tg_thread, &gctx, n_threads);"
assert a in s
b = ("    if (n_tokens == 1 && (octx->flags & HTP_OPFLAGS_GDN_RB) && !kparams->kda && (kparams->S_v % GDN_RB_ROWS) == 0 && octx->ctx->mdev.count <= 1 &&\n"
     "        gctx.layout.bytes_per_thread >= (size_t) GDN_RB_NBUF * GDN_RB_ROWS * kparams->S_v * sizeof(float) && n_threads > 1) {\n"
     "        work_queue_run(octx->ctx->work_queue, gated_delta_net_f32_tg_rb_thread, &gctx, n_threads);\n"
     "    } else if (n_tokens == 1) {\n"
     "        work_queue_run(octx->ctx->work_queue, gated_delta_net_f32_tg_thread, &gctx, n_threads);")
s = s.replace(a, b, 1)
wr(p, s.replace('\n', nl))

p = 'htp/htp-ops.h'
s = rd(p)
a = "    HTP_OPFLAGS_MV_XPIPE "
assert a in s
s = s.replace(a, "    HTP_OPFLAGS_GDN_RB         = (1U << 17), // qwen4: decode GDN with row-block work units (bit-exact)\n" + a, 1)
wr(p, s)

p = 'ggml-hexagon.cpp'
s = rd(p)
nl = '\r\n' if '\r\n' in s else '\n'
s = s.replace('\r\n', '\n')
a = "        if (opt_mv_xpipe && (node.opcode == HTP_OP_MUL_MAT_ID || node.opcode == HTP_OP_MUL_MAT_ID_NX)) { o.flags |= HTP_OPFLAGS_MV_XPIPE; }"
assert a in s
s = s.replace(a, a + "\n        if (opt_gdn_rb && node.opcode == HTP_OP_GATED_DELTA_NET) { o.flags |= HTP_OPFLAGS_GDN_RB; }", 1)
a = "static int opt_mv_xpipe = 0;"
assert a in s
s = s.replace(a, a + "\n// qwen4: GGML_HEXAGON_GDN_RB=1 -> decode GDN with row-block work units (bit-exact; see htp/gdn-rb.h)\nstatic int opt_gdn_rb = 0;", 1)
a = '    if (const char * sxp = getenv("GGML_HEXAGON_MV_XPIPE")) { opt_mv_xpipe = atoi(sxp); }'
assert a in s
s = s.replace(a, a + '\n    if (const char * sgr = getenv("GGML_HEXAGON_GDN_RB")) { opt_gdn_rb = atoi(sgr); }', 1)
wr(p, s.replace('\n', nl))
print('ok')
