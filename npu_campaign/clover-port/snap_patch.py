import sys
import os
def rw(p, f):
    s = open(p, newline='').read()
    nl_win = chr(13) + chr(10)
    crlf = nl_win in s
    s = s.replace(nl_win, chr(10))
    s = f(s)
    if crlf:
        s = s.replace(chr(10), nl_win)
    open(p, 'w', newline='').write(s)

def h(s):
    old = "    HTP_GDN_KERNEL_HMX_CHUNKED   = 1,\n"
    assert old in s
    return s.replace(old, old + "    HTP_GDN_KERNEL_HVX_RECURRENT_SNAP = 2, // HVX prompt kernel, snapshots written by async DMA from a third state buffer\n", 1)

def c(s):
    old = "extern int    opt_gdn_select;\n"
    assert old in s
    return s.replace(old, old + "extern int    opt_gdn_snapdma;\n", 1)

def host(s):
    old = "int    opt_gdn_select    = 2; // 2 = HMX -> HVX, 1 = HVX, 0 = CPU (unsupported)\n"
    assert old in s
    s = s.replace(old, old + "int    opt_gdn_snapdma   = 1; // 1 = write GDN state snapshots by DMA in the HVX prompt kernel (default), 0 = HVX copy\n", 1)
    old2 = "    opt_gdn_select    = str_gdn_select    ? atoi(str_gdn_select)          : opt_gdn_select;\n"
    assert old2 in s
    s = s.replace(old2, old2 + "    if (const char * str_gdn_snapdma = getenv(\"GGML_HEXAGON_GDN_SNAPDMA\")) { opt_gdn_snapdma = atoi(str_gdn_snapdma); }\n", 1)
    return s

def kp(s):
    old = """        kparams->kernel_type   = HTP_GDN_KERNEL_HVX_RECURRENT;
        kparams->pipeline      = 0;
        kparams->n_heads_batch = 1;
        kparams->vtcm_size     = (uint32_t) hvx_layout.total_bytes;
"""
    new = old + """
        // third per-thread state buffer for snapshot DMA
        const size_t snap_bytes = (hvx_layout.bytes_per_thread + hvx_layout.state_aligned) * n_threads;
        if (opt_gdn_snapdma && n_tokens > 1 && K > 1 && g->ne[0] == 1 && snap_bytes <= vtcm_size) {
            kparams->kernel_type = HTP_GDN_KERNEL_HVX_RECURRENT_SNAP;
            kparams->vtcm_size   = (uint32_t) snap_bytes;
        }
"""
    assert old in s
    return s.replace(old, new, 1)

SNAP = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'snap_kernel.c.txt')).read()

def dsp(s):
    marker = "static void gated_delta_net_f32_tg_thread("
    assert marker in s
    s = s.replace(marker, SNAP + marker, 1)
    old = """    htp_gdn_vtcm_layout_build(&gctx.layout, S_v, n_threads);
    gctx.vtcm_base = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_base, gctx.layout.off_state);
"""
    new = """    htp_gdn_vtcm_layout_build(&gctx.layout, S_v, n_threads);
    const bool snap_dma = kparams->kernel_type == HTP_GDN_KERNEL_HVX_RECURRENT_SNAP && n_tokens > 1;
    if (snap_dma) {
        gctx.layout.bytes_per_thread += gctx.layout.state_aligned;
        gctx.layout.total_bytes       = gctx.layout.bytes_per_thread * n_threads;
    }
    gctx.vtcm_base = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_base, gctx.layout.off_state);
"""
    assert old in s
    s = s.replace(old, new, 1)
    old = """    } else {
        work_queue_run(octx->ctx->work_queue, gated_delta_net_f32_pp_thread, &gctx, n_threads);"""
    new = """    } else if (snap_dma) {
        work_queue_run(octx->ctx->work_queue, gated_delta_net_f32_pp_snap_thread, &gctx, n_threads);
    } else {
        work_queue_run(octx->ctx->work_queue, gated_delta_net_f32_pp_thread, &gctx, n_threads);"""
    assert old in s
    return s.replace(old, new, 1)

root = sys.argv[1]
os.chdir(root)
rw('htp/gated-delta-net-ops.h', h)
rw('htp-common.h', c)
rw('ggml-hexagon.cpp', host)
rw('htp-kparams.cpp', kp)
rw('htp/gated-delta-net-ops.c', dsp)
print('ok')
