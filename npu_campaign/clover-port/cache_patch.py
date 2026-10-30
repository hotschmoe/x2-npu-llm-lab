import sys
import os

NL = chr(13) + chr(10)

def rw(p, f):
    s = open(p, newline='').read()
    crlf = NL in s
    s = s.replace(NL, chr(10))
    s = f(s)
    if crlf:
        s = s.replace(chr(10), NL)
    open(p, 'w', newline='').write(s)

def rep(s, old, new):
    assert old in s, old[:60]
    return s.replace(old, new, 1)

def host(s):
    # options
    s = rep(s, "int    opt_opfusion  = 1;    // enable/disable op fusion\n",
            "int    opt_opfusion  = 1;    // enable/disable op fusion\n"
            "int    opt_graph_cache = 1;  // number of per-split graphs whose solved op nodes are kept (1 = last graph only)\n"
            "int    opt_hoststat  = 0;    // print host-side time accounting at session release\n")
    # stats struct, before session struct usage: put after options block marker
    s = rep(s, "static std::regex* opt_opfilter = NULL; // regex of ops to not claim\n",
            "static std::regex* opt_opfilter = NULL; // regex of ops to not claim\n"
            "\n"
            "struct ggml_hexagon_hoststat {\n"
            "    std::atomic<uint64_t> n_hit{0}, n_miss{0}, n_ops{0}, us_compute{0}, us_solve{0}, us_enq{0}, us_wait{0}, n_wait{0};\n"
            "};\n"
            "static ggml_hexagon_hoststat g_hoststat;\n")
    # session members
    s = rep(s, "    uint64_t                cached_uid = 0;\n    std::vector<htp_opnode> cached_nodes;\n",
            "    std::unordered_map<uint64_t, std::vector<htp_opnode>> graph_cache;\n")
    # graph compute
    old = s[s.index("    bool cache_hit = (graph->uid != 0 && sess->cached_uid == graph->uid);"):s.index("    for (const auto & node : *nodes_ptr) {\n        sess->enqueue_op(node);")]
    new = '''    const int64_t t_start = opt_hoststat ? ggml_time_us() : 0;

    auto cached = graph->uid != 0 ? sess->graph_cache.find(graph->uid) : sess->graph_cache.end();
    if (cached != sess->graph_cache.end()) {
        nodes_ptr = &cached->second;
        g_hoststat.n_hit++;
    } else {
        g_hoststat.n_miss++;
        htp_fusion_tag_graph(graph);

        computed_nodes.reserve(graph->n_nodes);

        for (int i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * n = graph->nodes[i];
            if (!op_is_compute(n)) {
                continue;
            }

            const htp_op_code opcode = htp_op_remap(n);
            if (opcode == HTP_OP_INVALID) {
                GGML_ABORT("\\nggml-hex: graph-compute %s is not supported\\n", ggml_op_desc(n));
            }

            htp_opnode node(opcode, n);
            if (node.opcode == HTP_OP_MUL_MAT || node.opcode == HTP_OP_MUL_MAT_ID || node.opcode == HTP_OP_GET_ROWS) {
                ggml_tensor * src0 = node.node->src[0];
                if (htp_needs_repack(src0)) {
                    HEX_VERBOSE("ggml-hex: %s rt-repack %s|size %zu\\n", sess->c_name(), htp_format_tensor(src0, HTP_OPFMT_TENSOR).c_str(), ggml_nbytes(src0));
                    std::vector<uint8_t> raw(ggml_nbytes(src0));
                    memcpy(raw.data(), src0->data, raw.size());
                    htp_repack_tensor_tiled(src0, raw.data(), raw.size());
                    auto * extra = (htp_tensor_extra *) src0->extra;
                    extra->flags |= HTP_TENSOR_TILED;
                }
            }

            GGML_ASSERT(htp_kparams_solve(sess, node));
            if (node.opcode == HTP_OP_CPY && node.kparams<htp_copy_kernel_params>()->total_elems == 0) {
                continue;
            }

            computed_nodes.push_back(std::move(node));
        }

        if (graph->uid != 0) {
            if ((int) sess->graph_cache.size() >= opt_graph_cache) {
                sess->graph_cache.clear();
            }
            auto & slot = sess->graph_cache[graph->uid];
            slot        = std::move(computed_nodes);
            nodes_ptr   = &slot;
        } else {
            nodes_ptr = &computed_nodes;
        }
    }

    const int64_t t_solved = opt_hoststat ? ggml_time_us() : 0;
    g_hoststat.n_ops += nodes_ptr->size();

'''
    s = rep(s, old, new)
    # timing at the end of graph compute: after the enqueue loop
    s = rep(s, "    for (const auto & node : *nodes_ptr) {\n        sess->enqueue_op(node);\n    }\n",
            "    for (const auto & node : *nodes_ptr) {\n        sess->enqueue_op(node);\n    }\n"
            "    if (opt_hoststat) {\n"
            "        const int64_t t_end = ggml_time_us();\n"
            "        g_hoststat.us_solve += t_solved - t_start;\n"
            "        g_hoststat.us_enq   += t_end - t_solved;\n"
            "        g_hoststat.us_compute += t_end - t_start;\n"
            "    }\n")
    # wait accounting around dspqueue_read in flush_pending
    s = rep(s, "        int err = dspqueue_read(this->queue, &flags, 1, &n_dbufs, &dbuf, sizeof(rsp), &rsp_size, (uint8_t *) &rsp, timeo);\n        if (err == AEE_EEXPIRED",
            "        const int64_t t_rd0 = opt_hoststat ? ggml_time_us() : 0;\n"
            "        int err = dspqueue_read(this->queue, &flags, 1, &n_dbufs, &dbuf, sizeof(rsp), &rsp_size, (uint8_t *) &rsp, timeo);\n"
            "        if (opt_hoststat) { g_hoststat.us_wait += ggml_time_us() - t_rd0; g_hoststat.n_wait++; }\n"
            "        if (err == AEE_EEXPIRED")
    # print at release
    s = rep(s, "void ggml_hexagon_session::release() noexcept(true) {\n",
            "void ggml_hexagon_session::release() noexcept(true) {\n"
            "    if (opt_hoststat) {\n"
            "        GGML_LOG_INFO(\"ggml-hex: hoststat: graphs hit %llu miss %llu, ops %llu, graph_compute %.1f ms (solve %.1f ms, enqueue %.1f ms), dspqueue_read wait %.1f ms in %llu reads\\n\",\n"
            "                      (unsigned long long) g_hoststat.n_hit.load(), (unsigned long long) g_hoststat.n_miss.load(), (unsigned long long) g_hoststat.n_ops.load(),\n"
            "                      g_hoststat.us_compute.load() / 1e3, g_hoststat.us_solve.load() / 1e3, g_hoststat.us_enq.load() / 1e3,\n"
            "                      g_hoststat.us_wait.load() / 1e3, (unsigned long long) g_hoststat.n_wait.load());\n"
            "    }\n")
    # env
    s = rep(s, "    if (const char * str_gdn_snapdma = getenv(\"GGML_HEXAGON_GDN_SNAPDMA\")) { opt_gdn_snapdma = atoi(str_gdn_snapdma); }\n",
            "    if (const char * str_gdn_snapdma = getenv(\"GGML_HEXAGON_GDN_SNAPDMA\")) { opt_gdn_snapdma = atoi(str_gdn_snapdma); }\n"
            "    if (const char * str_gc = getenv(\"GGML_HEXAGON_GRAPH_CACHE\")) { opt_graph_cache = std::max(1, atoi(str_gc)); }\n"
            "    if (const char * str_hs = getenv(\"GGML_HEXAGON_HOSTSTAT\")) { opt_hoststat = atoi(str_hs); }\n")
    return s

root = sys.argv[1]
os.chdir(root)
rw('ggml-hexagon.cpp', host)
print('ok')
