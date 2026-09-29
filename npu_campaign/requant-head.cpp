// decode2: write a copy of a GGUF whose output.weight (Q6_K) is requantized to another type (default Q4_0). Source untouched.
//   requant-head.exe src.gguf dst.gguf [q4_0|iq4_nl|q5_0|q5_k|q8_0]
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
static void require(bool ok, const char * m) { if (!ok) { std::fprintf(stderr, "%s\n", m); std::exit(1); } }
int main(int argc, char ** argv) {
    require(argc >= 3, "usage: requant-head src dst [type]");
    std::string ty = argc > 3 ? argv[3] : "q4_0";
    ggml_type nt = ty == "q4_0" ? GGML_TYPE_Q4_0 : ty == "iq4_nl" ? GGML_TYPE_IQ4_NL : ty == "q5_0" ? GGML_TYPE_Q5_0 : ty == "q5_k" ? GGML_TYPE_Q5_K : ty == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_COUNT;
    require(nt != GGML_TYPE_COUNT, "bad type");
    FILE * ex = std::fopen(argv[2], "rb"); if (ex) { std::fclose(ex); require(false, "destination exists"); }
    gguf_init_params params = { true, nullptr };
    gguf_context * meta = gguf_init_from_file(argv[1], params);
    require(meta, "source invalid");
    const size_t data0 = gguf_get_data_offset(meta), align = gguf_get_alignment(meta);
    const int64_t count = gguf_get_n_tensors(meta);
    std::vector<size_t> off(count), sz(count);
    int64_t head = -1;
    for (int64_t i = 0; i < count; ++i) { off[i] = gguf_get_tensor_offset(meta, i); sz[i] = gguf_get_tensor_size(meta, i); if (!std::strcmp(gguf_get_tensor_name(meta, i), "output.weight")) head = i; }
    require(head >= 0 && gguf_get_tensor_type(meta, head) == GGML_TYPE_Q6_K, "output.weight not Q6_K");
    const int64_t n_per_row = 2048, nrows = 248320;
    require((int64_t)(sz[head]) == nrows * (n_per_row / 256) * 210, "unexpected head size");
    gguf_set_tensor_type(meta, "output.weight", nt);
    FILE * src = std::fopen(argv[1], "rb"); FILE * dst = std::fopen(argv[2], "wbx");
    require(src && dst, "open failed");
    require(gguf_write_to_file_ptr(meta, dst, true), "write meta failed");
    const size_t nd0 = (size_t) _ftelli64(dst);
    std::vector<unsigned char> buf(8u << 20);
    for (int64_t i = 0; i < count; ++i) {
        require((size_t) _ftelli64(dst) == nd0 + gguf_get_tensor_offset(meta, i), "dest offset mismatch");
        require(_fseeki64(src, data0 + off[i], SEEK_SET) == 0, "seek failed");
        if (i == head) {
            std::vector<unsigned char> q6(sz[head]);
            require(std::fread(q6.data(), 1, q6.size(), src) == q6.size(), "read head failed");
            const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_Q6_K);
            std::vector<float> row(n_per_row);
            const size_t q6row = sz[head] / nrows, nrow_bytes = ggml_row_size(nt, n_per_row);
            std::vector<unsigned char> out(nrow_bytes * nrows);
            for (int64_t r = 0; r < nrows; ++r) {
                tr->to_float(q6.data() + r * q6row, row.data(), n_per_row);
                size_t w = ggml_quantize_chunk(nt, row.data(), out.data() + r * nrow_bytes, 0, 1, n_per_row, nullptr);
                require(w == nrow_bytes, "quantize size mismatch");
            }
            require(std::fwrite(out.data(), 1, out.size(), dst) == out.size(), "write head failed");
            std::printf("requantized output.weight Q6_K -> %s: %zu -> %zu bytes\n", ty.c_str(), q6.size(), out.size());
        } else {
            size_t left = sz[i];
            while (left) { size_t c = left < buf.size() ? left : buf.size(); require(std::fread(buf.data(), 1, c, src) == c, "read failed"); require(std::fwrite(buf.data(), 1, c, dst) == c, "write failed"); left -= c; }
        }
        size_t bytes = gguf_get_tensor_size(meta, i), pad = (align - bytes % align) % align;
        const std::vector<unsigned char> z(pad, 0);
        require(std::fwrite(z.data(), 1, pad, dst) == pad, "pad failed");
    }
    require(std::fclose(dst) == 0, "close failed");
    std::fclose(src); gguf_free(meta);
    meta = gguf_init_from_file(argv[2], params);
    require(meta && gguf_get_n_tensors(meta) == count, "dest validation failed");
    std::printf("ok\n");
    return 0;
}
