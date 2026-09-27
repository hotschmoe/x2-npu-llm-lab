// Requant the Q5_0 shared-expert weights in one GGUF to Q5_K, in place.
// Q5_K is the same bytes per weight as Q5_0 and Hexagon already multiplies it.
// A backup of the original blobs is written before any byte of the model changes.
//
//   requant-shexp.exe <model.gguf> <backup.bin>            patch
//   requant-shexp.exe --dry <model.gguf>                   print only
//   requant-shexp.exe --restore <model.gguf> <backup.bin>  put the Q5_0 bytes back

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static constexpr uint32_t GGUF_MAGIC = 0x46554747u;
static constexpr uint32_t BAK_MAGIC  = 0x4B423551u; // 'Q5BK'

enum gguf_vtype {
    GGUF_U8 = 0, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32,
    GGUF_F32, GGUF_BOOL, GGUF_STR, GGUF_ARR, GGUF_U64, GGUF_I64, GGUF_F64,
};

struct Tensor {
    std::string name;
    int n_dims = 0;
    int64_t ne[4] = {1, 1, 1, 1};
    uint32_t type = 0;
    uint64_t offset = 0;
    uint64_t type_pos = 0;
};

static void die(const char * msg) {
    std::fprintf(stderr, "requant-shexp: %s\n", msg);
    std::exit(1);
}

static void rd(FILE * f, void * p, size_t n) {
    if (std::fread(p, 1, n, f) != n) {
        die("short read");
    }
}

static uint32_t ru32(FILE * f) {
    uint32_t v;
    rd(f, &v, 4);
    return v;
}

static uint64_t ru64(FILE * f) {
    uint64_t v;
    rd(f, &v, 8);
    return v;
}

static std::string rstr(FILE * f) {
    uint64_t n = ru64(f);
    if (n > 16ull << 20) {
        die("string too long");
    }
    std::string s(n, '\0');
    if (n) {
        rd(f, s.data(), (size_t) n);
    }
    return s;
}

static size_t vsize(uint32_t t) {
    switch (t) {
        case GGUF_U8: case GGUF_I8: case GGUF_BOOL: return 1;
        case GGUF_U16: case GGUF_I16: return 2;
        case GGUF_U32: case GGUF_I32: case GGUF_F32: return 4;
        case GGUF_U64: case GGUF_I64: case GGUF_F64: return 8;
        default: die("bad value type"); return 0;
    }
}

static void skip_value(FILE * f, uint32_t t) {
    if (t == GGUF_STR) {
        uint64_t n = ru64(f);
        if (_fseeki64(f, (long long) n, SEEK_CUR) != 0) {
            die("seek string");
        }
        return;
    }
    if (t == GGUF_ARR) {
        uint32_t et = ru32(f);
        uint64_t n = ru64(f);
        if (et == GGUF_STR) {
            for (uint64_t i = 0; i < n; i++) {
                skip_value(f, GGUF_STR);
            }
        } else if (et == GGUF_ARR) {
            die("nested array");
        } else {
            uint64_t bytes = n * vsize(et);
            if (_fseeki64(f, (long long) bytes, SEEK_CUR) != 0) {
                die("seek array");
            }
        }
        return;
    }
    if (_fseeki64(f, (long long) vsize(t), SEEK_CUR) != 0) {
        die("seek value");
    }
}

static bool is_shexp_q5(const Tensor & t) {
    if (t.type != GGML_TYPE_Q5_0) {
        return false;
    }
    const std::string & n = t.name;
    if (n.find("ffn_gate_inp") != std::string::npos) {
        return false;
    }
    return n.find("ffn_gate_shexp.weight") != std::string::npos
        || n.find("ffn_up_shexp.weight") != std::string::npos
        || n.find("ffn_down_shexp.weight") != std::string::npos;
}

static size_t row_bytes(const ggml_type_traits * tr, int64_t ne0) {
    if (ne0 % tr->blck_size != 0) {
        die("row not divisible by block");
    }
    return (size_t) (ne0 / tr->blck_size) * tr->type_size;
}

static void patch(const char * model, const char * backup, bool dry) {
    FILE * f = std::fopen(model, dry ? "rb" : "r+b");
    if (!f) {
        die("open model");
    }
    if (ru32(f) != GGUF_MAGIC) {
        die("not gguf");
    }
    uint32_t version = ru32(f);
    if (version < 2) {
        die("gguf version");
    }
    uint64_t n_tensors = ru64(f);
    uint64_t n_kv = ru64(f);
    uint32_t alignment = 32;
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = rstr(f);
        uint32_t vt = ru32(f);
        if (key == "general.alignment" && vt == GGUF_U32) {
            alignment = ru32(f);
        } else {
            skip_value(f, vt);
        }
    }
    std::vector<Tensor> all;
    all.reserve((size_t) n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        Tensor t;
        t.name = rstr(f);
        t.n_dims = (int) ru32(f);
        if (t.n_dims < 1 || t.n_dims > 4) {
            die("n_dims");
        }
        for (int d = 0; d < t.n_dims; d++) {
            t.ne[d] = (int64_t) ru64(f);
        }
        t.type_pos = (uint64_t) _ftelli64(f);
        t.type = ru32(f);
        t.offset = ru64(f);
        all.push_back(std::move(t));
    }
    uint64_t meta_end = (uint64_t) _ftelli64(f);
    uint64_t data0 = (meta_end + alignment - 1) / alignment * alignment;

    std::vector<Tensor> hits;
    for (const auto & t : all) {
        if (is_shexp_q5(t)) {
            hits.push_back(t);
        }
    }
    std::printf("alignment %u data0 %llu q5_0 shexp %zu\n", alignment, (unsigned long long) data0, hits.size());
    if (hits.size() != 60) {
        die("expected 60 Q5_0 shared-expert tensors");
    }

    const ggml_type_traits * tr0 = ggml_get_type_traits(GGML_TYPE_Q5_0);
    const ggml_type_traits * trk = ggml_get_type_traits(GGML_TYPE_Q5_K);
    if (!tr0 || !tr0->to_float || !trk || !trk->to_float) {
        die("type traits");
    }

    FILE * bak = nullptr;
    if (!dry) {
        bak = std::fopen(backup, "wb");
        if (!bak) {
            die("open backup");
        }
        uint32_t magic = BAK_MAGIC;
        uint32_t n = (uint32_t) hits.size();
        std::fwrite(&magic, 4, 1, bak);
        std::fwrite(&n, 4, 1, bak);
    }

    double worst = 0;
    for (const auto & t : hits) {
        int64_t nrows = t.ne[1] * t.ne[2] * t.ne[3];
        int64_t ne0 = t.ne[0];
        if (ne0 % 256 != 0) {
            die("ne0 not divisible by 256");
        }
        size_t src_row = row_bytes(tr0, ne0);
        size_t dst_row = row_bytes(trk, ne0);
        if (src_row != dst_row) {
            die("Q5_0 and Q5_K row size differ");
        }
        size_t nbytes = src_row * (size_t) nrows;
        uint64_t abs = data0 + t.offset;
        std::printf("%s ne=[%lld,%lld,%lld,%lld] off=%llu bytes=%zu\n",
                    t.name.c_str(),
                    (long long) t.ne[0], (long long) t.ne[1], (long long) t.ne[2], (long long) t.ne[3],
                    (unsigned long long) abs, nbytes);

        std::vector<uint8_t> src(nbytes);
        if (_fseeki64(f, (long long) abs, SEEK_SET) != 0) {
            die("seek tensor");
        }
        rd(f, src.data(), nbytes);
        if (!ggml_validate_row_data(GGML_TYPE_Q5_0, src.data(), nbytes)) {
            die("Q5_0 row invalid");
        }

        int64_t nelt = nrows * ne0;
        std::vector<float> f32((size_t) nelt);
        tr0->to_float(src.data(), f32.data(), nelt);

        std::vector<uint8_t> dst(nbytes);
        size_t wrote = ggml_quantize_chunk(GGML_TYPE_Q5_K, f32.data(), dst.data(), 0, nrows, ne0, nullptr);
        if (wrote != nbytes) {
            die("quantized size mismatch");
        }
        if (!ggml_validate_row_data(GGML_TYPE_Q5_K, dst.data(), nbytes)) {
            die("Q5_K row invalid");
        }

        std::vector<float> back((size_t) nelt);
        trk->to_float(dst.data(), back.data(), nelt);
        double max_abs = 0;
        double sse = 0;
        for (int64_t i = 0; i < nelt; i++) {
            double d = (double) back[(size_t) i] - (double) f32[(size_t) i];
            double a = d < 0 ? -d : d;
            if (a > max_abs) {
                max_abs = a;
            }
            sse += d * d;
        }
        double rmse = std::sqrt(sse / (double) nelt);
        std::printf("  rmse %.6g max_abs %.6g\n", rmse, max_abs);
        if (!(rmse < 0.05) || !(max_abs < 1.0)) {
            die("requant error too large");
        }
        if (max_abs > worst) {
            worst = max_abs;
        }

        if (!dry) {
            uint64_t name_n = t.name.size();
            std::fwrite(&t.type_pos, 8, 1, bak);
            uint32_t old_type = t.type;
            uint32_t new_type = GGML_TYPE_Q5_K;
            std::fwrite(&old_type, 4, 1, bak);
            std::fwrite(&new_type, 4, 1, bak);
            std::fwrite(&abs, 8, 1, bak);
            uint64_t nb = nbytes;
            std::fwrite(&nb, 8, 1, bak);
            std::fwrite(&name_n, 8, 1, bak);
            std::fwrite(t.name.data(), 1, t.name.size(), bak);
            std::fwrite(src.data(), 1, nbytes, bak);
            if (std::ferror(bak)) {
                die("backup write");
            }
        }
    }

    if (dry) {
        std::printf("dry-run ok, worst max_abs %.6g\n", worst);
        std::fclose(f);
        return;
    }

    std::fflush(bak);
    std::fclose(bak);
    bak = std::fopen(backup, "rb");
    if (!bak) {
        die("reopen backup");
    }
    uint32_t magic = 0, n = 0;
    rd(bak, &magic, 4);
    rd(bak, &n, 4);
    if (magic != BAK_MAGIC || n != hits.size()) {
        die("backup header");
    }
    std::fclose(bak);

    for (size_t i = 0; i < hits.size(); i++) {
        const Tensor & t = hits[i];
        int64_t nrows = t.ne[1] * t.ne[2] * t.ne[3];
        size_t nbytes = row_bytes(tr0, t.ne[0]) * (size_t) nrows;
        uint64_t abs = data0 + t.offset;

        std::vector<uint8_t> src(nbytes);
        if (_fseeki64(f, (long long) abs, SEEK_SET) != 0) {
            die("seek rewrite");
        }
        rd(f, src.data(), nbytes);
        int64_t nelt = nrows * t.ne[0];
        std::vector<float> f32((size_t) nelt);
        tr0->to_float(src.data(), f32.data(), nelt);
        std::vector<uint8_t> dst(nbytes);
        size_t wrote = ggml_quantize_chunk(GGML_TYPE_Q5_K, f32.data(), dst.data(), 0, nrows, t.ne[0], nullptr);
        if (wrote != nbytes || !ggml_validate_row_data(GGML_TYPE_Q5_K, dst.data(), nbytes)) {
            die("rewrite quant failed");
        }
        if (_fseeki64(f, (long long) abs, SEEK_SET) != 0) {
            die("seek write data");
        }
        if (std::fwrite(dst.data(), 1, nbytes, f) != nbytes) {
            die("write data");
        }
    }
    for (const auto & t : hits) {
        uint32_t neu = GGML_TYPE_Q5_K;
        if (_fseeki64(f, (long long) t.type_pos, SEEK_SET) != 0) {
            die("seek type");
        }
        if (std::fwrite(&neu, 4, 1, f) != 1) {
            die("write type");
        }
    }
    std::fflush(f);
    std::fclose(f);
    std::printf("patched %zu tensors to Q5_K, worst max_abs %.6g, backup %s\n", hits.size(), worst, backup);
}

static void restore(const char * model, const char * backup) {
    FILE * bak = std::fopen(backup, "rb");
    if (!bak) {
        die("open backup");
    }
    uint32_t magic = 0, n = 0;
    rd(bak, &magic, 4);
    rd(bak, &n, 4);
    if (magic != BAK_MAGIC) {
        die("backup magic");
    }
    FILE * f = std::fopen(model, "r+b");
    if (!f) {
        die("open model");
    }
    for (uint32_t i = 0; i < n; i++) {
        uint64_t type_pos = 0, abs = 0, nbytes = 0, name_n = 0;
        uint32_t old_type = 0, new_type = 0;
        rd(bak, &type_pos, 8);
        rd(bak, &old_type, 4);
        rd(bak, &new_type, 4);
        rd(bak, &abs, 8);
        rd(bak, &nbytes, 8);
        rd(bak, &name_n, 8);
        if (name_n > 1024) {
            die("backup name");
        }
        std::string name(name_n, '\0');
        if (name_n) {
            rd(bak, name.data(), (size_t) name_n);
        }
        std::vector<uint8_t> src((size_t) nbytes);
        rd(bak, src.data(), (size_t) nbytes);
        if (!ggml_validate_row_data(GGML_TYPE_Q5_0, src.data(), (size_t) nbytes)) {
            die("backup blob is not Q5_0");
        }
        if (_fseeki64(f, (long long) abs, SEEK_SET) != 0) {
            die("seek restore data");
        }
        if (std::fwrite(src.data(), 1, (size_t) nbytes, f) != (size_t) nbytes) {
            die("restore data");
        }
        if (_fseeki64(f, (long long) type_pos, SEEK_SET) != 0) {
            die("seek restore type");
        }
        if (std::fwrite(&old_type, 4, 1, f) != 1) {
            die("restore type");
        }
        std::printf("restored %s\n", name.c_str());
    }
    std::fflush(f);
    std::fclose(f);
    std::fclose(bak);
    std::printf("restored %u tensors\n", n);
}

int main(int argc, char ** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--dry") == 0) {
        patch(argv[2], nullptr, true);
        return 0;
    }
    if (argc == 4 && std::strcmp(argv[1], "--restore") == 0) {
        restore(argv[2], argv[3]);
        return 0;
    }
    if (argc == 3) {
        patch(argv[1], argv[2], false);
        return 0;
    }
    std::fprintf(stderr, "usage: requant-shexp <model.gguf> <backup.bin> | --dry <model.gguf> | --restore <model.gguf> <backup.bin>\n");
    return 2;
}
