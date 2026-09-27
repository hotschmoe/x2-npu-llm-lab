// Host side of the Hexagon DMA memcpy. The skel times itself with the DSP
// qtimer, so the reported rate is the DMA engine, not FastRPC.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsvc.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct remote_buf {
    void * pv;
    size_t nLen;
};
union remote_arg {
    remote_buf buf;
    uint32_t h;
    uint64_t h64;
};
struct unsigned_module {
    int domain;
    int enable;
};

using remote_handle64 = uint64_t;
using open_fn = int (*)(const char *, remote_handle64 *);
using invoke_fn = int (*)(remote_handle64, uint32_t, remote_arg *);
using close_fn = int (*)(remote_handle64);
using session_fn = int (*)(uint32_t, void *, uint32_t);
using alloc2_fn = void * (*)(int, uint32_t, size_t);
using free_fn = void (*)(void *);

constexpr uint32_t kScalars = (1u << 24) | (2u << 16) | (2u << 8);

std::string driver_dir() {
    SC_HANDLE scm = OpenSCManagerW(nullptr, nullptr, STANDARD_RIGHTS_READ);
    if (!scm) return {};
    SC_HANDLE svc = OpenServiceW(scm, L"qcnspmcdm", SERVICE_QUERY_CONFIG);
    if (!svc) {
        CloseServiceHandle(scm);
        return {};
    }
    DWORD bytes = 0;
    QueryServiceConfigW(svc, nullptr, 0, &bytes);
    std::vector<char> storage(bytes);
    auto * cfg = reinterpret_cast<QUERY_SERVICE_CONFIGW *>(storage.data());
    std::string dir;
    if (QueryServiceConfigW(svc, cfg, bytes, &bytes) && cfg->lpBinaryPathName) {
        std::wstring path = cfg->lpBinaryPathName;
        auto slash = path.find_last_of(L'\\');
        if (slash != std::wstring::npos) path.resize(slash);
        const std::wstring root = L"\\SystemRoot";
        if (path.rfind(root, 0) == 0) {
            wchar_t windir[MAX_PATH] = {};
            GetEnvironmentVariableW(L"windir", windir, MAX_PATH);
            path.replace(0, root.size(), windir);
        }
        int n = WideCharToMultiByte(CP_UTF8, 0, path.data(), (int) path.size(), nullptr, 0, nullptr,
                                    nullptr);
        dir.resize(n);
        WideCharToMultiByte(CP_UTF8, 0, path.data(), (int) path.size(), dir.data(), n, nullptr,
                            nullptr);
    }
    CloseServiceHandle(svc);
    CloseServiceHandle(scm);
    return dir;
}

double qpc() {
    static const double inv = [] {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return 1.0 / static_cast<double>(f.QuadPart);
    }();
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return static_cast<double>(c.QuadPart) * inv;
}

const char * mode_name(uint32_t mode) {
    if (mode == 1) return "dma-read";
    if (mode == 2) return "hvx-copy";
    return "dma-copy";
}

}  // namespace

int main(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char * skel_dir = argc > 1
                                ? argv[1]
                                : R"(C:\Users\hotschmoe\Documents\llm_inference\native_nvfp4\island_bw\dsp)";
    // FastRPC reads this through the C runtime, which does not see
    // SetEnvironmentVariable calls made after process start.
    std::string lib_path = std::string(skel_dir) + ";";
    _putenv_s("ADSP_LIBRARY_PATH", lib_path.c_str());
    SetEnvironmentVariableA("ADSP_LIBRARY_PATH", lib_path.c_str());
    std::printf("ADSP_LIBRARY_PATH=%s\n", lib_path.c_str());
    std::string dll = driver_dir() + "\\libcdsprpc.dll";
    std::printf("cdsp %s\n", dll.c_str());
    HMODULE mod = LoadLibraryA(dll.c_str());
    if (!mod) {
        std::printf("load libcdsprpc failed %lu\n", GetLastError());
        return 1;
    }
    auto open = reinterpret_cast<open_fn>(GetProcAddress(mod, "remote_handle64_open"));
    auto invoke = reinterpret_cast<invoke_fn>(GetProcAddress(mod, "remote_handle64_invoke"));
    auto session = reinterpret_cast<session_fn>(GetProcAddress(mod, "remote_session_control"));
    auto alloc2 = reinterpret_cast<alloc2_fn>(GetProcAddress(mod, "rpcmem_alloc2"));
    auto release = reinterpret_cast<free_fn>(GetProcAddress(mod, "rpcmem_free"));
    if (!open || !invoke || !session || !alloc2 || !release) {
        std::printf("missing fastrpc exports\n");
        return 1;
    }
    int domain = 3;
    int kst = session(6, &domain, sizeof(domain));
    int cst = session(7, &domain, sizeof(domain));
    std::printf("pd kill=%d session_close=%d\n", kst, cst);
    unsigned_module uns{3, 1};
    int ust = session(2, &uns, sizeof(uns));
    std::printf("unsigned pd status=%d\n", ust);
    const char * uris_default[] = {
        "file:///libdma2_skel.so?dma_bw_skel_handle_invoke&_dom=cdsp",
        "file:///libdma2_skel.so?dma_bw_skel_handle_invoke&_modver=1.0&_dom=cdsp&_session=0",
        "file:///libdma2_skel.so?dma_bw_skel_handle_invoke&_modver=1.0&_dom=cdsp",
        "file:///libdma_bw_skel.so?dma_bw_skel_handle_invoke&_dom=cdsp",
    };
    const char * one_uri = argc > 2 ? argv[2] : nullptr;
    const char ** uris = one_uri ? &one_uri : uris_default;
    int n_uris = one_uri ? 1 : 4;
    remote_handle64 handle = 0;
    int opened = -1;
    for (int i = 0; i < n_uris; ++i) {
        const char * uri = uris[i];
        opened = open(uri, &handle);
        std::printf("open %d uri %s handle=%llu\n", opened, uri, (unsigned long long) handle);
        if (opened == 0) break;
    }
    if (opened != 0) return 1;

    const size_t sizes[] = {16ull << 20, 64ull << 20, 128ull << 20};
    const uint32_t modes[] = {0, 1, 2};
    for (size_t bytes : sizes) {
        auto * src = static_cast<uint8_t *>(alloc2(25, 1, bytes));
        auto * dst = static_cast<uint8_t *>(alloc2(25, 1, bytes));
        if (!src || !dst) {
            std::printf("rpcmem alloc failed at %zu\n", bytes);
            break;
        }
        std::memset(src, 0x3C, bytes);
        for (uint32_t mode : modes) {
            std::memset(dst, 0, bytes);
            uint32_t passes = bytes >= (128ull << 20) ? 2u : 4u;
            auto * params = static_cast<uint32_t *>(alloc2(25, 1, 64));
            auto * results = static_cast<uint32_t *>(alloc2(25, 1, 64));
            if (!params || !results) {
                std::printf("rpcmem scalar alloc failed\n");
                return 1;
            }
            params[0] = passes;
            params[1] = mode;
            params[2] = 0;
            params[3] = 0;
            results[0] = results[1] = results[2] = results[3] = 0;
            remote_arg args[4];
            args[0].buf = {src, bytes};
            args[1].buf = {params, 16};
            args[2].buf = {dst, bytes};
            args[3].buf = {results, 16};
            double t0 = qpc();
            int st = invoke(handle, kScalars, args);
            double wall = qpc() - t0;
            double moved = static_cast<double>(bytes) * passes * (mode == 1 ? 1.0 : 2.0);
            double dsp_s = results[0] / 1e6;
            double dsp_gbs = dsp_s > 0 ? moved / dsp_s / 1e9 : 0;
            std::printf(
                "RESULT mode=%s mib=%.0f passes=%u status=%d dsp_ms=%.3f dsp_gbs=%.2f "
                "wall_ms=%.3f checksum=%u vtcm=%u flag=%u\n",
                mode_name(mode), bytes / 1048576.0, passes, st, results[0] / 1e3, dsp_gbs,
                wall * 1e3, results[1], results[2], results[3]);
            release(params);
            release(results);
            if (st != 0) return st;
        }
        release(src);
        release(dst);
    }
    std::printf("done\n");
    return 0;
}
