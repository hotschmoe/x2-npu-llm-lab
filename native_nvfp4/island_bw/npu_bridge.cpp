// NPU data-bridge sweep. Rates are decimal GB/s (moved bytes / seconds / 1e9).
// Raw client buffers are what yesterday measured. rpcmem + QnnMem is the
// zero-copy path llama.cpp uses for Hexagon weights.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsvc.h>

#include <QnnInterface.h>
#include <QnnLog.h>
#include <QnnMem.h>
#include <QnnOpDef.h>
#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpMem.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr double kNominal = 228.0;
int g_graphs = 0;

void qnn_log(const char* fmt, QnnLog_Level_t level, uint64_t, va_list args) {
    if (level > QNN_LOG_LEVEL_ERROR) return;
    std::fputs("QNN ", stdout);
    std::vprintf(fmt, args);
    std::fputc('\n', stdout);
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

double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

struct Rpc {
    HMODULE module = nullptr;
    void* (*alloc2)(int, uint32_t, size_t) = nullptr;
    void* (*alloc)(int, uint32_t, int) = nullptr;
    void (*release)(void*) = nullptr;
    int (*to_fd)(void*) = nullptr;

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
        auto* cfg = reinterpret_cast<QUERY_SERVICE_CONFIGW*>(storage.data());
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
            int n = WideCharToMultiByte(CP_UTF8, 0, path.data(), (int)path.size(), nullptr, 0,
                                        nullptr, nullptr);
            dir.resize(n);
            WideCharToMultiByte(CP_UTF8, 0, path.data(), (int)path.size(), dir.data(), n, nullptr,
                                nullptr);
        }
        CloseServiceHandle(svc);
        CloseServiceHandle(scm);
        return dir;
    }

    bool open() {
        std::string dir = driver_dir();
        std::string full = dir.empty() ? "libcdsprpc.dll" : dir + "\\libcdsprpc.dll";
        std::printf("rpcmem dll %s\n", full.c_str());
        module = LoadLibraryA(full.c_str());
        if (!module) {
            std::printf("libcdsprpc load failed %lu\n", GetLastError());
            return false;
        }
        alloc2 = reinterpret_cast<decltype(alloc2)>(GetProcAddress(module, "rpcmem_alloc2"));
        alloc = reinterpret_cast<decltype(alloc)>(GetProcAddress(module, "rpcmem_alloc"));
        release = reinterpret_cast<decltype(release)>(GetProcAddress(module, "rpcmem_free"));
        to_fd = reinterpret_cast<decltype(to_fd)>(GetProcAddress(module, "rpcmem_to_fd"));
        std::printf("rpcmem alloc2=%d alloc=%d to_fd=%d\n", alloc2 != nullptr, alloc != nullptr,
                    to_fd != nullptr);
        return release && to_fd && (alloc2 || alloc);
    }

    void* allocate(size_t bytes, uint32_t flags) {
        if (alloc2) return alloc2(25, flags, bytes);
        if (bytes > 0x7fffffff) return nullptr;
        return alloc(25, flags, static_cast<int>(bytes));
    }
};

struct Qnn {
    HMODULE module = nullptr;
    const QNN_INTERFACE_VER_TYPE* api = nullptr;
    Qnn_LogHandle_t log = nullptr;
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device = nullptr;
    Qnn_ContextHandle_t context = nullptr;

    bool open() {
        const char* root = R"(C:\Qualcomm\AIStack\QAIRT\2.45.40.260406)";
        std::string lib = std::string(root) + R"(\lib\aarch64-windows-msvc)";
        std::string skel = std::string(root) + R"(\lib\hexagon-v81\unsigned)";
        SetEnvironmentVariableA("ADSP_LIBRARY_PATH", (skel + ";").c_str());
        SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
        AddDllDirectory(std::wstring(lib.begin(), lib.end()).c_str());
        module = LoadLibraryA((lib + R"(\QnnHtp.dll)").c_str());
        if (!module) return false;
        auto get = reinterpret_cast<Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*)>(
            GetProcAddress(module, "QnnInterface_getProviders"));
        const QnnInterface_t** providers = nullptr;
        uint32_t count = 0;
        if (!get || get(&providers, &count) != QNN_SUCCESS || !count) return false;
        api = &providers[0]->QNN_INTERFACE_VER_NAME;
        if (api->logCreate) api->logCreate(qnn_log, QNN_LOG_LEVEL_ERROR, &log);
        if (!api->backendCreate || api->backendCreate(log, nullptr, &backend) != QNN_SUCCESS)
            return false;
        if (api->deviceCreate && api->deviceCreate(log, nullptr, &device) != QNN_SUCCESS)
            return false;
        vote();
        return api->contextCreate &&
               api->contextCreate(backend, device, nullptr, &context) == QNN_SUCCESS;
    }

    void vote() {
        if (!api->deviceGetInfrastructure) return;
        QnnDevice_Infrastructure_t infra = nullptr;
        if (api->deviceGetInfrastructure(&infra) != QNN_SUCCESS || !infra) return;
        auto* htp = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(infra);
        uint32_t id = 0;
        if (!htp->perfInfra.createPowerConfigId ||
            htp->perfInfra.createPowerConfigId(0, 0, &id) != QNN_SUCCESS)
            return;
        QnnHtpPerfInfrastructure_PowerConfig_t power =
            QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
        power.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
        power.dcvsV3Config.contextId = id;
        power.dcvsV3Config.setDcvsEnable = 1;
        power.dcvsV3Config.dcvsEnable = 0;
        power.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
        power.dcvsV3Config.setSleepLatency = 1;
        power.dcvsV3Config.sleepLatency = 40;
        power.dcvsV3Config.setBusParams = 1;
        power.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        power.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        power.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        power.dcvsV3Config.setCoreParams = 1;
        power.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        power.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        power.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        const QnnHtpPerfInfrastructure_PowerConfig_t* cfgs[] = {&power, nullptr};
        auto st = htp->perfInfra.setPowerConfig(id, cfgs);
        std::printf("perf vote %llu\n", (unsigned long long)st);
    }
};

struct Share {
    void* data = nullptr;
    size_t bytes = 0;
    int fd = -1;
    Qnn_MemHandle_t handle = nullptr;
};

enum class Hold { Raw, StaticIn, Shared };

size_t elem_size(Qnn_DataType_t dtype) {
    if (dtype == QNN_DATATYPE_FLOAT_32) return 4;
    if (dtype == QNN_DATATYPE_FLOAT_16) return 2;
    return 1;
}

void shape_for(size_t bytes, size_t elem, uint32_t dims[4]) {
    uint64_t elements = bytes / elem;
    uint64_t spatial = elements / 128;
    dims[0] = 1;
    dims[3] = 128;
    if (spatial <= 4096) {
        dims[1] = 1;
        dims[2] = static_cast<uint32_t>(spatial);
    } else {
        dims[1] = static_cast<uint32_t>(spatial / 1024);
        dims[2] = 1024;
    }
}

Qnn_Tensor_t blank_tensor(const char* name, Qnn_TensorType_t type, Qnn_DataType_t dtype,
                          uint32_t* dims, void* data, size_t bytes, bool quant) {
    Qnn_Tensor_t t = QNN_TENSOR_INIT;
    t.v1.name = name;
    t.v1.type = type;
    t.v1.dataType = dtype;
    t.v1.rank = 4;
    t.v1.dimensions = dims;
    t.v1.memType = QNN_TENSORMEMTYPE_RAW;
    if (type == QNN_TENSOR_TYPE_STATIC) {
        t.v1.clientBuf.data = data;
        t.v1.clientBuf.dataSize = static_cast<uint32_t>(bytes);
    }
    if (quant) {
        t.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
        t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
        t.v1.quantizeParams.scaleOffsetEncoding.scale = 1.f;
        t.v1.quantizeParams.scaleOffsetEncoding.offset = 0;
    }
    return t;
}

bool register_share(Qnn& qnn, Share& share, Qnn_DataType_t dtype, uint32_t* dims,
                    const char* kind) {
    if (!qnn.api->memRegister || share.fd < 0) return false;
    QnnMemHtp_Descriptor_t htp{};
    htp.type = QNN_HTP_MEM_SHARED_BUFFER;
    htp.size = share.bytes;
    htp.sharedBufferConfig.fd = share.fd;
    htp.sharedBufferConfig.offset = 0;
    Qnn_MemDescriptor_t desc = QNN_MEM_DESCRIPTOR_INIT;
    desc.memShape.numDim = 4;
    desc.memShape.dimSize = dims;
    desc.dataType = dtype;
    Qnn_ErrorHandle_t st = QNN_MEM_ERROR_UNSUPPORTED_MEMTYPE;
    if (std::strcmp(kind, "custom") == 0) {
        desc.memType = QNN_MEM_TYPE_CUSTOM;
        desc.customInfo = &htp;
        st = qnn.api->memRegister(qnn.context, &desc, 1, &share.handle);
    } else if (std::strcmp(kind, "ion") == 0) {
        desc.memType = QNN_MEM_TYPE_ION;
        desc.ionInfo.fd = share.fd;
        st = qnn.api->memRegister(qnn.context, &desc, 1, &share.handle);
    } else {
        desc.memType = QNN_MEM_TYPE_DMA_BUF;
        desc.dmaBufInfo.fd = share.fd;
        desc.dmaBufInfo.data = share.data;
        st = qnn.api->memRegister(qnn.context, &desc, 1, &share.handle);
    }
    std::printf("  memRegister %s status=%llu handle=%p\n", kind, (unsigned long long)st,
                share.handle);
    return st == QNN_SUCCESS && share.handle;
}

uint64_t checksum(const void* data, size_t bytes) {
    const auto* p = static_cast<const uint8_t*>(data);
    uint64_t sum = 0;
    size_t n = std::min<size_t>(bytes, 4096);
    for (size_t i = 0; i < n; i += 64) sum += p[i];
    return sum;
}

struct Case {
    const char* name;
    const char* op;
    Qnn_DataType_t dtype;
    int inputs;
    Hold hold;
    const char* share_kind;
    int depth;
    size_t passes;
};

bool run_case(Qnn& qnn, Rpc& rpc, const Case& c, size_t bytes) {
    size_t elem = elem_size(c.dtype);
    if (bytes % (elem * 128) != 0) return false;
    uint32_t dims[4];
    shape_for(bytes, elem, dims);
    bool quant = c.dtype == QNN_DATATYPE_UFIXED_POINT_8;
    int tensors = c.inputs == 2 ? 2 : 1;
    int slots = tensors + std::max(c.depth, 1);
    std::vector<Share> shares(slots);
    auto fill = [&](Share& s) {
        s.bytes = bytes;
        if (c.hold == Hold::Shared) {
            s.data = rpc.allocate(bytes, 1);
            if (!s.data) return false;
            s.fd = rpc.to_fd(s.data);
        } else {
            s.data = VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        }
        if (!s.data) return false;
        std::memset(s.data, 0x3C, bytes);
        return true;
    };
    for (int i = 0; i < slots; ++i) {
        if (!fill(shares[i])) {
            std::printf("RESULT case=%s bytes=%zu status=alloc-fail\n", c.name, bytes);
            return false;
        }
    }
    std::vector<Qnn_MemHandle_t> handles;
    if (c.hold == Hold::Shared) {
        for (int i = 0; i < slots; ++i) {
            if (!register_share(qnn, shares[i], c.dtype, dims, c.share_kind)) {
                std::printf("RESULT case=%s mib=%.0f status=share-fail\n", c.name,
                            bytes / 1048576.0);
                for (auto& s : shares)
                    if (s.data && rpc.release) rpc.release(s.data);
                return false;
            }
            handles.push_back(shares[i].handle);
        }
    }

    std::string gname = std::string(c.name) + "_" + std::to_string(bytes) + "_" +
                        std::to_string(++g_graphs);
    std::vector<std::string> names;
    names.reserve(8);
    auto nm = [&](const char* p, int i) {
        names.push_back(gname + "_" + p + std::to_string(i));
        return names.back().c_str();
    };
    std::vector<uint32_t> dimstore(8 * 4);
    auto dims_of = [&](int i) { return dimstore.data() + i * 4; };
    for (int i = 0; i < 8; ++i) std::memcpy(dims_of(i), dims, sizeof(dims));

    Qnn_GraphHandle_t graph = nullptr;
    if (qnn.api->graphCreate(qnn.context, gname.c_str(), nullptr, &graph) != QNN_SUCCESS) {
        std::printf("RESULT case=%s status=graph-fail\n", c.name);
        return false;
    }
    auto bind = [&](Qnn_Tensor_t& t, Share& s, bool as_static) {
        if (c.hold == Hold::Shared) {
            t.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
            t.v1.memHandle = s.handle;
            t.v1.clientBuf.data = nullptr;
            t.v1.clientBuf.dataSize = 0;
        } else if (!as_static) {
            t.v1.clientBuf.data = nullptr;
            t.v1.clientBuf.dataSize = 0;
        }
    };
    int produced = 0;
    std::vector<Qnn_Tensor_t> live;
    live.reserve(8);
    Qnn_TensorType_t in_type =
        c.hold == Hold::StaticIn ? QNN_TENSOR_TYPE_STATIC : QNN_TENSOR_TYPE_APP_WRITE;
    Qnn_Tensor_t in0 =
        blank_tensor(nm("in", 0), in_type, c.dtype, dims_of(0), shares[0].data, bytes, quant);
    bind(in0, shares[0], c.hold == Hold::StaticIn);
    Qnn_Tensor_t in1 =
        blank_tensor(nm("in", 1), in_type, c.dtype, dims_of(1), shares[1].data, bytes, quant);
    if (tensors == 2) bind(in1, shares[1], c.hold == Hold::StaticIn);
    if (qnn.api->tensorCreateGraphTensor(graph, &in0) != QNN_SUCCESS ||
        (tensors == 2 && qnn.api->tensorCreateGraphTensor(graph, &in1) != QNN_SUCCESS)) {
        std::printf("RESULT case=%s mib=%.0f status=tensor-fail\n", c.name, bytes / 1048576.0);
        return false;
    }
    live.push_back(in0);
    if (tensors == 2) live.push_back(in1);
    Qnn_Tensor_t cursor = in0;
    int stages = std::max(c.depth, 1);
    for (int stage = 0; stage < stages; ++stage) {
        bool last = stage + 1 == stages;
        int out_slot = tensors + stage;
        Qnn_TensorType_t out_type = last ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
        Qnn_Tensor_t out = blank_tensor(nm("y", stage), out_type, c.dtype, dims_of(2 + stage),
                                        shares[out_slot].data, bytes, quant);
        if (last || c.hold == Hold::Shared) bind(out, shares[out_slot], false);
        if (qnn.api->tensorCreateGraphTensor(graph, &out) != QNN_SUCCESS) {
            std::printf("RESULT case=%s status=out-tensor-fail\n", c.name);
            return false;
        }
        Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
        op.version = QNN_OPCONFIG_VERSION_1;
        names.push_back(gname + "_op" + std::to_string(stage));
        op.v1.name = names.back().c_str();
        op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
        op.v1.typeName = c.op;
        Qnn_Tensor_t op_in[2] = {cursor, tensors == 2 && stage == 0 ? in1 : cursor};
        op.v1.numOfInputs = (tensors == 2 && stage == 0) ? 2 : 1;
        op.v1.inputTensors = op_in;
        op.v1.numOfOutputs = 1;
        op.v1.outputTensors = &out;
        auto st = qnn.api->graphAddNode(graph, op);
        if (st != QNN_SUCCESS) {
            std::printf("RESULT case=%s mib=%.0f status=add-%llu\n", c.name, bytes / 1048576.0,
                        (unsigned long long)st);
            return false;
        }
        cursor = out;
        produced = out_slot;
        if (!last) live.push_back(cursor);
    }
    Qnn_Tensor_t output = cursor;
    auto fin = qnn.api->graphFinalize(graph, nullptr, nullptr);
    if (fin != QNN_SUCCESS) {
        std::printf("RESULT case=%s mib=%.0f status=finalize-%llu\n", c.name, bytes / 1048576.0,
                    (unsigned long long)fin);
        return false;
    }
    if (c.hold == Hold::Shared) {
        in0.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        in0.v1.memHandle = shares[0].handle;
        if (tensors == 2) {
            in1.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
            in1.v1.memHandle = shares[1].handle;
        }
        output.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        output.v1.memHandle = shares[produced].handle;
        std::printf("  execute handles in=%p out=%p\n", shares[0].handle, shares[produced].handle);
    } else if (c.hold != Hold::StaticIn) {
        in0.v1.clientBuf = {shares[0].data, static_cast<uint32_t>(bytes)};
        if (tensors == 2) in1.v1.clientBuf = {shares[1].data, static_cast<uint32_t>(bytes)};
        output.v1.clientBuf = {shares[produced].data, static_cast<uint32_t>(bytes)};
    } else {
        output.v1.clientBuf = {shares[produced].data, static_cast<uint32_t>(bytes)};
    }
    Qnn_Tensor_t exec_in[2] = {in0, in1};
    std::vector<double> samples;
    Qnn_ErrorHandle_t exec = QNN_SUCCESS;
    for (int i = 0; i < 5; ++i) {
        double t0 = qpc();
        exec = qnn.api->graphExecute(graph, exec_in, tensors, &output, 1, nullptr, nullptr);
        double dt = qpc() - t0;
        if (exec != QNN_SUCCESS) break;
        if (i) samples.push_back(dt);
    }
    if (exec != QNN_SUCCESS || samples.empty()) {
        std::printf("RESULT case=%s mib=%.0f status=exec-%llu\n", c.name, bytes / 1048576.0,
                    (unsigned long long)exec);
    } else {
        size_t moved = bytes * static_cast<size_t>(c.passes);
        double ms = median(samples) * 1e3;
        double gbs = moved / (median(samples) * 1e9);
        std::printf(
            "RESULT case=%s op=%s dtype=0x%x hold=%s depth=%d mib=%.0f moved_mib=%.0f "
            "median_ms=%.3f gbs=%.2f pct228=%.1f checksum=%llu\n",
            c.name, c.op, c.dtype, c.hold == Hold::Raw ? "raw" : c.hold == Hold::StaticIn ? "static" : c.share_kind,
            c.depth, bytes / 1048576.0, moved / 1048576.0, ms, gbs, 100.0 * gbs / kNominal,
            (unsigned long long)checksum(shares[produced].data, bytes));
    }
    if (qnn.api->memDeRegister && !handles.empty())
        qnn.api->memDeRegister(handles.data(), (uint32_t)handles.size());
    for (auto& s : shares) {
        if (!s.data) continue;
        if (c.hold == Hold::Shared && rpc.release) rpc.release(s.data);
        else VirtualFree(s.data, 0, MEM_RELEASE);
    }
    return exec == QNN_SUCCESS;
}

void host_copy(size_t bytes) {
    void* a = VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    void* b = VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    std::memset(a, 0x3C, bytes);
    std::vector<double> samples;
    for (int i = 0; i < 5; ++i) {
        double t0 = qpc();
        std::memcpy(b, a, bytes);
        if (i) samples.push_back(qpc() - t0);
    }
    double gbs = (bytes * 2) / (median(samples) * 1e9);
    std::printf("RESULT case=host-memcpy mib=%.0f moved_mib=%.0f median_ms=%.3f gbs=%.2f\n",
                bytes / 1048576.0, (bytes * 2) / 1048576.0, median(samples) * 1e3, gbs);
    VirtualFree(a, 0, MEM_RELEASE);
    VirtualFree(b, 0, MEM_RELEASE);
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const bool share_only = argc > 1 && std::strcmp(argv[1], "share") == 0;
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    std::printf("npu bridge sweep nominal=%.0f\n", kNominal);
    host_copy(128ull << 20);
    Rpc rpc;
    bool have_rpc = rpc.open();
    Qnn qnn;
    if (!qnn.open()) {
        std::printf("QNN open failed\n");
        return 1;
    }
    const size_t sizes[] = {8ull << 20, 32ull << 20, 128ull << 20, 256ull << 20};
    Case raw_fp16{"raw-fp16-relu", QNN_OP_RELU, QNN_DATATYPE_FLOAT_16, 1, Hold::Raw, "", 1, 2};
    Case raw_u8{"raw-u8-relu", QNN_OP_RELU, QNN_DATATYPE_UFIXED_POINT_8, 1, Hold::Raw, "", 1, 2};
    Case raw_fp32{"raw-fp32-relu", QNN_OP_RELU, QNN_DATATYPE_FLOAT_32, 1, Hold::Raw, "", 1, 2};
    Case raw_add{"raw-fp16-add", QNN_OP_ELEMENT_WISE_ADD, QNN_DATATYPE_FLOAT_16, 2, Hold::Raw, "", 1, 3};
    Case static_fp16{"static-fp16-relu", QNN_OP_RELU, QNN_DATATYPE_FLOAT_16, 1, Hold::StaticIn, "", 1, 2};
    Case deep{"raw-fp16-relu-x4", QNN_OP_RELU, QNN_DATATYPE_FLOAT_16, 1, Hold::Raw, "", 4, 8};
    Case ion{"ion-fp16-relu", QNN_OP_RELU, QNN_DATATYPE_FLOAT_16, 1, Hold::Shared, "ion", 1, 2};
    Case dma{"dmabuf-fp16-relu", QNN_OP_RELU, QNN_DATATYPE_FLOAT_16, 1, Hold::Shared, "dma", 1, 2};
    Case custom{"custom-fp16-relu", QNN_OP_RELU, QNN_DATATYPE_FLOAT_16, 1, Hold::Shared, "custom", 1, 2};
    for (size_t bytes : sizes) {
        if (share_only) {
            if (bytes == (8ull << 20) || bytes == (256ull << 20)) continue;
            run_case(qnn, rpc, raw_fp16, bytes);
            if (have_rpc) run_case(qnn, rpc, custom, bytes);
            continue;
        }
        run_case(qnn, rpc, raw_fp16, bytes);
        if (bytes <= (128ull << 20)) {
            run_case(qnn, rpc, raw_u8, bytes);
            run_case(qnn, rpc, raw_fp32, bytes);
            run_case(qnn, rpc, raw_add, bytes);
            run_case(qnn, rpc, static_fp16, bytes);
        }
        if (bytes == (32ull << 20) || bytes == (128ull << 20)) run_case(qnn, rpc, deep, bytes);
        if (have_rpc && bytes <= (128ull << 20)) {
            bool ok = run_case(qnn, rpc, custom, bytes);
            if (!ok) ok = run_case(qnn, rpc, ion, bytes);
            if (!ok) run_case(qnn, rpc, dma, bytes);
        }
    }
    std::printf("done\n");
    return 0;
}
