// Streaming-bandwidth probe for the three Snapdragon X2 memory clients.
// Rates are decimal GB/s (bytes/s / 1e9), the same unit as the 228 GB/s
// LPDDR pin rate. A read moves N bytes; copy moves 2N; triad moves 3N.

#define WIN32_LEAN_AND_MEAN
#define CL_TARGET_OPENCL_VERSION 300
#include <windows.h>

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_ext_qcom.h>

#include <arm_neon.h>

#include <QnnInterface.h>
#include <QnnLog.h>
#include <QnnOpDef.h>
#include <QnnProfile.h>
#include <HTP/QnnHtpDevice.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr double kNominalGbs = 228.0;
constexpr int kSamples = 5;

struct Buffer {
    void* data = nullptr;
    size_t bytes = 0;

    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept : data(other.data), bytes(other.bytes) {
        other.data = nullptr;
        other.bytes = 0;
    }
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            reset();
            data = other.data;
            bytes = other.bytes;
            other.data = nullptr;
            other.bytes = 0;
        }
        return *this;
    }
    ~Buffer() { reset(); }

    void reset() {
        if (data) VirtualFree(data, 0, MEM_RELEASE);
        data = nullptr;
        bytes = 0;
    }

    explicit operator bool() const { return data != nullptr; }
    uint8_t* bytes_ptr() const { return static_cast<uint8_t*>(data); }
};

Buffer allocate(size_t bytes) {
    Buffer buffer;
    buffer.data = VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    buffer.bytes = buffer.data ? bytes : 0;
    return buffer;
}

void fill_bytes(Buffer& buffer, uint8_t value) {
    if (buffer) std::memset(buffer.data, value, buffer.bytes);
}

FILE* g_log = nullptr;

void say(const char* text) {
    std::fputs(text, stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
    if (g_log) {
        std::fputs(text, g_log);
        std::fputc('\n', g_log);
        std::fflush(g_log);
    }
}

double qpc_seconds() {
    static const double inv = [] {
        LARGE_INTEGER frequency;
        QueryPerformanceFrequency(&frequency);
        return 1.0 / static_cast<double>(frequency.QuadPart);
    }();
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return static_cast<double>(counter.QuadPart) * inv;
}

double gbs(size_t bytes, double seconds) {
    return seconds > 0.0 ? static_cast<double>(bytes) / seconds / 1.0e9 : 0.0;
}

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if (values.size() % 2) return values[mid];
    return 0.5 * (values[mid - 1] + values[mid]);
}

void report(const char* island, const char* op, const char* detail, int threads,
            size_t buffer_bytes, size_t traffic_bytes, const std::vector<double>& seconds,
            uint64_t checksum) {
    const double med = median(seconds);
    const double rate = gbs(traffic_bytes, med);
    std::printf(
        "RESULT island=%s op=%s detail=%s threads=%d buffer_mib=%.1f traffic_mib=%.1f "
        "median_ms=%.3f gbs=%.2f pct_of_228=%.1f checksum=%llu\n",
        island, op, detail, threads, buffer_bytes / 1048576.0, traffic_bytes / 1048576.0,
        med * 1.0e3, rate, 100.0 * rate / kNominalGbs,
        static_cast<unsigned long long>(checksum));
    std::fflush(stdout);
}

struct CpuCore {
    uint32_t logical = 0;
    uint8_t efficiency = 0;
    uint8_t llc = 0;
};

std::vector<CpuCore> cpu_topology() {
    DWORD bytes = 0;
    GetSystemCpuSetInformation(nullptr, 0, &bytes, nullptr, 0);
    std::vector<char> storage(bytes);
    std::vector<CpuCore> cores;
    if (!GetSystemCpuSetInformation(
            reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(storage.data()), bytes, &bytes,
            nullptr, 0)) {
        return cores;
    }
    auto* cursor = reinterpret_cast<SYSTEM_CPU_SET_INFORMATION*>(storage.data());
    const auto* end = storage.data() + bytes;
    while (reinterpret_cast<char*>(cursor) < end) {
        if (cursor->Type == CpuSetInformation) {
            const auto& cpu = cursor->CpuSet;
            cores.push_back(CpuCore{cpu.LogicalProcessorIndex, cpu.EfficiencyClass,
                                     static_cast<uint8_t>(cpu.LastLevelCacheIndex)});
        }
        cursor = reinterpret_cast<SYSTEM_CPU_SET_INFORMATION*>(
            reinterpret_cast<char*>(cursor) + cursor->Size);
    }
    return cores;
}

void pin_current_thread(uint32_t logical) {
    SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << logical);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
}

uint64_t fold_u8(uint8x16_t value) {
    const uint64x2_t lanes = vreinterpretq_u64_u8(value);
    return vgetq_lane_u64(lanes, 0) ^ vgetq_lane_u64(lanes, 1);
}

uint64_t neon_read(const uint8_t* data, size_t begin, size_t end) {
    uint8x16_t acc = vdupq_n_u8(0);
    size_t offset = begin;
    for (; offset + 128 <= end; offset += 128) {
        __builtin_prefetch(data + offset + 512, 0, 0);
        uint8x16_t a0 = vld1q_u8(data + offset);
        uint8x16_t a1 = vld1q_u8(data + offset + 16);
        uint8x16_t a2 = vld1q_u8(data + offset + 32);
        uint8x16_t a3 = vld1q_u8(data + offset + 48);
        uint8x16_t a4 = vld1q_u8(data + offset + 64);
        uint8x16_t a5 = vld1q_u8(data + offset + 80);
        uint8x16_t a6 = vld1q_u8(data + offset + 96);
        uint8x16_t a7 = vld1q_u8(data + offset + 112);
        acc = veorq_u8(acc, veorq_u8(veorq_u8(a0, a1), veorq_u8(a2, a3)));
        acc = veorq_u8(acc, veorq_u8(veorq_u8(a4, a5), veorq_u8(a6, a7)));
    }
    for (; offset + 16 <= end; offset += 16) {
        acc = veorq_u8(acc, vld1q_u8(data + offset));
    }
    return fold_u8(acc);
}

uint64_t neon_copy(const uint8_t* src, uint8_t* dst, size_t begin, size_t end) {
    uint8x16_t acc = vdupq_n_u8(0);
    size_t offset = begin;
    for (; offset + 64 <= end; offset += 64) {
        uint8x16_t a0 = vld1q_u8(src + offset);
        uint8x16_t a1 = vld1q_u8(src + offset + 16);
        uint8x16_t a2 = vld1q_u8(src + offset + 32);
        uint8x16_t a3 = vld1q_u8(src + offset + 48);
        vst1q_u8(dst + offset, a0);
        vst1q_u8(dst + offset + 16, a1);
        vst1q_u8(dst + offset + 32, a2);
        vst1q_u8(dst + offset + 48, a3);
        acc = veorq_u8(acc, a0);
    }
    return fold_u8(acc);
}

uint64_t neon_triad(const uint8_t* a, const uint8_t* b, uint8_t* c, size_t begin, size_t end) {
    const uint8x16_t scale = vdupq_n_u8(3);
    uint8x16_t acc = vdupq_n_u8(0);
    size_t offset = begin;
    for (; offset + 64 <= end; offset += 64) {
        uint8x16_t a0 = vld1q_u8(a + offset);
        uint8x16_t b0 = vld1q_u8(b + offset);
        uint8x16_t a1 = vld1q_u8(a + offset + 16);
        uint8x16_t b1 = vld1q_u8(b + offset + 16);
        uint8x16_t a2 = vld1q_u8(a + offset + 32);
        uint8x16_t b2 = vld1q_u8(b + offset + 32);
        uint8x16_t a3 = vld1q_u8(a + offset + 48);
        uint8x16_t b3 = vld1q_u8(b + offset + 48);
        uint8x16_t c0 = veorq_u8(a0, vandq_u8(b0, scale));
        uint8x16_t c1 = veorq_u8(a1, vandq_u8(b1, scale));
        uint8x16_t c2 = veorq_u8(a2, vandq_u8(b2, scale));
        uint8x16_t c3 = veorq_u8(a3, vandq_u8(b3, scale));
        vst1q_u8(c + offset, c0);
        vst1q_u8(c + offset + 16, c1);
        vst1q_u8(c + offset + 32, c2);
        vst1q_u8(c + offset + 48, c3);
        acc = veorq_u8(acc, c0);
    }
    return fold_u8(acc);
}

struct CpuJob {
    const char* op = "";
    const Buffer* a = nullptr;
    const Buffer* b = nullptr;
    Buffer* c = nullptr;
    int passes = 1;
};

double g_last_cpu_seconds = 0.0;
bool g_cpu_llc_only = false;

uint64_t run_cpu_job(const CpuJob& job, const std::vector<uint32_t>& logicals) {
    const unsigned workers = static_cast<unsigned>(logicals.size());
    const size_t bytes = job.a->bytes;
    std::atomic<unsigned> ready{0};
    std::atomic<int> go{0};
    std::atomic<unsigned> done{0};
    std::vector<uint64_t> partials(workers, 0);
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (unsigned worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&, worker] {
            pin_current_thread(logicals[worker]);
            const size_t begin = (bytes * worker / workers) & ~size_t{63};
            const size_t end = worker + 1 == workers
                                   ? bytes
                                   : (bytes * (worker + 1) / workers) & ~size_t{63};
            ready.fetch_add(1, std::memory_order_release);
            while (go.load(std::memory_order_acquire) == 0) {
                YieldProcessor();
            }
            uint64_t checksum = 0;
            for (int pass = 0; pass < job.passes; ++pass) {
                if (job.op[0] == 'r') {
                    checksum ^= neon_read(job.a->bytes_ptr(), begin, end);
                } else if (job.op[0] == 'c') {
                    checksum ^= neon_copy(job.a->bytes_ptr(), job.c->bytes_ptr(), begin, end);
                } else {
                    checksum ^= neon_triad(job.a->bytes_ptr(), job.b->bytes_ptr(),
                                           job.c->bytes_ptr(), begin, end);
                }
            }
            partials[worker] = checksum;
            done.fetch_add(1, std::memory_order_release);
        });
    }
    while (ready.load(std::memory_order_acquire) < workers) YieldProcessor();
    const double started = qpc_seconds();
    go.store(1, std::memory_order_release);
    while (done.load(std::memory_order_acquire) < workers) YieldProcessor();
    const double elapsed = qpc_seconds() - started;
    for (std::thread& thread : threads) thread.join();
    uint64_t checksum = 0;
    for (uint64_t partial : partials) checksum ^= partial;
    // Stash elapsed in the high unused path by returning checksum only; caller times
    // independently. The elapsed value is recovered through a side channel below.
    g_last_cpu_seconds = elapsed;
    return checksum;
}

void bench_cpu() {
    const auto cores = cpu_topology();
    std::printf("CPU topology: %zu logical processors\n", cores.size());
    for (const CpuCore& core : cores) {
        std::printf("  cpu %u efficiency_class=%u llc=%u\n", core.logical, core.efficiency,
                    core.llc);
    }
    std::vector<uint32_t> classes;
    for (const CpuCore& core : cores) {
        if (std::find(classes.begin(), classes.end(), core.efficiency) == classes.end()) {
            classes.push_back(core.efficiency);
        }
    }
    struct Group {
        std::string name;
        std::vector<uint32_t> logicals;
    };
    std::vector<Group> groups;
    for (uint32_t efficiency : classes) {
        Group one;
        Group all;
        one.name = "class" + std::to_string(efficiency) + "-1t";
        all.name = "class" + std::to_string(efficiency) + "-all";
        for (const CpuCore& core : cores) {
            if (core.efficiency == efficiency) {
                if (one.logicals.empty()) one.logicals.push_back(core.logical);
                all.logicals.push_back(core.logical);
            }
        }
        groups.push_back(std::move(one));
        groups.push_back(std::move(all));
    }
    Group every{"all-cores", {}};
    for (const CpuCore& core : cores) every.logicals.push_back(core.logical);
    if (g_cpu_llc_only) {
        groups.clear();
        std::vector<uint8_t> llcs;
        for (const CpuCore& core : cores) {
            if (std::find(llcs.begin(), llcs.end(), core.llc) == llcs.end()) llcs.push_back(core.llc);
        }
        for (uint8_t llc : llcs) {
            Group group;
            group.name = "llc" + std::to_string(llc);
            for (const CpuCore& core : cores) {
                if (core.llc == llc) group.logicals.push_back(core.logical);
            }
            groups.push_back(std::move(group));
        }
    }
    groups.push_back(std::move(every));

    const size_t sizes_full[] = {256ull << 20, 1024ull << 20};
    const size_t sizes_llc[] = {1024ull << 20};
    const size_t* sizes = g_cpu_llc_only ? sizes_llc : sizes_full;
    const size_t size_count = g_cpu_llc_only ? 1 : 2;
    for (size_t size_index = 0; size_index < size_count; ++size_index) {
        const size_t bytes = sizes[size_index];
        Buffer a = allocate(bytes);
        Buffer b = allocate(bytes);
        Buffer c = allocate(bytes);
        if (!a || !b || !c) {
            std::printf("CPU alloc failed at %zu bytes\n", bytes);
            break;
        }
        fill_bytes(a, 0xA5);
        fill_bytes(b, 0x5A);
        fill_bytes(c, 0x00);
        for (const Group& group : groups) {
            const int passes = bytes >= (1024ull << 20) ? 2 : 4;
            auto time_op = [&](const char* op, size_t traffic_per_pass) {
                CpuJob job;
                job.op = op;
                job.a = &a;
                job.b = &b;
                job.c = &c;
                job.passes = passes;
                std::vector<double> samples;
                uint64_t checksum = 0;
                for (int sample = 0; sample < kSamples + 1; ++sample) {
                    checksum = run_cpu_job(job, group.logicals);
                    if (sample > 0) samples.push_back(g_last_cpu_seconds);
                }
                report("cpu", op, group.name.c_str(), static_cast<int>(group.logicals.size()),
                       bytes, traffic_per_pass * static_cast<size_t>(passes), samples, checksum);
            };
            time_op("read", bytes);
            if (group.logicals.size() > 1) {
                time_op("copy", bytes * 2);
                time_op("triad", bytes * 3);
            }
        }
    }
}

using ClGetPlatformIDsFn = cl_int(CL_API_CALL*)(cl_uint, cl_platform_id*, cl_uint*);
using ClGetDeviceIDsFn = cl_int(CL_API_CALL*)(cl_platform_id, cl_device_type, cl_uint,
                                              cl_device_id*, cl_uint*);
using ClGetDeviceInfoFn = cl_int(CL_API_CALL*)(cl_device_id, cl_device_info, size_t, void*,
                                               size_t*);
using ClCreateContextFn = cl_context(CL_API_CALL*)(const cl_context_properties*, cl_uint,
                                                  const cl_device_id*,
                                                  void(CL_CALLBACK*)(const char*, const void*,
                                                                     size_t, void*),
                                                  void*, cl_int*);
using ClCreateCommandQueueFn = cl_command_queue(CL_API_CALL*)(cl_context, cl_device_id,
                                                              cl_command_queue_properties,
                                                              cl_int*);
using ClCreateProgramWithSourceFn = cl_program(CL_API_CALL*)(cl_context, cl_uint, const char**,
                                                             const size_t*, cl_int*);
using ClBuildProgramFn = cl_int(CL_API_CALL*)(cl_program, cl_uint, const cl_device_id*,
                                              const char*,
                                              void(CL_CALLBACK*)(cl_program, void*), void*);
using ClGetProgramBuildInfoFn = cl_int(CL_API_CALL*)(cl_program, cl_device_id,
                                                     cl_program_build_info, size_t, void*,
                                                     size_t*);
using ClCreateKernelFn = cl_kernel(CL_API_CALL*)(cl_program, const char*, cl_int*);
using ClCreateBufferFn = cl_mem(CL_API_CALL*)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
using ClSetKernelArgFn = cl_int(CL_API_CALL*)(cl_kernel, cl_uint, size_t, const void*);
using ClEnqueueNDRangeKernelFn = cl_int(CL_API_CALL*)(cl_command_queue, cl_kernel, cl_uint,
                                                      const size_t*, const size_t*,
                                                      const size_t*, cl_uint, const cl_event*,
                                                      cl_event*);
using ClEnqueueFillBufferFn = cl_int(CL_API_CALL*)(cl_command_queue, cl_mem, const void*,
                                                   size_t, size_t, size_t, cl_uint,
                                                   const cl_event*, cl_event*);
using ClFinishFn = cl_int(CL_API_CALL*)(cl_command_queue);
using ClWaitForEventsFn = cl_int(CL_API_CALL*)(cl_uint, const cl_event*);
using ClGetEventProfilingInfoFn = cl_int(CL_API_CALL*)(cl_event, cl_profiling_info, size_t,
                                                       void*, size_t*);
using ClReleaseEventFn = cl_int(CL_API_CALL*)(cl_event);
using ClReleaseMemObjectFn = cl_int(CL_API_CALL*)(cl_mem);
using ClReleaseKernelFn = cl_int(CL_API_CALL*)(cl_kernel);
using ClReleaseProgramFn = cl_int(CL_API_CALL*)(cl_program);
using ClReleaseCommandQueueFn = cl_int(CL_API_CALL*)(cl_command_queue);
using ClReleaseContextFn = cl_int(CL_API_CALL*)(cl_context);

struct OpenCl {
    HMODULE module = nullptr;
    ClGetPlatformIDsFn getPlatformIDs = nullptr;
    ClGetDeviceIDsFn getDeviceIDs = nullptr;
    ClGetDeviceInfoFn getDeviceInfo = nullptr;
    ClCreateContextFn createContext = nullptr;
    ClCreateCommandQueueFn createCommandQueue = nullptr;
    ClCreateProgramWithSourceFn createProgramWithSource = nullptr;
    ClBuildProgramFn buildProgram = nullptr;
    ClGetProgramBuildInfoFn getProgramBuildInfo = nullptr;
    ClCreateKernelFn createKernel = nullptr;
    ClCreateBufferFn createBuffer = nullptr;
    ClSetKernelArgFn setKernelArg = nullptr;
    ClEnqueueNDRangeKernelFn enqueueNDRangeKernel = nullptr;
    ClEnqueueFillBufferFn enqueueFillBuffer = nullptr;
    ClFinishFn finish = nullptr;
    ClWaitForEventsFn waitForEvents = nullptr;
    ClGetEventProfilingInfoFn getEventProfilingInfo = nullptr;
    ClReleaseEventFn releaseEvent = nullptr;
    ClReleaseMemObjectFn releaseMemObject = nullptr;
    ClReleaseKernelFn releaseKernel = nullptr;
    ClReleaseProgramFn releaseProgram = nullptr;
    ClReleaseCommandQueueFn releaseCommandQueue = nullptr;
    ClReleaseContextFn releaseContext = nullptr;

    bool load() {
        say("gpu: LoadLibrary OpenCL.dll");
        module = LoadLibraryA("OpenCL.dll");
        if (!module) {
            say("gpu: LoadLibrary failed");
            return false;
        }
        say("gpu: resolving entry points");
#define LOAD(field, symbol)                                                       \
    field = reinterpret_cast<decltype(field)>(GetProcAddress(module, symbol));   \
    if (!field) return false
        LOAD(getPlatformIDs, "clGetPlatformIDs");
        LOAD(getDeviceIDs, "clGetDeviceIDs");
        LOAD(getDeviceInfo, "clGetDeviceInfo");
        LOAD(createContext, "clCreateContext");
        LOAD(createCommandQueue, "clCreateCommandQueue");
        LOAD(createProgramWithSource, "clCreateProgramWithSource");
        LOAD(buildProgram, "clBuildProgram");
        LOAD(getProgramBuildInfo, "clGetProgramBuildInfo");
        LOAD(createKernel, "clCreateKernel");
        LOAD(createBuffer, "clCreateBuffer");
        LOAD(setKernelArg, "clSetKernelArg");
        LOAD(enqueueNDRangeKernel, "clEnqueueNDRangeKernel");
        LOAD(enqueueFillBuffer, "clEnqueueFillBuffer");
        LOAD(finish, "clFinish");
        LOAD(waitForEvents, "clWaitForEvents");
        LOAD(getEventProfilingInfo, "clGetEventProfilingInfo");
        LOAD(releaseEvent, "clReleaseEvent");
        LOAD(releaseMemObject, "clReleaseMemObject");
        LOAD(releaseKernel, "clReleaseKernel");
        LOAD(releaseProgram, "clReleaseProgram");
        LOAD(releaseCommandQueue, "clReleaseCommandQueue");
        LOAD(releaseContext, "clReleaseContext");
#undef LOAD
        return true;
    }
};

const char* kGpuSource = R"CLC(
__attribute__((reqd_work_group_size(256, 1, 1)))
kernel void read16(global const uint4* src, global uint* partials, ulong n) {
    size_t i = get_global_id(0);
    uint acc = 0;
    if (i < n) {
        uint4 v = src[i];
        acc = v.s0 + v.s1 + v.s2 + v.s3;
    }
    local uint scratch[256];
    uint lid = get_local_id(0);
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = 128; off > 0; off >>= 1) {
        if (lid < off) scratch[lid] += scratch[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) partials[get_group_id(0)] = scratch[0];
}
__attribute__((reqd_work_group_size(256, 1, 1)))
kernel void read64(global const uint16* src, global uint* partials, ulong n) {
    size_t i = get_global_id(0);
    uint acc = 0;
    if (i < n) {
        uint16 v = src[i];
        acc = v.s0 + v.s1 + v.s2 + v.s3 + v.s4 + v.s5 + v.s6 + v.s7 +
              v.s8 + v.s9 + v.sa + v.sb + v.sc + v.sd + v.se + v.sf;
    }
    local uint scratch[256];
    uint lid = get_local_id(0);
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = 128; off > 0; off >>= 1) {
        if (lid < off) scratch[lid] += scratch[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) partials[get_group_id(0)] = scratch[0];
}
__attribute__((reqd_work_group_size(256, 1, 1)))
kernel void copy4(global const float4* a, global float4* c, ulong n) {
    size_t i = get_global_id(0);
    if (i < n) c[i] = a[i];
}
__attribute__((reqd_work_group_size(256, 1, 1)))
kernel void triad4(global const float4* a, global const float4* b, global float4* c,
                   ulong n, float s) {
    size_t i = get_global_id(0);
    if (i < n) c[i] = a[i] + s * b[i];
}
)CLC";

double profiled_ns(OpenCl& cl, cl_event event) {
    cl_ulong start = 0;
    cl_ulong end = 0;
    cl.getEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
    cl.getEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
    return static_cast<double>(end - start);
}

void bench_gpu() {
    say("gpu: load OpenCL");
    OpenCl cl;
    if (!cl.load()) {
        say("GPU skipped: OpenCL.dll did not load");
        return;
    }
    cl_uint platforms = 0;
    cl.getPlatformIDs(0, nullptr, &platforms);
    std::vector<cl_platform_id> platform_ids(platforms);
    cl.getPlatformIDs(platforms, platform_ids.data(), nullptr);
    cl_device_id device = nullptr;
    for (cl_platform_id platform : platform_ids) {
        cl_uint devices = 0;
        if (cl.getDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &devices) != CL_SUCCESS ||
            devices == 0) {
            continue;
        }
        cl.getDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        break;
    }
    if (!device) {
        std::printf("GPU skipped: no OpenCL GPU\n");
        return;
    }
    char name[256] = {};
    cl_ulong max_alloc = 0;
    cl_uint cus = 0;
    cl.getDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    cl.getDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, nullptr);
    cl.getDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, nullptr);
    std::printf("GPU %s compute_units=%u max_alloc_mib=%.1f\n", name, cus,
                max_alloc / 1048576.0);
    std::fflush(stdout);

    cl_int status = CL_SUCCESS;
    const cl_context_properties hinted[] = {
        CL_CONTEXT_PERF_HINT_QCOM, CL_PERF_HINT_HIGH_QCOM,
        CL_CONTEXT_PRIORITY_HINT_QCOM, CL_PRIORITY_HINT_HIGH_QCOM, 0};
    cl_context context = cl.createContext(hinted, 1, &device, nullptr, nullptr, &status);
    const char* hint = "qcom-high";
    if (!context) {
        context = cl.createContext(nullptr, 1, &device, nullptr, nullptr, &status);
        hint = "default";
    }
    if (!context) {
        std::printf("GPU context failed: %d\n", status);
        return;
    }
    std::printf("GPU context perf hint: %s\n", hint);
    std::fflush(stdout);
    cl_command_queue queue =
        cl.createCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    const char* source = kGpuSource;
    const size_t source_len = std::strlen(source);
    cl_program program =
        cl.createProgramWithSource(context, 1, &source, &source_len, &status);
    say("gpu: build program");
    status = cl.buildProgram(program, 1, &device, "-cl-std=CL3.0", nullptr, nullptr);
    say("gpu: build returned");
    if (status != CL_SUCCESS) {
        size_t log_size = 0;
        cl.getProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        cl.getProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(),
                               nullptr);
        std::printf("GPU build failed:\n%s\n", log.c_str());
        return;
    }
    cl_kernel read16 = cl.createKernel(program, "read16", &status);
    cl_kernel read64 = cl.createKernel(program, "read64", &status);
    cl_kernel copy4 = cl.createKernel(program, "copy4", &status);
    cl_kernel triad4 = cl.createKernel(program, "triad4", &status);

    const size_t sizes[] = {256ull << 20, 512ull << 20};
    for (size_t bytes : sizes) {
        if (static_cast<cl_ulong>(bytes) > max_alloc) continue;
        cl_mem a = cl.createBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, &status);
        cl_mem b = cl.createBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, &status);
        cl_mem c = cl.createBuffer(context, CL_MEM_READ_WRITE, bytes, nullptr, &status);
        const cl_uint pattern = 0xA5A5A5A5u;
        cl.enqueueFillBuffer(queue, a, &pattern, sizeof(pattern), 0, bytes, 0, nullptr, nullptr);
        cl.enqueueFillBuffer(queue, b, &pattern, sizeof(pattern), 0, bytes, 0, nullptr, nullptr);
        say("gpu: fill buffers");
        cl.finish(queue);
        say("gpu: fill done");
        const size_t partial_bytes = 4096 * sizeof(cl_uint);
        cl_mem partials =
            cl.createBuffer(context, CL_MEM_WRITE_ONLY, partial_bytes, nullptr, &status);

        auto time_kernel = [&](const char* op, cl_kernel kernel, int argc_kind,
                               size_t vector_bytes, size_t traffic) {
            const size_t local = 256;
            const size_t elements = bytes / vector_bytes;
            const size_t global = (elements + local - 1) / local * local;
            const cl_ulong count = static_cast<cl_ulong>(elements);
            const float scale = 1.125f;
            std::vector<double> samples;
            for (int sample = 0; sample < kSamples + 2; ++sample) {
                cl_uint arg = 0;
                if (argc_kind == 0) {
                    cl.setKernelArg(kernel, arg++, sizeof(cl_mem), &a);
                    cl.setKernelArg(kernel, arg++, sizeof(cl_mem), &partials);
                    cl.setKernelArg(kernel, arg++, sizeof(count), &count);
                } else if (argc_kind == 1) {
                    cl.setKernelArg(kernel, arg++, sizeof(cl_mem), &a);
                    cl.setKernelArg(kernel, arg++, sizeof(cl_mem), &c);
                    cl.setKernelArg(kernel, arg++, sizeof(count), &count);
                } else {
                    cl.setKernelArg(kernel, arg++, sizeof(cl_mem), &a);
                    cl.setKernelArg(kernel, arg++, sizeof(cl_mem), &b);
                    cl.setKernelArg(kernel, arg++, sizeof(cl_mem), &c);
                    cl.setKernelArg(kernel, arg++, sizeof(count), &count);
                    cl.setKernelArg(kernel, arg++, sizeof(scale), &scale);
                }
                cl_event event = nullptr;
                status = cl.enqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0,
                                                 nullptr, &event);
                if (status != CL_SUCCESS) {
                    std::printf("GPU enqueue %s failed: %d\n", op, status);
                    return;
                }
                cl.waitForEvents(1, &event);
                const double seconds = profiled_ns(cl, event) / 1.0e9;
                cl.releaseEvent(event);
                if (sample >= 2) samples.push_back(seconds);
            }
            report("gpu", op, "opencl-event", static_cast<int>(cus), bytes, traffic, samples, 0);
        };
        time_kernel("read16", read16, 0, 16, bytes);
        time_kernel("read64", read64, 0, 64, bytes);
        time_kernel("copy", copy4, 1, 16, bytes * 2);
        time_kernel("triad", triad4, 2, 16, bytes * 3);
        cl.releaseMemObject(partials);
        cl.releaseMemObject(a);
        cl.releaseMemObject(b);
        cl.releaseMemObject(c);
    }
    cl.releaseKernel(read16);
    cl.releaseKernel(read64);
    cl.releaseKernel(copy4);
    cl.releaseKernel(triad4);
    cl.releaseProgram(program);
    cl.releaseCommandQueue(queue);
    cl.releaseContext(context);
}

void qnn_log(const char* fmt, QnnLog_Level_t level, uint64_t, va_list args) {
    if (level > QNN_LOG_LEVEL_WARN) return;
    std::fputs("QNN ", stdout);
    std::vprintf(fmt, args);
    std::fputc('\n', stdout);
}

const char* profile_name(uint32_t type) {
    switch (type) {
        case QNN_PROFILE_EVENTTYPE_EXECUTE: return "EXECUTE";
        case QNN_PROFILE_EVENTTYPE_EXECUTE_QUEUE_WAIT: return "QUEUE_WAIT";
        case QNN_PROFILE_EVENTTYPE_EXECUTE_PREPROCESS: return "PREPROCESS";
        case QNN_PROFILE_EVENTTYPE_EXECUTE_DEVICE: return "DEVICE";
        case QNN_PROFILE_EVENTTYPE_EXECUTE_POSTPROCESS: return "POSTPROCESS";
        case QNN_PROFILE_EVENTTYPE_NODE: return "NODE";
        case QNN_PROFILE_EVENTTYPE_INIT: return "INIT";
        case QNN_PROFILE_EVENTTYPE_FINALIZE: return "FINALIZE";
        default: return "OTHER";
    }
}

struct QnnApi {
    HMODULE module = nullptr;
    const QNN_INTERFACE_VER_TYPE* api = nullptr;
    Qnn_LogHandle_t log = nullptr;
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device = nullptr;
    Qnn_ContextHandle_t context = nullptr;
    uint32_t power_id = 0;
    bool power_ready = false;

    bool open() {
        const char* root = R"(C:\Qualcomm\AIStack\QAIRT\2.45.40.260406)";
        const std::string lib = std::string(root) + R"(\lib\aarch64-windows-msvc)";
        const std::string skel = std::string(root) + R"(\lib\hexagon-v81\unsigned)";
        SetEnvironmentVariableA("ADSP_LIBRARY_PATH", skel.c_str());
        SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
        AddDllDirectory(std::wstring(lib.begin(), lib.end()).c_str());
        char path[32768] = {};
        GetEnvironmentVariableA("PATH", path, sizeof(path));
        std::string merged = lib + ";" + skel + ";" + path;
        SetEnvironmentVariableA("PATH", merged.c_str());
        module = LoadLibraryA((lib + R"(\QnnHtp.dll)").c_str());
        if (!module) {
            std::printf("NPU skipped: QnnHtp.dll load error %lu\n", GetLastError());
            return false;
        }
        auto get_providers = reinterpret_cast<Qnn_ErrorHandle_t (*)(const QnnInterface_t***,
                                                                    uint32_t*)>(
            GetProcAddress(module, "QnnInterface_getProviders"));
        if (!get_providers) {
            std::printf("NPU skipped: QnnInterface_getProviders missing\n");
            return false;
        }
        const QnnInterface_t** providers = nullptr;
        uint32_t count = 0;
        if (get_providers(&providers, &count) != QNN_SUCCESS || count == 0) {
            std::printf("NPU skipped: no QNN interface providers\n");
            return false;
        }
        api = &providers[0]->QNN_INTERFACE_VER_NAME;
        std::printf("NPU provider %s api %u.%u\n",
                    providers[0]->providerName ? providers[0]->providerName : "(null)",
                    providers[0]->apiVersion.coreApiVersion.major,
                    providers[0]->apiVersion.coreApiVersion.minor);
        if (api->logCreate) {
            api->logCreate(qnn_log, QNN_LOG_LEVEL_ERROR, &log);
        }
        if (!api->backendCreate ||
            api->backendCreate(log, nullptr, &backend) != QNN_SUCCESS) {
            std::printf("NPU backendCreate failed\n");
            return false;
        }
        if (api->propertyHasCapability) {
            const Qnn_ErrorHandle_t online =
                api->propertyHasCapability(QNN_PROPERTY_GRAPH_SUPPORT_ONLINE_PREPARE);
            std::printf("NPU online-prepare capability status=%llu\n",
                        static_cast<unsigned long long>(online));
        }
        if (api->deviceCreate &&
            api->deviceCreate(log, nullptr, &device) != QNN_SUCCESS) {
            std::printf("NPU deviceCreate failed\n");
            return false;
        }
        vote_performance();
        if (!api->contextCreate ||
            api->contextCreate(backend, device, nullptr, &context) != QNN_SUCCESS) {
            std::printf("NPU contextCreate failed\n");
            return false;
        }
        describe_platform();
        return true;
    }

    void vote_performance() {
        if (!api->deviceGetInfrastructure) return;
        QnnDevice_Infrastructure_t infra = nullptr;
        if (api->deviceGetInfrastructure(&infra) != QNN_SUCCESS || !infra) {
            std::printf("NPU perf infrastructure unavailable\n");
            return;
        }
        auto* htp = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(infra);
        if (!htp->perfInfra.createPowerConfigId || !htp->perfInfra.setPowerConfig) return;
        if (htp->perfInfra.createPowerConfigId(0, 0, &power_id) != QNN_SUCCESS) {
            std::printf("NPU createPowerConfigId failed\n");
            return;
        }
        QnnHtpPerfInfrastructure_PowerConfig_t power =
            QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
        power.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
        power.dcvsV3Config.contextId = power_id;
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
        const QnnHtpPerfInfrastructure_PowerConfig_t* configs[] = {&power, nullptr};
        const Qnn_ErrorHandle_t status = htp->perfInfra.setPowerConfig(power_id, configs);
        power_ready = status == QNN_SUCCESS;
        std::printf("NPU perf vote status=%llu power_id=%u\n",
                    static_cast<unsigned long long>(status), power_id);
    }

    void describe_platform() {
        if (!api->deviceGetPlatformInfo) return;
        const QnnDevice_PlatformInfo_t* info = nullptr;
        if (api->deviceGetPlatformInfo(log, &info) != QNN_SUCCESS || !info) return;
        std::printf("NPU hw devices=%u\n", info->v1.numHwDevices);
        for (uint32_t device_index = 0; device_index < info->v1.numHwDevices; ++device_index) {
            const auto& hw = info->v1.hwDevices[device_index].v1;
            std::printf("  device %u type=%u cores=%u\n", hw.deviceId, hw.deviceType,
                        hw.numCores);
            if (!hw.deviceInfoExtension) continue;
            auto* ext = reinterpret_cast<QnnHtpDevice_DeviceInfoExtension_t*>(
                hw.deviceInfoExtension);
            std::printf("  vtcm_mb=%zu arch=%d soc=%u signed_pd=%d\n",
                        ext->onChipDevice.vtcmSize, static_cast<int>(ext->onChipDevice.arch),
                        ext->onChipDevice.socModel, ext->onChipDevice.signedPdSupport ? 1 : 0);
        }
        if (api->deviceFreePlatformInfo) api->deviceFreePlatformInfo(log, info);
    }
};

Qnn_Tensor_t make_tensor(const char* name, Qnn_TensorType_t type, Qnn_DataType_t dtype,
                         uint32_t* dims, uint32_t rank, void* data, uint64_t bytes,
                         bool quantize) {
    Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
    tensor.v1.name = name;
    tensor.v1.type = type;
    tensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    tensor.v1.dataType = dtype;
    tensor.v1.rank = rank;
    tensor.v1.dimensions = dims;
    tensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
    tensor.v1.clientBuf.data = nullptr;
    tensor.v1.clientBuf.dataSize = 0;
    (void)data;
    (void)bytes;
    if (quantize) {
        tensor.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
        tensor.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
        tensor.v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
        tensor.v1.quantizeParams.scaleOffsetEncoding.offset = 0;
    }
    return tensor;
}

void dump_profile(const QNN_INTERFACE_VER_TYPE* api, Qnn_ProfileHandle_t profile, int depth,
                  QnnProfile_EventId_t id) {
    QnnProfile_EventData_t data = QNN_PROFILE_EVENT_DATA_INIT;
    Qnn_ErrorHandle_t status = api->profileGetEventData(id, &data);
    if (status == QNN_SUCCESS) {
        std::printf("%*s%s id=%s unit=%u value=%llu\n", depth * 2, "", profile_name(data.type),
                    data.identifier ? data.identifier : "", data.unit,
                    static_cast<unsigned long long>(data.value));
    } else if (api->profileGetExtendedEventData) {
        QnnProfile_ExtendedEventData_t extended = QNN_PROFILE_EXTENDED_EVENT_DATA_INIT;
        if (api->profileGetExtendedEventData(id, &extended) == QNN_SUCCESS) {
            std::printf("%*s%s id=%s unit=%u value=%llu\n", depth * 2, "",
                        profile_name(extended.v1.type),
                        extended.v1.identifier ? extended.v1.identifier : "", extended.v1.unit,
                        static_cast<unsigned long long>(extended.v1.value.uint64Value));
        }
    }
    const QnnProfile_EventId_t* children = nullptr;
    uint32_t count = 0;
    if (api->profileGetSubEvents &&
        api->profileGetSubEvents(id, &children, &count) == QNN_SUCCESS) {
        for (uint32_t child = 0; child < count; ++child) {
            dump_profile(api, profile, depth + 1, children[child]);
        }
    }
}

bool bench_one_npu(QnnApi& qnn, const char* op_name, Qnn_DataType_t dtype, size_t bytes,
                   int inputs, size_t element_bytes) {
    if (bytes % (element_bytes * 128) != 0) return false;
    const uint64_t elements = bytes / element_bytes;
    uint32_t dims[4] = {1, 1, 1, 128};
    const uint64_t spatial = elements / 128;
    dims[1] = 1;
    dims[2] = static_cast<uint32_t>(spatial);
    if (spatial > 4096) {
        dims[1] = static_cast<uint32_t>(spatial / 1024);
        dims[2] = 1024;
    }
    std::printf("NPU prepare %s dtype=0x%x bytes=%zu shape=%u,%u,%u,%u\n", op_name, dtype, bytes,
                dims[0], dims[1], dims[2], dims[3]);
    Buffer a = allocate(bytes);
    Buffer b = allocate(bytes);
    Buffer c = allocate(bytes);
    if (!a || !b || !c) {
        std::printf("NPU alloc failed\n");
        return false;
    }
    fill_bytes(a, 0x3C);
    fill_bytes(b, 0x3C);
    fill_bytes(c, 0x00);
    const bool quantize = dtype == QNN_DATATYPE_UFIXED_POINT_8;
    uint32_t dims_a[4] = {dims[0], dims[1], dims[2], dims[3]};
    uint32_t dims_b[4] = {dims[0], dims[1], dims[2], dims[3]};
    uint32_t dims_c[4] = {dims[0], dims[1], dims[2], dims[3]};
    Qnn_Tensor_t input_a = make_tensor("npu_a", QNN_TENSOR_TYPE_APP_WRITE, dtype, dims_a, 4,
                                       a.data, bytes, quantize);
    Qnn_Tensor_t input_b = make_tensor("npu_b", QNN_TENSOR_TYPE_APP_WRITE, dtype, dims_b, 4,
                                       b.data, bytes, quantize);
    Qnn_Tensor_t output = make_tensor("npu_c", QNN_TENSOR_TYPE_APP_READ, dtype, dims_c, 4, c.data,
                                      bytes, quantize);
    Qnn_Tensor_t op_inputs[2] = {input_a, input_b};
    Qnn_GraphHandle_t graph = nullptr;
    const std::string graph_name =
        std::string(op_name) + "_" + std::to_string(bytes) + "_" + std::to_string(dtype);
    if (qnn.api->graphCreate(qnn.context, graph_name.c_str(), nullptr, &graph) != QNN_SUCCESS) {
        std::printf("NPU graphCreate failed\n");
        return false;
    }
    if (!qnn.api->tensorCreateGraphTensor) {
        std::printf("NPU tensorCreateGraphTensor missing\n");
        return false;
    }
    const Qnn_ErrorHandle_t tensor_a =
        qnn.api->tensorCreateGraphTensor(graph, &input_a);
    const Qnn_ErrorHandle_t tensor_b =
        inputs > 1 ? qnn.api->tensorCreateGraphTensor(graph, &input_b) : QNN_SUCCESS;
    const Qnn_ErrorHandle_t tensor_c =
        qnn.api->tensorCreateGraphTensor(graph, &output);
    std::printf("NPU tensor register a=%llu b=%llu c=%llu ids=%u,%u,%u\n",
                static_cast<unsigned long long>(tensor_a),
                static_cast<unsigned long long>(tensor_b),
                static_cast<unsigned long long>(tensor_c), input_a.v1.id, input_b.v1.id,
                output.v1.id);
    if (tensor_a != QNN_SUCCESS || tensor_b != QNN_SUCCESS || tensor_c != QNN_SUCCESS) {
        return false;
    }
    input_a.v1.clientBuf = {a.data, static_cast<uint32_t>(bytes)};
    input_b.v1.clientBuf = {b.data, static_cast<uint32_t>(bytes)};
    output.v1.clientBuf = {c.data, static_cast<uint32_t>(bytes)};
    op_inputs[0] = input_a;
    op_inputs[1] = input_b;
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.version = QNN_OPCONFIG_VERSION_1;
    op.v1.name = graph_name.c_str();
    op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    op.v1.typeName = op_name;
    op.v1.numOfParams = 0;
    op.v1.numOfInputs = static_cast<uint32_t>(inputs);
    op.v1.inputTensors = op_inputs;
    op.v1.numOfOutputs = 1;
    op.v1.outputTensors = &output;
    const Qnn_ErrorHandle_t add_status = qnn.api->graphAddNode(graph, op);
    if (add_status != QNN_SUCCESS) {
        std::printf("NPU graphAddNode %s failed status=%llu\n", op_name,
                    static_cast<unsigned long long>(add_status));
        return false;
    }
    const double finalize_started = qpc_seconds();
    const Qnn_ErrorHandle_t finalize_status =
        qnn.api->graphFinalize(graph, nullptr, nullptr);
    std::printf("NPU finalize status=%llu seconds=%.2f\n",
                static_cast<unsigned long long>(finalize_status),
                qpc_seconds() - finalize_started);
    if (finalize_status != QNN_SUCCESS) return false;

    Qnn_Tensor_t exec_inputs[2] = {input_a, input_b};
    std::vector<double> samples;
    uint64_t checksum = 0;
    for (int sample = 0; sample < kSamples + 1; ++sample) {
        const double started = qpc_seconds();
        const Qnn_ErrorHandle_t exec = qnn.api->graphExecute(
            graph, exec_inputs, static_cast<uint32_t>(inputs), &output, 1, nullptr, nullptr);
        const double elapsed = qpc_seconds() - started;
        if (exec != QNN_SUCCESS) {
            std::printf("NPU execute failed status=%llu\n",
                        static_cast<unsigned long long>(exec));
            return false;
        }
        if (sample > 0) samples.push_back(elapsed);
    }
    const uint16_t* words = reinterpret_cast<uint16_t*>(c.data);
    const size_t words_n = std::min<size_t>(4096, bytes / 2);
    for (size_t i = 0; i < words_n; ++i) checksum += words[i];
    const size_t traffic = bytes * static_cast<size_t>(inputs + 1);
    report("npu", op_name, qnn.power_ready ? "htp-wall-perf" : "htp-wall-default", 1, bytes,
           traffic, samples, checksum);

    if (qnn.api->profileCreate && qnn.api->profileGetEvents) {
        Qnn_ProfileHandle_t profile = nullptr;
        if (qnn.api->profileCreate(qnn.backend, QNN_PROFILE_LEVEL_DETAILED, &profile) ==
            QNN_SUCCESS) {
            const Qnn_ErrorHandle_t exec = qnn.api->graphExecute(
                graph, exec_inputs, static_cast<uint32_t>(inputs), &output, 1, profile, nullptr);
            std::printf("NPU profiled execute status=%llu\n",
                        static_cast<unsigned long long>(exec));
            const QnnProfile_EventId_t* events = nullptr;
            uint32_t count = 0;
            if (exec == QNN_SUCCESS &&
                qnn.api->profileGetEvents(profile, &events, &count) == QNN_SUCCESS) {
                for (uint32_t event = 0; event < count; ++event) {
                    dump_profile(qnn.api, profile, 0, events[event]);
                }
            }
            if (qnn.api->profileFree) qnn.api->profileFree(profile);
        }
    }
    return true;
}

void bench_npu() {
    QnnApi qnn;
    if (!qnn.open()) return;
    const size_t sizes[] = {16ull << 20, 64ull << 20, 256ull << 20};
    for (size_t bytes : sizes) {
        if (!bench_one_npu(qnn, QNN_OP_RELU, QNN_DATATYPE_FLOAT_16, bytes, 1, 2)) {
            std::printf("NPU fp16 relu failed at %zu; trying fp16 add\n", bytes);
            if (!bench_one_npu(qnn, QNN_OP_ELEMENT_WISE_ADD, QNN_DATATYPE_FLOAT_16, bytes, 2,
                               2) &&
                !bench_one_npu(qnn, QNN_OP_RELU, QNN_DATATYPE_UFIXED_POINT_8, bytes, 1, 1)) {
                std::printf("NPU stopped at %zu bytes\n", bytes);
                break;
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    g_log = std::fopen(
        R"(C:\Users\hotschmoe\Documents\llm_inference\native_nvfp4\island_bw\island_bw.log)",
        "w");
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    g_cpu_llc_only = argc >= 2 && std::strcmp(argv[1], "cpu-llc") == 0;
    const bool cpu = argc < 2 || g_cpu_llc_only || std::strcmp(argv[1], "cpu") == 0 ||
                     std::strcmp(argv[1], "all") == 0;
    const bool gpu = argc < 2 || std::strcmp(argv[1], "gpu") == 0 || std::strcmp(argv[1], "all") == 0;
    const bool npu = argc < 2 || std::strcmp(argv[1], "npu") == 0 || std::strcmp(argv[1], "all") == 0;
    std::printf("island bandwidth probe  nominal_gbs=%.1f\n", kNominalGbs);
    std::printf("decimal GB/s = moved bytes / elapsed seconds / 1e9\n");
    if (cpu) {
        say("--- cpu ---");
        bench_cpu();
    }
    if (gpu) {
        say("--- gpu ---");
        bench_gpu();
    }
    if (npu) {
        say("--- npu ---");
        bench_npu();
    }
    say("done");
    if (g_log) std::fclose(g_log);
    return 0;
}
