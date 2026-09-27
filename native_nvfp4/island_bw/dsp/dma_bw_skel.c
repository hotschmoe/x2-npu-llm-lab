// On-DSP DMA memcpy. Timing is the Hexagon qtimer inside this call, so the
// number does not include FastRPC. The copy path is the same 1D descriptor
// llama.cpp pushes: DDR -> VTCM -> DDR, with cache bypass on both sides.

#include <stdint.h>
#include <string.h>

#include "AEEStdDef.h"
#include "AEEStdErr.h"
#include "remote.h"
#include "HAP_compute_res.h"
#include "HAP_power.h"
#include "HAP_perf.h"

typedef struct {
    void * next;
    uint32_t size : 24;
    uint32_t desc_size : 2;
    uint32_t dst_comp : 1;
    uint32_t src_comp : 1;
    uint32_t dst_bypass : 1;
    uint32_t src_bypass : 1;
    uint32_t order : 1;
    uint32_t done : 1;
    void * src;
    void * dst;
} dma_desc_1d;

static void dmstart(void * next) {
    asm volatile(" release(%0):at" : : "r"(next));
    asm volatile(" dmstart(%0)" : : "r"(next));
}

static unsigned dmwait_poll(volatile dma_desc_1d * desc, uint64_t deadline) {
    while (!desc->done) {
        unsigned ret = 0;
        asm volatile(" %0 = dmpoll" : "=r"(ret) : : "memory");
        (void) ret;
        unsigned long long now = 0;
        asm volatile(" %0 = c31:30 " : "=r"(now));
        if (now > deadline) return 1;
    }
    return 0;
}

static int dma_copy_block(void * dst, void * src, uint32_t size, void * desc_mem) {
    volatile dma_desc_1d * desc = (volatile dma_desc_1d *) desc_mem;
    memset((void *) desc, 0, sizeof(*desc));
    desc->src = src;
    desc->dst = dst;
    desc->size = size;
    desc->desc_size = 0;
    desc->src_bypass = 1;
    desc->dst_bypass = 1;
    unsigned long long start = 0;
    asm volatile(" %0 = c31:30 " : "=r"(start));
    dmstart((void *) desc);
    return dmwait_poll(desc, start + 192ull * 1000000ull);
}

enum {
    MODE_DMA_COPY = 0,
    MODE_DMA_READ = 1,
    MODE_HVX_COPY = 2,
};

static void hvx_copy(uint8_t * dst, const uint8_t * src, uint32_t n) {
    uint32_t i = 0;
    for (; i + 128 <= n; i += 128) {
        asm volatile(
            " v0 = vmem(%0)\n"
            " vmem(%1) = v0\n"
            :
            : "r"(src + i), "r"(dst + i)
            : "v0", "memory");
    }
}

static uint32_t checksum(const uint8_t * p, uint32_t n) {
    uint32_t sum = 0;
    uint32_t limit = n < 4096 ? n : 4096;
    for (uint32_t i = 0; i < limit; i += 64) sum += p[i];
    return sum;
}

static int vote_perf(void) {
    HAP_power_request_t request;
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_DCVS_v3;
    request.dcvs_v3.set_dcvs_enable = TRUE;
    request.dcvs_v3.dcvs_enable = FALSE;
    request.dcvs_v3.set_latency = TRUE;
    request.dcvs_v3.latency = 40;
    request.dcvs_v3.set_core_params = TRUE;
    request.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.set_bus_params = TRUE;
    request.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
    return HAP_power_set((void *) 0xD5A, &request);
}

static int run(const uint8_t * src, uint32_t src_len, uint8_t * dst, uint32_t dst_len,
               const uint32_t * params, uint32_t * results) {
    if (!src || !dst || src_len != dst_len || (src_len & 127)) return AEE_EBADPARM;
    uint32_t passes = params ? params[0] : 1;
    uint32_t mode = params ? params[1] : 0;
    if (passes == 0 || passes > 32) passes = 1;

    int power = vote_perf();
    unsigned int vtcm_size = 0;
    HAP_compute_res_query_VTCM(0, &vtcm_size, 0, 0, 0);
    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    unsigned int request = vtcm_size > (4u << 20) ? (4u << 20) : vtcm_size;
    if (request < (256u << 10)) return AEE_ENOMEMORY;
    if (HAP_compute_res_attr_set_vtcm_param(&attr, request, 1) != 0) return AEE_ENOMEMORY;
    unsigned int ctx = HAP_compute_res_acquire(&attr, 200000);
    void * vtcm = HAP_compute_res_attr_get_vtcm_ptr(&attr);
    if (!ctx || !vtcm) return AEE_ENOMEMORY;

    uint8_t * vtcm_bytes = (uint8_t *) vtcm;
    dma_desc_1d * desc = (dma_desc_1d *) ((uintptr_t) (vtcm_bytes + request - 256) & ~127u);
    uint32_t chunk = 2u << 20;
    if (chunk + 256 > request) chunk = request / 2 & ~127u;

    uint64_t t0 = HAP_perf_get_time_us();
    int timed_out = 0;
    for (uint32_t pass = 0; pass < passes && !timed_out; ++pass) {
        for (uint32_t off = 0; off < src_len && !timed_out; off += chunk) {
            uint32_t n = src_len - off;
            if (n > chunk) n = chunk;
            if (mode == MODE_HVX_COPY) {
                hvx_copy(dst + off, src + off, n);
            } else if (mode == MODE_DMA_READ) {
                timed_out = dma_copy_block(vtcm_bytes, (void *) (src + off), n, desc);
            } else {
                timed_out = dma_copy_block(vtcm_bytes, (void *) (src + off), n, desc);
                if (!timed_out) timed_out = dma_copy_block(dst + off, vtcm_bytes, n, desc);
            }
        }
    }
    uint64_t usec = HAP_perf_get_time_us() - t0;
    if (results) {
        results[0] = (uint32_t) usec;
        results[1] = mode == MODE_DMA_READ ? checksum(src, src_len) : checksum(dst, dst_len);
        results[2] = request;
        results[3] = timed_out ? 2u : (uint32_t) (power == 0 ? 0 : 1);
    }
    HAP_compute_res_release(ctx);
    return timed_out ? AEE_EEXPIRED : AEE_SUCCESS;
}

__attribute__((visibility("default")))
uint32_t dma_bw_skel_handle_invoke_qaic_version = 10051;

__attribute__((visibility("default")))
char dma_bw_skel_handle_invoke_uri[] =
    "file:///libdma2_skel.so?dma_bw_skel_handle_invoke&_modver=1.0";

__attribute__((visibility("default")))
int dma_bw_skel_handle_invoke(remote_handle64 handle, uint32_t sc, remote_arg * args) {
    (void) handle;
    if (REMOTE_SCALARS_METHOD(sc) != 1) return 0x1000 + (int) REMOTE_SCALARS_METHOD(sc);
    if (REMOTE_SCALARS_INBUFS(sc) < 2 || REMOTE_SCALARS_OUTBUFS(sc) < 2) {
        return 0x2000 + ((int) REMOTE_SCALARS_INBUFS(sc) << 8) + (int) REMOTE_SCALARS_OUTBUFS(sc);
    }
    if (!args || !args[0].buf.pv || !args[2].buf.pv || !args[1].buf.pv || !args[3].buf.pv) return 0x3000;
    if (args[0].buf.nLen != args[2].buf.nLen) return 0x4000;
    if ((args[0].buf.nLen & 127) != 0) return 0x5000;
    const uint8_t * src = (const uint8_t *) args[0].buf.pv;
    uint32_t src_len = (uint32_t) args[0].buf.nLen;
    const uint32_t * params = (const uint32_t *) args[1].buf.pv;
    uint8_t * dst = (uint8_t *) args[2].buf.pv;
    uint32_t dst_len = (uint32_t) args[2].buf.nLen;
    uint32_t * results = (uint32_t *) args[3].buf.pv;
    return run(src, src_len, dst, dst_len, params, results);
}

__attribute__((visibility("default")))
int dma_bw_open(const char * uri, remote_handle64 * handle) {
    (void) uri;
    if (handle) *handle = 0xD5A;
    return AEE_SUCCESS;
}

__attribute__((visibility("default")))
int dma_bw_close(remote_handle64 handle) {
    (void) handle;
    return AEE_SUCCESS;
}
