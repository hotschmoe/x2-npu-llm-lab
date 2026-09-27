#include <stdint.h>
#include "remote.h"
#include "AEEStdErr.h"

struct Parameter;
struct Method {
    uint32_t uScalars;
    int32_t primInSize;
    int32_t primROutSize;
    int maxArgs;
    int numParams;
    const struct Parameter * const * params;
    uint8_t primInAlignment;
    uint8_t primROutAlignment;
};

struct Interface {
    int nMethods;
    const struct Method * const * methodArray;
    int nIIds;
    const uint32_t * iids;
    const uint16_t * methodStringArray;
    const uint16_t * methodStrings;
    const char * strings;
};

static const struct Method dma_method = {
    REMOTE_SCALARS_MAKEX(0, 0, 2, 2, 0, 0), 0, 0, 4, 0, 0, 1, 1
};
static const struct Method * const dma_methods[] = { &dma_method, &dma_method };

__attribute__((visibility("default")))
const struct Interface dma_bw_slim = { 2, dma_methods, 0, 0, 0, 0, 0 };

__attribute__((visibility("default")))
const struct Interface dma_bw_skel_slim = { 2, dma_methods, 0, 0, 0, 0, 0 };

__attribute__((visibility("default")))
uint32_t dma_bw_skel_handle_invoke_qaic_version = 10051;

__attribute__((visibility("default")))
char dma_bw_skel_handle_invoke_uri[] =
    "file:///libdma2_skel.so?dma_bw_skel_handle_invoke&_modver=1.0";

__attribute__((visibility("default")))
int dma_bw_open(const char * uri, remote_handle64 * handle) {
    (void) uri;
    if (handle) *handle = 1;
    return 0;
}

__attribute__((visibility("default")))
int dma_bw_close(remote_handle64 handle) {
    (void) handle;
    return 0;
}

__attribute__((visibility("default")))
int dma_bw_skel_handle_invoke(remote_handle64 handle, uint32_t sc, remote_arg * args) {
    (void) handle;
    (void) args;
    return 0x4200 + (int) (sc & 0xff);
}
