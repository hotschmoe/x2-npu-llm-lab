/* Stand-in for ggml-opencl.dll. The Adreno driver fast-fails in CB.dll
   (0xC0000409) while probing devices, which takes the whole process down
   before Hexagon can open a session. This registry reports zero devices. */

typedef struct ggml_backend_reg * ggml_backend_reg_t;

struct ggml_backend_reg_i {
    const char * (*get_name)(ggml_backend_reg_t reg);
    unsigned long long (*get_device_count)(ggml_backend_reg_t reg);
    void * (*get_device)(ggml_backend_reg_t reg, unsigned long long index);
    void * (*get_proc_address)(ggml_backend_reg_t reg, const char * name);
};

struct ggml_backend_reg {
    int api_version;
    struct ggml_backend_reg_i iface;
    void * context;
};

static const char * stub_name(ggml_backend_reg_t reg) {
    (void) reg;
    return "OpenCL";
}

static unsigned long long stub_count(ggml_backend_reg_t reg) {
    (void) reg;
    return 0;
}

static struct ggml_backend_reg stub_reg = {
    2,
    { stub_name, stub_count, 0, 0 },
    0,
};

__declspec(dllexport) ggml_backend_reg_t ggml_backend_opencl_reg(void) {
    return &stub_reg;
}

__declspec(dllexport) void * ggml_backend_opencl_init(void) {
    return 0;
}

__declspec(dllexport) int ggml_backend_is_opencl(void * backend) {
    (void) backend;
    return 0;
}
