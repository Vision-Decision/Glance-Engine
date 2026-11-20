// Standalone cuBLASLt heuristic sweep over the six Qwen BF16 projection shapes.
// Build with nvcc -O2 -arch=sm_120 scan/lm_blaslt.cu -lcublasLt -o /tmp/lm_blaslt
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                                                          \
    do {                                                                                                     \
        auto status = (call);                                                                                \
        if (status != 0) {                                                                                   \
            std::fprintf(stderr, "%s failed (%d)\n", #call, int(status));                                    \
            std::exit(1);                                                                                    \
        }                                                                                                    \
    } while (0)

__global__ void initialize(__nv_bfloat16 *data, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        unsigned u = (unsigned(i) ^ 2947351u) * 747796405u;
        data[i] = __float2bfloat16(float((u >> 20) & 2047) / 1024.f - 1.f);
    }
}

void measure(cublasLtHandle_t handle, int m, int k, int n) {
    __nv_bfloat16 *x, *w, *y;
    void *scratch;
    size_t bytes = 32ul << 20;
    CHECK(cudaMalloc(&x, size_t(m) * k * 2));
    CHECK(cudaMalloc(&w, size_t(n) * k * 2));
    CHECK(cudaMalloc(&y, size_t(m) * n * 2));
    CHECK(cudaMalloc(&scratch, bytes));
    initialize<<<(m * k + 255) / 256, 256>>>(x, m * k);
    initialize<<<(n * k + 255) / 256, 256>>>(w, n * k);
    cublasLtMatrixLayout_t a, b, c;
    CHECK(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, k, n, k));
    CHECK(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, k, m, k));
    CHECK(cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, n, m, n));
    cublasLtMatmulDesc_t op;
    CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t transpose = CUBLAS_OP_T;
    CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));
    cublasLtMatmulPreference_t pref;
    CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &bytes,
                                               sizeof(bytes)));
    std::array<cublasLtMatmulHeuristicResult_t, 16> candidates;
    int count = 0;
    CHECK(cublasLtMatmulAlgoGetHeuristic(handle, op, a, b, c, c, pref, candidates.size(), candidates.data(),
                                         &count));
    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    for (int i = 0; i < count; ++i) {
        const auto &r = candidates[i];
        if (r.state != CUBLAS_STATUS_SUCCESS)
            continue;
        float one = 1.f, zero = 0.f;
        auto run = [&]() {
            return cublasLtMatmul(handle, op, &one, w, a, x, b, &zero, y, c, y, c, &r.algo, scratch, bytes,
                                  0);
        };
        if (run() != CUBLAS_STATUS_SUCCESS)
            continue;
        CHECK(run());
        std::array<float, 7> samples;
        for (auto &ms : samples) {
            CHECK(cudaEventRecord(start));
            CHECK(run());
            CHECK(cudaEventRecord(end));
            CHECK(cudaEventSynchronize(end));
            CHECK(cudaEventElapsedTime(&ms, start, end));
        }
        std::sort(samples.begin(), samples.end());
        int id, tile, stages;
        size_t written;
        CHECK(cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof(id),
                                                   &written));
        CHECK(cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile),
                                                   &written));
        CHECK(cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages,
                                                   sizeof(stages), &written));
        std::printf("%d,%d,%d,%d,%d,%d,%d,%zu,%g,%.6f\n", m, k, n, i, id, tile, stages, r.workspaceSize,
                    r.wavesCount, samples[3]);
    }
    std::fflush(stdout);
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));
    CHECK(cublasLtMatmulPreferenceDestroy(pref));
    CHECK(cublasLtMatmulDescDestroy(op));
    CHECK(cublasLtMatrixLayoutDestroy(a));
    CHECK(cublasLtMatrixLayoutDestroy(b));
    CHECK(cublasLtMatrixLayoutDestroy(c));
    CHECK(cudaFree(scratch));
    CHECK(cudaFree(x));
    CHECK(cudaFree(w));
    CHECK(cudaFree(y));
}

int main() {
    cublasLtHandle_t h;
    CHECK(cublasLtCreate(&h));
    std::printf("M,K,N,rank,algorithm,tile,stages,workspace,waves,median_ms\n");
    // GDN out and attention out have identical dimensions.
    for (int m : {4672, 9344, 18688, 37376}) {
        measure(h, m, 2560, 12352);
        measure(h, m, 4096, 2560);
        measure(h, m, 2560, 10240);
        measure(h, m, 2560, 18432);
        measure(h, m, 9216, 2560);
    }
    CHECK(cublasLtDestroy(h));
}
