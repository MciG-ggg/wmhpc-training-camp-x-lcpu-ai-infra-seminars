#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cstdint>
#include <cuda_fp8.h>  // 间接拉入 <cuda_runtime.h>

static __device__ int a_row_of(int lane, int i) {
    int gid = lane >> 2;
    int rid = i / 4;
    return gid + 8 * (rid & 1);
}
static __device__ int a_col_of(int lane, int i) {
    int tig = lane & 3;
    int j   = i & 3;
    int rid = i / 4;
    return 4 * tig + j + 16 * (rid >> 1);
}
static __device__ int b_row_of(int lane, int i) {  // k
    int tig = lane & 3;
    int j   = i & 3;
    int rid = i / 4;
    return 4 * tig + j + 16 * rid;
}
static __device__ int b_col_of(int lane, int i) {  // n
    return lane >> 2;
}

__global__ void mma_fp8(
    const __nv_fp8_e4m3* A,
    const __nv_fp8_e4m3* B,
    float* D
) {
    int lane = threadIdx.x;
    unsigned a[4]; // 4 个 b32, A 的 16 个 fp8 → 4×4 = 16 字节
    unsigned b[2]; // 2 个 b32, B 的  8 个 fp8 → 2×4 =  8 字节

    // 装载 A: 每个 lane 拿 16 个 fp8
    // A 在显存里是 16×32 (row-major, stride=32), MMA 用整 32 列 (K=32)
    uint8_t a_buf[16];
    for (int i = 0; i < 16; i++) {
        int r = a_row_of(lane, i);
        int c = a_col_of(lane, i);
        // 必须 reinterpret_cast 拿 raw byte; static_cast<uint8_t> 在 nvcc 13
        // 下会走 fp8→fp16→fp32→... 的转换链, 不保留 raw bits
        a_buf[i] = *reinterpret_cast<const uint8_t*>(&A[r * 32 + c]);
    }
    for (int rid = 0; rid < 4; rid++) {
        unsigned packed = 0;
        for (int j = 0; j < 4; j++)
            packed |= (unsigned)a_buf[rid*4 + j] << (8 * j);
        a[rid] = packed;
    }

    // 装载 B: 每个 lane 拿 8 个 fp8
    // B 在显存里是 32×8 (row-major, stride=8)
    uint8_t b_buf[8];
    for (int i = 0; i < 8; i++) {
        int k = b_row_of(lane, i);
        int n = b_col_of(lane, i);
        b_buf[i] = *reinterpret_cast<const uint8_t*>(&B[k * 8 + n]);
    }
    for (int rid = 0; rid < 2; rid++) {
        unsigned packed = 0;
        for (int j = 0; j < 4; j++)
            packed |= (unsigned)b_buf[rid*4 + j] << (8 * j);
        b[rid] = packed;
    }

    // 累加器: 4 个 fp32 (16×8 = 128 / 32 lane = 4)
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float d[4];

    // mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
    //   D[m,n] = sum_{k=0..31} A[m,k] * B[k,n] + C[m,n]
    //   A: 16×32 fp8, B: 32×8 fp8, D: 16×8 fp32
    //   per lane: A=4×b32, B=2×b32, D=4×fp32
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),
           "r"(b[0]),  "r"(b[1]),
           "f"(c[0]),  "f"(c[1]),  "f"(c[2]),  "f"(c[3])
    );

    // 存 D (16×8 fp32, row-major)
    int gid  = lane >> 2;
    int tig  = lane & 3;
    int col0 = tig * 2;
    D[gid       * 8 + col0]       = d[0];
    D[gid       * 8 + col0 + 1]   = d[1];
    D[(gid + 8) * 8 + col0]       = d[2];
    D[(gid + 8) * 8 + col0 + 1]   = d[3];
}

// CPU 参考: D = A @ B (A: 16×32, B: 32×8, D: 16×8)
static void cpu_ref(
    const __nv_fp8_e4m3* A,
    const __nv_fp8_e4m3* B,
    float* D
) {
    for (int m = 0; m < 16; m++) {
        for (int n = 0; n < 8; n++) {
            float acc = 0.0f;
            for (int k = 0; k < 32; k++) {
                acc += float(A[m * 32 + k]) * float(B[k * 8 + n]);
            }
            D[m * 8 + n] = acc;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <seed>\n", argv[0]);
        return 1;
    }
    int seed = atoi(argv[1]);
    srand(seed);

    __nv_fp8_e4m3* hA    = (__nv_fp8_e4m3*)malloc(16 * 32);
    __nv_fp8_e4m3* hB    = (__nv_fp8_e4m3*)malloc(32 * 8);
    float*       hD     = (float*)malloc(16 * 8 * sizeof(float));
    float*       hRef   = (float*)malloc(16 * 8 * sizeof(float));

    // 小整数 (-2..2), e4m3 全可精确表示, 不引入 round-to-nearest 误差
    for (int i = 0; i < 16 * 32; i++)
        hA[i] = __nv_fp8_e4m3(static_cast<float>(rand() % 5 - 2));
    for (int i = 0; i < 32 * 8; i++)
        hB[i] = __nv_fp8_e4m3(static_cast<float>(rand() % 5 - 2));

    cpu_ref(hA, hB, hRef);

    __nv_fp8_e4m3 *dA, *dB;
    float* dD;
    cudaMalloc(&dA, 16 * 32);
    cudaMalloc(&dB, 32 * 8);
    cudaMalloc(&dD, 16 * 8 * sizeof(float));
    cudaMemcpy(dA, hA, 16 * 32, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 32 * 8, cudaMemcpyHostToDevice);

    mma_fp8<<<1, 32>>>(dA, dB, dD);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaMemcpy(hD, dD, 16 * 8 * sizeof(float), cudaMemcpyDeviceToHost);

    // 严格相等 (bit-exact) 比对
    int   errors  = 0;
    for (int i = 0; i < 16 * 8; i++) {
        uint32_t gpu_bits, cpu_bits;
        memcpy(&gpu_bits, &hD[i],   4);
        memcpy(&cpu_bits, &hRef[i], 4);
        if (gpu_bits != cpu_bits) {
            errors++;
            if (errors <= 8)
                printf("MISMATCH @ %d (m=%d,n=%d): gpu=%.6f cpu=%.6f\n",
                       i, i/8, i%8, hD[i], hRef[i]);
        }
    }
    if (errors == 0) {
        printf("PASS seed=%d\n", seed);
    } else {
        printf("FAIL seed=%d errors=%d / 128\n", seed, errors);
    }

    free(hA); free(hB); free(hD); free(hRef);
    cudaFree(dA); cudaFree(dB); cudaFree(dD);
    return errors ? 1 : 0;
}