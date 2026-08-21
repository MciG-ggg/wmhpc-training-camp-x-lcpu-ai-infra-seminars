// 问题 1.4:把 1.3 的手工装载换成 ldmatrix,两条路径共存、都要 PASS。
//
// 把你 1.3 的 kernel 拆成两个装载函数移植进来:
//   load_manual:1.3 的逐 byte 手工装载(公式来自 1.1)
//   load_ldsm:  用 ldmatrix 装载。A 是 fp8——ldmatrix 的元素是 16 个
//               原始 bit,不关心类型;1.1 附加问的打包方向在这里起作用。
//               变体(.x1/.x2/.x4、是否 .trans)自己从 PTX 文档选。
// 数据已在 smem(main 里先从 global 拷入),两条路径都从 smem 装载。
//
// 都 PASS 之后:make ptx/m1_sm80/04_ldmatrix 或 nvdisasm 反汇编,
// 数两条路径 smem->fragment 段的指令构成(装载条数、地址算术条数),
// 报告里回答:ldmatrix 消掉的是哪部分工作?为什么手工路径绕不开它?
//
// 运行:make run/m1_sm80/04_ldmatrix(内部两条路径各跑多 seed)
#include <cuda_fp8.h>
#include <cstdlib>
#include <random>
#include "../common.h"

// smem 布局:sA 按 [16][32] 行主序;B 备了两种布局——sBk 按 [32][8]
// (k-major,1.3 用的就是它),sBn 按 [8][32](n-major,每个 n 的 32
// 个 k 字节连续)。手工路径用哪种都行;ldmatrix 的每个"行地址"要求
// 16 byte 连续,B 的 fragment 需要 k 方向相邻的字节成对进 b16——
// 想清楚哪种布局能满足它。
//
// TODO: 实现两个装载函数。

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

__device__ void load_manual(const uint8_t* sA, const uint8_t* sBk,
                            const uint8_t* sBn, unsigned (&a)[4],
                            unsigned (&b)[2]) {
    (void)sA; (void)sBk; (void)sBn; (void)a; (void)b;
    int lane = threadIdx.x;

    // 装载 A: 每个 lane 拿 16 个 fp8
    // A 在显存里是 16×32 (row-major, stride=32), MMA 用整 32 列 (K=32)
    uint8_t a_buf[16];
    for (int i = 0; i < 16; i++) {
        int r = a_row_of(lane, i);
        int c = a_col_of(lane, i);
        // 必须 reinterpret_cast 拿 raw byte; static_cast<uint8_t> 在 nvcc 13
        // 下会走 fp8→fp16→fp32→... 的转换链, 不保留 raw bits
        a_buf[i] = *reinterpret_cast<const uint8_t*>(&sA[r * 32 + c]);
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
        b_buf[i] = *reinterpret_cast<const uint8_t*>(&sBk[k * 8 + n]);
    }
    for (int rid = 0; rid < 2; rid++) {
        unsigned packed = 0;
        for (int j = 0; j < 4; j++)
            packed |= (unsigned)b_buf[rid*4 + j] << (8 * j);
        b[rid] = packed;
    }
}

__device__ void load_ldsm(const uint8_t* sA, const uint8_t* sBk,
                          const uint8_t* sBn, unsigned (&a)[4],
                          unsigned (&b)[2]) {
    (void)sBk;                          // 这条路径读 sA + sBn
    int lane = threadIdx.x & 31;

    uint32_t a_smem = (uint32_t)__cvta_generic_to_shared(sA);
    uint32_t b_smem = (uint32_t)__cvta_generic_to_shared(sBn);

    // A: ldmatrix.x4.b16, no .trans
    //   A 是 16×32 fp8 行主序 (sA, 每行 32 byte).
    //   4 个子矩阵按 2×2 排布:
    //     sub k 行 r → A[ (k/2)*8 + r ][ (k%2)*16 .. (k%2)*16 + 15 ]
    //   每个 lane 提供 4 个地址,每个子矩阵对应一个,row = lane & 7.
    uint32_t a_addr[4];
    uint32_t r = lane & 7;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        uint32_t row    = ((k >> 1) << 3) + r;   // 0..15
        uint32_t col_st = (k & 1) << 4;          // 0 或 16
        a_addr[k] = a_smem + row * 32 + col_st;
    }
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.b16 "
        "{%0,%1,%2,%3}, "
        "[%4,%5,%6,%7];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(a_addr[0]), "r"(a_addr[1]), "r"(a_addr[2]), "r"(a_addr[3])
    );

    // B: ldmatrix.x2.trans.b16
    //   B 是 32 K × 8 N. sBn 是 [8 N-row × 32 K-byte] n-major.
    //   .trans 让 smem 行 → fragment 列:每读 16 byte (来自一个 n-row 的连续 16 K 字节),
    //   8 b16 派给 4 lane (每 lane 2 b16 = 4 fp8).
    //   4 lane 共享一个 n-row (n_idx = lane>>2);
    //     sub 0 → n_idx 行 K 字节 0..15
    //     sub 1 → n_idx 行 K 字节 16..31
    //   8 行 × 4 lane = 32 全覆盖;2 sub 拼回 32 K = B 的全部 K.
    uint32_t n_idx = lane >> 2;
    uint32_t b_addr[2];
    b_addr[0] = b_smem + n_idx * 32;        // sub 0: K[0..15] of n-row n_idx
    b_addr[1] = b_smem + n_idx * 32 + 16;   // sub 1: K[16..31] of n-row n_idx
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.b16 "
        "{%0,%1}, "
        "[%2,%3];\n"
        : "=r"(b[0]), "=r"(b[1])
        : "r"(b_addr[0]), "r"(b_addr[1])
    );
}

template <bool USE_LDSM>
__global__ void mma_kernel(const uint8_t* A, const uint8_t* B, float* D) {
    __shared__ uint8_t sA[16 * 32], sBk[32 * 8], sBn[8 * 32];
    for (int i = threadIdx.x; i < 16 * 32; i += 32) sA[i] = A[i];
    for (int i = threadIdx.x; i < 32 * 8; i += 32) {
        sBk[i] = B[i];
        sBn[(i & 7) * 32 + (i >> 3)] = B[i];  // 转成 n-major
    }
    __syncwarp();
    unsigned a[4], b[2];
    if constexpr (USE_LDSM)
        load_ldsm(sA, sBk, sBn, a, b);
    else
        load_manual(sA, sBk, sBn, a, b);
    float c[4] = {0, 0, 0, 0}, d[4];
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    int group = threadIdx.x >> 2, tig = threadIdx.x & 3;
    D[group * 8 + tig * 2] = d[0];
    D[group * 8 + tig * 2 + 1] = d[1];
    D[(group + 8) * 8 + tig * 2] = d[2];
    D[(group + 8) * 8 + tig * 2 + 1] = d[3];
}

static int run_path(bool ldsm, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 15);
    uint8_t hA[16 * 32], hB[32 * 8];
    float fA[16 * 32], fB[32 * 8], ref[16 * 8] = {};
    for (int i = 0; i < 16 * 32; i++) {
        __nv_fp8_e4m3 v = __nv_fp8_e4m3((float)(dist(rng) - 8));
        hA[i] = *(uint8_t*)&v;
        fA[i] = float(v);
    }
    for (int i = 0; i < 32 * 8; i++) {
        __nv_fp8_e4m3 v = __nv_fp8_e4m3((float)(dist(rng) - 8));
        hB[i] = *(uint8_t*)&v;
        fB[i] = float(v);
    }
    for (int r = 0; r < 16; r++)
        for (int n = 0; n < 8; n++)
            for (int k = 0; k < 32; k++)
                ref[r * 8 + n] += fA[r * 32 + k] * fB[k * 8 + n];
    uint8_t *dA, *dB;
    float* dD;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dD, 16 * 8 * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
    if (ldsm)
        mma_kernel<true><<<1, 32>>>(dA, dB, dD);
    else
        mma_kernel<false><<<1, 32>>>(dA, dB, dD);
    CUDA_CHECK_KERNEL();
    float got[16 * 8];
    CUDA_CHECK(cudaMemcpy(got, dD, sizeof(got), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < 16 * 8; i++) bad += got[i] != ref[i];
    cudaFree(dA); cudaFree(dB); cudaFree(dD);
    return bad;
}

int main() {
    long total = 0;
    for (unsigned s : {1u, 7u, 42u}) {
        int bm = run_path(false, s), bl = run_path(true, s);
        printf("seed=%-6u manual %s(%d)  ldsm %s(%d)\n", s,
               bm ? "FAIL" : "PASS", bm, bl ? "FAIL" : "PASS", bl);
        total += bm + bl;
    }
    return total != 0;
}
