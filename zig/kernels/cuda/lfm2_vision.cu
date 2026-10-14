// LFM2-VL's vision glue (families/lfm2/vision.zig): SigLIP2's LayerNorm, Linear epilogues, attention heads, unshuffle.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "glue_math.cuh"

using namespace tf_glue;

namespace {

constexpr int NT = 256;    // threads a row block
constexpr int MAXE = 8;    // elements a thread holds: LayerNorm rows up to NT * MAXE = 2048 wide
constexpr int AR = 8;      // query rows an attention block takes, a warp each
constexpr int KMAX = 1024; // keys a query row sees
constexpr int VC = 32;     // keys a value chunk stages
constexpr int DMAX = 128;  // head dims a value chunk stages
constexpr int OUTS = AR * DMAX / NT;  // outputs a thread accumulates

// A block's values summed: warp butterflies, then the warps in order; every thread gets the sum.
__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = red[0];
    for (int w = 1; w < NT / 32; ++w) s = __fadd_rn(s, red[w]);
    __syncthreads();
    return s;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

// torch's GELU kernels in fp32: the tanh approximation, and the exact form through erf.
__device__ __forceinline__ float gelu_tanh(float x) {
    constexpr float beta = M_SQRT2 * M_2_SQRTPI * 0.5;
    constexpr float kappa = 0.044715f;
    const float cube = __fmul_rn(__fmul_rn(x, x), x);
    const float inner = __fmul_rn(beta, __fadd_rn(x, __fmul_rn(kappa, cube)));
    return __fmul_rn(__fmul_rn(0.5f, x), __fadd_rn(1.0f, tanhf(inner)));
}

__device__ __forceinline__ float gelu_erf(float x) {
    constexpr float alpha = M_SQRT1_2;
    return __fmul_rn(__fmul_rn(x, 0.5f), __fadd_rn(1.0f, erff(__fmul_rn(x, alpha))));
}

}  // namespace

// Block a row: y = bf16((x - mean) * rstd * w + b), mean and variance in two fixed-order passes.
extern "C" __global__ void __launch_bounds__(NT) tf_lfm2v_layer_norm(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                       const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ Y, int D,
                                                                       float eps) {
    __shared__ float red[NT / 32];
    const int64_t at = static_cast<int64_t>(blockIdx.x) * D;
    float x[MAXE];
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXE; ++k) {
        const int i = threadIdx.x + k * NT;
        if (i >= D) continue;
        x[k] = bf(X[at + i]);
        acc = __fadd_rn(acc, x[k]);
    }
    const float mean = __fdiv_rn(block_sum(acc, red), static_cast<float>(D));
    acc = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXE; ++k) {
        if (threadIdx.x + k * NT >= D) continue;
        x[k] = __fsub_rn(x[k], mean);
        acc = __fadd_rn(acc, __fmul_rn(x[k], x[k]));
    }
    const float var = __fdiv_rn(block_sum(acc, red), static_cast<float>(D));
    const float rstd = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(var, eps)));
#pragma unroll
    for (int k = 0; k < MAXE; ++k) {
        const int i = threadIdx.x + k * NT;
        if (i < D) Y[at + i] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(__fmul_rn(x[k], rstd), bf(W[i])), bf(B[i])));
    }
}

// Block (row, column block): v = bf16(acc + bias), then bf16(gelu(v)) (act 1: tanh, 2: erf), then bf16(r + v) when R.
extern "C" __global__ void __launch_bounds__(NT) tf_lfm2v_bias(const float* __restrict__ ACC, const __nv_bfloat16* __restrict__ BIAS,
                                                                 const __nv_bfloat16* R, __nv_bfloat16* O, int N, int act) {
    const int c = blockIdx.y * NT + threadIdx.x;
    if (c >= N) return;
    const int64_t at = static_cast<int64_t>(blockIdx.x) * N + c;
    float v = rbf(__fadd_rn(ACC[at], bf(BIAS[c])));
    if (act == 1) v = rbf(gelu_tanh(v));
    if (act == 2) v = rbf(gelu_erf(v));
    O[at] = __float2bfloat16_rn(R ? __fadd_rn(bf(R[at]), v) : v);
}

// Block a row: its q and k heads of qkv [rows, 3 H D] to [H][rows][D], heads hs apart.
extern "C" __global__ void __launch_bounds__(NT) tf_lfm2v_split_heads(const __nv_bfloat16* __restrict__ QKV, __nv_bfloat16* __restrict__ Q,
                                                                        __nv_bfloat16* __restrict__ K, int H, int D, int64_t hs) {
    const int64_t r = blockIdx.x;
    const __nv_bfloat16* src = QKV + r * 3 * H * D;
    for (int i = threadIdx.x; i < H * D; i += NT) {
        const int64_t at = (i / D) * hs + r * D + i % D;
        Q[at] = src[i];
        K[at] = src[H * D + i];
    }
}

// Block (8 query rows, head), a warp a row: p = softmax(s * scale) kept f32, a = bf16(sum_j p_j v_j) into [rows, H D].
extern "C" __global__ void __launch_bounds__(NT) tf_lfm2v_attend(const float* __restrict__ S, const __nv_bfloat16* __restrict__ QKV,
                                                                   __nv_bfloat16* __restrict__ A, int rows, int H, int D, int64_t ss, float scale) {
    __shared__ float p[AR][KMAX];
    __shared__ __nv_bfloat16 vs[VC][DMAX];
    const int h = blockIdx.y, r0 = blockIdx.x * AR, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (r0 + warp < rows) {
        const float* s = S + h * ss + static_cast<int64_t>(r0 + warp) * rows;
        float* pr = p[warp];
        float m = -INFINITY;
        for (int j = lane; j < rows; j += 32) {
            pr[j] = __fmul_rn(s[j], scale);
            m = fmaxf(m, pr[j]);
        }
        m = warp_max(m);
        float acc = 0.0f;
        for (int j = lane; j < rows; j += 32) {
            pr[j] = tf_exp(__fsub_rn(pr[j], m));
            acc = __fadd_rn(acc, pr[j]);
        }
        const float sum = warp_sum(acc);
        for (int j = lane; j < rows; j += 32) pr[j] = __fdiv_rn(pr[j], sum);
    }
    const int64_t width = 3 * static_cast<int64_t>(H) * D;
    const __nv_bfloat16* v = QKV + 2 * H * D + h * D;
    float o[OUTS] = {};
    for (int j0 = 0; j0 < rows; j0 += VC) {
        const int n = min(VC, rows - j0);
        __syncthreads();
        for (int i = threadIdx.x; i < n * D; i += NT) vs[i / D][i % D] = v[(j0 + i / D) * width + i % D];
        __syncthreads();
#pragma unroll
        for (int k = 0; k < OUTS; ++k) {
            const int at = threadIdx.x + k * NT, r = at / D, d = at % D;
            if (at >= AR * D || r0 + r >= rows) continue;
            for (int j = 0; j < n; ++j) o[k] = __fmaf_rn(p[r][j0 + j], bf(vs[j][d]), o[k]);
        }
    }
#pragma unroll
    for (int k = 0; k < OUTS; ++k) {
        const int at = threadIdx.x + k * NT, r = r0 + at / D;
        if (at < AR * D && r < rows) A[static_cast<int64_t>(r) * H * D + h * D + at % D] = __float2bfloat16_rn(o[k]);
    }
}

// Block an output row (i, j) of an h x w grid: rows (2i, 2j), (2i, 2j+1), (2i+1, 2j), (2i+1, 2j+1) side by side.
extern "C" __global__ void __launch_bounds__(NT) tf_lfm2v_unshuffle(const __nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ Y, int w,
                                                                      int C) {
    const int o = blockIdx.x, i = o / (w / 2), j = o % (w / 2);
    for (int e = threadIdx.x; e < 4 * C; e += NT) {
        const int q = e / C;
        const int64_t src = static_cast<int64_t>(2 * i + q / 2) * w + 2 * j + q % 2;
        Y[static_cast<int64_t>(o) * 4 * C + e] = X[src * C + e % C];
    }
}
