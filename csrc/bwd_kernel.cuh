#pragma once

#include "fwd_kernel.cuh"

namespace flashattn_bwd_detail {

__device__ __forceinline__ void mma_m16n8k16(
    float* c, const uint32_t* a, const uint32_t* b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

template<int N, bool FULL_TILES>
__device__ __forceinline__ void load_stats_tile(
    const float* __restrict__ g, uint32_t smem, int tid, int tile, int total)
{
    static_assert(N > 0 && N <= 128, "stats tile must fit in one pass of 128 threads");
    if (tid < N) {
        const int  grow = tile * N + tid;
        const bool ok   = FULL_TILES || grow < total;
        const float* src = g + (ok ? grow : 0);
        const int bytes  = ok ? 4 : 0;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                     :: "r"(smem + static_cast<uint32_t>(tid * sizeof(float))),
                        "l"(src), "r"(bytes) : "memory");
    }
}

template<int D_PAD, bool FULL_TILES>
__device__ __forceinline__ void store_frag_f16(
    const float (&acc)[D_PAD / 8 * 4],
    __half* __restrict__ out, int row0, int rows_total, int lane4, float scale)
{
    constexpr int Dv = D_PAD / 8;
    const int row1 = row0 + 8;
    const int col0 = lane4 * 2;
    #pragma unroll
    for (int vs = 0; vs < Dv; ++vs) {
        const int c = vs * 8 + col0;
        if (FULL_TILES || row0 < rows_total) {
            *reinterpret_cast<__half2*>(&out[static_cast<size_t>(row0) * D_PAD + c]) =
                __floats2half2_rn(acc[vs*4+0] * scale, acc[vs*4+1] * scale);
        }
        if (FULL_TILES || row1 < rows_total) {
            *reinterpret_cast<__half2*>(&out[static_cast<size_t>(row1) * D_PAD + c]) =
                __floats2half2_rn(acc[vs*4+2] * scale, acc[vs*4+3] * scale);
        }
    }
}

}

template<int D_PAD>
__global__ void flashattn_bwd_preprocess(
    const __half* __restrict__ O,
    const __half* __restrict__ dO,
    const float*  __restrict__ L,
          float*  __restrict__ Delta,
          float*  __restrict__ L2,
    int actual_D,
    int total_q_rows
) {
    static_assert(D_PAD > 0 && D_PAD % 16 == 0, "D_PAD must be divisible by 16");
    if (blockDim.x < 32 || (blockDim.x & 31) != 0) return;
    if (actual_D <= 0 || actual_D > D_PAD) return;

    const int warp            = threadIdx.x >> 5;
    const int lane            = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int row             = blockIdx.x * warps_per_block + warp;

    if (row < total_q_rows) {
        const __half* o_row  = O  + static_cast<size_t>(row) * actual_D;
        const __half* do_row = dO + static_cast<size_t>(row) * actual_D;

        float sum = 0.0f;
        for (int col = lane; col < actual_D; col += 32)
            sum += __half2float(o_row[col]) * __half2float(do_row[col]);

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);

        if (lane == 0) {
            constexpr float kLog2e = 1.4426950408889634f;
            const float l = L[row];
            Delta[row] = sum;
            L2[row]    = (l == -INFINITY) ? INFINITY : l * kLog2e;
        }
    }
}

template<int Bc>
__host__ __device__ constexpr int flashattn_bwd_dq_region_rows() {
    return (4 * Bc > 64) ? 4 * Bc : 64;
}

template<int Bc, int D_PAD>
__host__ __device__ constexpr size_t flashattn_bwd_dq_smem_bytes() {
    return static_cast<size_t>(64 + flashattn_bwd_dq_region_rows<Bc>()) *
           (D_PAD + 8) * sizeof(__half);
}

template<int Bc, int D_PAD, bool masked, bool FULL_TILES>
__global__ void __launch_bounds__(128, 2)
flashattn_bwd_dq(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const __half* __restrict__ dO,
    const float*  __restrict__ L2,
    const float*  __restrict__ Delta,
          __half* __restrict__ dQ,
    const int actual_D,
    const int Skv,
    const int Sq,
    const int numKVheads
) {
    using flashattn_bwd_detail::mma_m16n8k16;
    using flashattn_bwd_detail::store_frag_f16;

    constexpr int Br = 64;
    static_assert(Bc >= 16 && Bc % 16 == 0, "Bc must be a positive multiple of 16");
    static_assert(D_PAD > 0 && D_PAD % 16 == 0, "D_PAD must be positive and divisible by 16");

    if (blockDim.x != 128) return;
    if (actual_D != D_PAD || Sq <= 0 || Skv <= 0) return;
    if constexpr (FULL_TILES) {
        if ((Sq % Br) != 0 || (Skv % Bc) != 0) return;
    }

    constexpr float kLog2e = 1.4426950408889634f;

    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int lane4  = lane & 3;
    const int lane16 = lane & 15;

    const int tileid    = blockIdx.x;
    const int headid    = blockIdx.y;
    const int batchid   = blockIdx.z;
    const int num_heads = gridDim.y;

    if (numKVheads <= 0 || num_heads % numKVheads != 0) return;
    const int kv_headid = headid / (num_heads / numKVheads);

    const long long q_base =
        (static_cast<long long>(batchid) * num_heads + headid) * Sq * D_PAD;
    const long long kv_base =
        (static_cast<long long>(batchid) * numKVheads + kv_headid) * Skv * D_PAD;
    const long long stat_base =
        (static_cast<long long>(batchid) * num_heads + headid) * Sq;

    const __half* Qptr  = Q  + q_base;
    const __half* dOptr = dO + q_base;
    const __half* Kptr  = K  + kv_base;
    const __half* Vptr  = V  + kv_base;
          __half* dQptr = dQ + q_base;
    const float*  L2ptr = L2    + stat_base;
    const float*  Dptr  = Delta + stat_base;

    const int Tr = (Sq + Br - 1) / Br;
    if (tileid >= Tr) return;
    const int q_block_start = tileid * Br;

    const int kv_off = Skv - Sq;
    int kv_end = Skv;
    if constexpr (masked) {
        const int causal_end = q_block_start + Br + kv_off;
        if (causal_end < kv_end) kv_end = causal_end;
    }
    const int kv_tiles_to_process = kv_end > 0 ? (kv_end + Bc - 1) / Bc : 0;

    constexpr int STRIDE = D_PAD + 8;
    constexpr int Dk = D_PAD / 16;
    constexpr int Bk = Bc / 8;
    constexpr int Dv = D_PAD / 8;

    extern __shared__ __align__(16) char smem_raw[];
    const uint32_t S0  = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
    const uint32_t DO0 = S0;
    const uint32_t REG = S0 + Br * STRIDE * sizeof(__half);
    constexpr uint32_t TILE_BYTES  = Bc * STRIDE * sizeof(__half);
    constexpr uint32_t STAGE_BYTES = 2 * TILE_BYTES;

    const uint32_t a_lane =
        static_cast<uint32_t>(((warp * 16 + lane16) * STRIDE + ((lane < 16) ? 0 : 8)) * sizeof(__half));
    const uint32_t k_lane =
        static_cast<uint32_t>(((lane16 & 7) * STRIDE + (lane16 >> 3) * 8) * sizeof(__half));
    const uint32_t v_lane =
        static_cast<uint32_t>((lane16 * STRIDE) * sizeof(__half));

    float dQ_frag[Dv * 4];
    #pragma unroll
    for (int i = 0; i < Dv * 4; ++i) dQ_frag[i] = 0.0f;

    const float scale      = 1.0f / sqrtf(static_cast<float>(D_PAD));
    const float scale_log2 = scale * kLog2e;

    load_tile_dispatch<Br, D_PAD, STRIDE, FULL_TILES>(Qptr,  REG, tid, tileid, Sq);
    load_tile_dispatch<Br, D_PAD, STRIDE, FULL_TILES>(dOptr, DO0, tid, tileid, Sq);
    asm volatile("cp.async.commit_group;\n");

    const int row0 = q_block_start + warp * 16 + (lane >> 2);
    const int row1 = row0 + 8;
    const float l2_0 = (FULL_TILES || row0 < Sq) ? L2ptr[row0] : INFINITY;
    const float l2_1 = (FULL_TILES || row1 < Sq) ? L2ptr[row1] : INFINITY;
    const float dl_0 = (FULL_TILES || row0 < Sq) ? Dptr[row0]  : 0.0f;
    const float dl_1 = (FULL_TILES || row1 < Sq) ? Dptr[row1]  : 0.0f;

    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    uint32_t q_frag[Dk][4];
    #pragma unroll
    for (int ks = 0; ks < Dk; ++ks) {
        ldmatrix_x4(q_frag[ks], REG + a_lane + static_cast<uint32_t>(ks * 16 * sizeof(__half)));
    }
    __syncthreads();

    auto issue_kv = [&](int tile, int stage) {
        const uint32_t base = REG + stage * STAGE_BYTES;
        load_tile_dispatch<Bc, D_PAD, STRIDE, FULL_TILES>(Kptr, base,              tid, tile, Skv);
        load_tile_dispatch<Bc, D_PAD, STRIDE, FULL_TILES>(Vptr, base + TILE_BYTES, tid, tile, Skv);
        asm volatile("cp.async.commit_group;\n");
    };

    if (kv_tiles_to_process > 0) {
        issue_kv(0, 0);
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
    }

    const int warp_row0 = q_block_start + warp * 16 + kv_off;
    const int row_lim0  = warp_row0 + (lane >> 2);
    const int row_lim1  = row_lim0 + 8;

    for (int kv_tile = 0; kv_tile < kv_tiles_to_process; ++kv_tile) {
        const int stage = kv_tile & 1;

        if (kv_tile + 1 < kv_tiles_to_process) issue_kv(kv_tile + 1, stage ^ 1);

        const uint32_t K_s = REG + stage * STAGE_BYTES;
        const uint32_t V_s = K_s + TILE_BYTES;

        float S_frag[Bk * 4];
        float dP_frag[Bk * 4];
        #pragma unroll
        for (int i = 0; i < Bk * 4; ++i) { S_frag[i] = 0.0f; dP_frag[i] = 0.0f; }

        #pragma unroll
        for (int ks = 0; ks < Dk; ++ks) {
            uint32_t do_frag[4];
            ldmatrix_x4(do_frag, DO0 + a_lane + static_cast<uint32_t>(ks * 16 * sizeof(__half)));
            #pragma unroll
            for (int kb = 0; kb < Bk; ++kb) {
                const uint32_t off =
                    k_lane + static_cast<uint32_t>((kb * 8 * STRIDE + ks * 16) * sizeof(__half));
                uint32_t k_frag[2], v_frag[2];
                ldmatrix_x2(k_frag, K_s + off);
                ldmatrix_x2(v_frag, V_s + off);
                mma_m16n8k16(&S_frag[kb * 4],  q_frag[ks], k_frag);
                mma_m16n8k16(&dP_frag[kb * 4], do_frag,    v_frag);
            }
        }

        const int kv_start = kv_tile * Bc;

        if constexpr (!FULL_TILES) {
            if (kv_start + Bc > Skv) {
                #pragma unroll
                for (int kb = 0; kb < Bk; ++kb) {
                    const int key0 = kv_start + kb * 8 + lane4 * 2;
                    if (key0     >= Skv) { S_frag[kb*4+0] = -INFINITY; S_frag[kb*4+2] = -INFINITY; }
                    if (key0 + 1 >= Skv) { S_frag[kb*4+1] = -INFINITY; S_frag[kb*4+3] = -INFINITY; }
                }
            }
        }

        if constexpr (masked) {
            if (kv_start + Bc - 1 > warp_row0) {
                #pragma unroll
                for (int kb = 0; kb < Bk; ++kb) {
                    const int key0 = kv_start + kb * 8 + lane4 * 2;
                    if (key0     > row_lim0) S_frag[kb*4+0] = -INFINITY;
                    if (key0 + 1 > row_lim0) S_frag[kb*4+1] = -INFINITY;
                    if (key0     > row_lim1) S_frag[kb*4+2] = -INFINITY;
                    if (key0 + 1 > row_lim1) S_frag[kb*4+3] = -INFINITY;
                }
            }
        }

        #pragma unroll
        for (int kb = 0; kb < Bk; kb += 2) {
            const float p0 = exp2f(S_frag[ kb    * 4 + 0] * scale_log2 - l2_0);
            const float p1 = exp2f(S_frag[ kb    * 4 + 1] * scale_log2 - l2_0);
            const float p2 = exp2f(S_frag[ kb    * 4 + 2] * scale_log2 - l2_1);
            const float p3 = exp2f(S_frag[ kb    * 4 + 3] * scale_log2 - l2_1);
            const float p4 = exp2f(S_frag[(kb+1) * 4 + 0] * scale_log2 - l2_0);
            const float p5 = exp2f(S_frag[(kb+1) * 4 + 1] * scale_log2 - l2_0);
            const float p6 = exp2f(S_frag[(kb+1) * 4 + 2] * scale_log2 - l2_1);
            const float p7 = exp2f(S_frag[(kb+1) * 4 + 3] * scale_log2 - l2_1);

            uint32_t ds_frag[4];
            ds_frag[0] = pack_float2_to_half2_u32(p0 * (dP_frag[ kb    * 4 + 0] - dl_0),
                                                  p1 * (dP_frag[ kb    * 4 + 1] - dl_0));
            ds_frag[1] = pack_float2_to_half2_u32(p2 * (dP_frag[ kb    * 4 + 2] - dl_1),
                                                  p3 * (dP_frag[ kb    * 4 + 3] - dl_1));
            ds_frag[2] = pack_float2_to_half2_u32(p4 * (dP_frag[(kb+1) * 4 + 0] - dl_0),
                                                  p5 * (dP_frag[(kb+1) * 4 + 1] - dl_0));
            ds_frag[3] = pack_float2_to_half2_u32(p6 * (dP_frag[(kb+1) * 4 + 2] - dl_1),
                                                  p7 * (dP_frag[(kb+1) * 4 + 3] - dl_1));

            #pragma unroll
            for (int vs = 0; vs < Dv; ++vs) {
                uint32_t kt_frag[2];
                ldmatrix_x2_trans(kt_frag,
                    K_s + v_lane +
                    static_cast<uint32_t>(((kb / 2) * 16 * STRIDE + vs * 8) * sizeof(__half)));
                mma_m16n8k16(&dQ_frag[vs * 4], ds_frag, kt_frag);
            }
        }

        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
    }

    store_frag_f16<D_PAD, FULL_TILES>(dQ_frag, dQptr, row0, Sq, lane4, scale);
}

template<int Bq, int NS, int D_PAD>
__host__ __device__ constexpr size_t flashattn_bwd_dkdv_smem_bytes() {
    return static_cast<size_t>(2 * 64 + 2 * NS * Bq) * (D_PAD + 8) * sizeof(__half)
         + static_cast<size_t>(2 * NS * Bq) * sizeof(float);
}

template<int Bq, int NS, int D_PAD, bool masked, bool FULL_TILES>
__global__ void __launch_bounds__(128, 2)
flashattn_bwd_dkdv(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const __half* __restrict__ dO,
    const float*  __restrict__ L2,
    const float*  __restrict__ Delta,
          __half* __restrict__ dK,
          __half* __restrict__ dV,
    const int actual_D,
    const int Skv,
    const int Sq,
    const int numKVheads,
    const int numQheads
) {
    using flashattn_bwd_detail::mma_m16n8k16;
    using flashattn_bwd_detail::load_stats_tile;
    using flashattn_bwd_detail::store_frag_f16;

    constexpr int Bkv = 64;
    static_assert(Bq >= 16 && Bq % 16 == 0 && Bq <= 128, "Bq must be a multiple of 16, <= 128");
    static_assert(NS == 1 || NS == 2, "NS must be 1 or 2");
    static_assert(D_PAD > 0 && D_PAD % 16 == 0, "D_PAD must be positive and divisible by 16");

    if (blockDim.x != 128) return;
    if (actual_D != D_PAD || Sq <= 0 || Skv <= 0) return;
    if (numKVheads <= 0 || numQheads % numKVheads != 0) return;
    if constexpr (FULL_TILES) {
        if ((Skv % Bkv) != 0 || (Sq % Bq) != 0) return;
    }

    constexpr float kLog2e = 1.4426950408889634f;

    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int lane4  = lane & 3;
    const int lane16 = lane & 15;

    const int tileid    = blockIdx.x;
    const int kv_headid = blockIdx.y;
    const int batchid   = blockIdx.z;
    const int group     = numQheads / numKVheads;

    const int Tkv = (Skv + Bkv - 1) / Bkv;
    if (tileid >= Tkv) return;
    const int kv_start = tileid * Bkv;

    const long long kv_base =
        (static_cast<long long>(batchid) * numKVheads + kv_headid) * Skv * D_PAD;
    const __half* Kptr = K + kv_base;
    const __half* Vptr = V + kv_base;
          __half* dKptr = dK + kv_base;
          __half* dVptr = dV + kv_base;

    const int kv_off = Skv - Sq;
    int first_tile = 0;
    if constexpr (masked) {
        const int first_q = kv_start - kv_off;
        if (first_q > 0) first_tile = first_q / Bq;
    }
    const int q_tiles = (Sq + Bq - 1) / Bq;
    int steps = q_tiles - first_tile;
    if (steps < 0) steps = 0;
    const int total_steps = steps * group;

    constexpr int STRIDE = D_PAD + 8;
    constexpr int Dk = D_PAD / 16;
    constexpr int Bk = Bq / 8;
    constexpr int Dv = D_PAD / 8;

    extern __shared__ __align__(16) char smem_raw[];
    const uint32_t S0 = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
    constexpr uint32_t KV_TILE  = Bkv * STRIDE * sizeof(__half);
    constexpr uint32_t Q_TILE   = Bq  * STRIDE * sizeof(__half);
    const uint32_t Ks   = S0;
    const uint32_t Vs   = Ks + KV_TILE;
    const uint32_t Qs0  = Vs + KV_TILE;
    const uint32_t dOs0 = Qs0 + NS * Q_TILE;
    constexpr uint32_t STAT_OFF = 2 * KV_TILE + 2 * NS * Q_TILE;
    const uint32_t L2s0 = S0 + STAT_OFF;
    const uint32_t Ds0  = L2s0 + NS * Bq * sizeof(float);
    const float* L2s_f  = reinterpret_cast<const float*>(smem_raw + STAT_OFF);
    const float* Ds_f   = L2s_f + NS * Bq;

    const uint32_t a_lane =
        static_cast<uint32_t>(((warp * 16 + lane16) * STRIDE + ((lane < 16) ? 0 : 8)) * sizeof(__half));
    const uint32_t k_lane =
        static_cast<uint32_t>(((lane16 & 7) * STRIDE + (lane16 >> 3) * 8) * sizeof(__half));
    const uint32_t v_lane =
        static_cast<uint32_t>((lane16 * STRIDE) * sizeof(__half));

    float dK_frag[Dv * 4];
    float dV_frag[Dv * 4];
    #pragma unroll
    for (int i = 0; i < Dv * 4; ++i) { dK_frag[i] = 0.0f; dV_frag[i] = 0.0f; }

    const float scale      = 1.0f / sqrtf(static_cast<float>(D_PAD));
    const float scale_log2 = scale * kLog2e;

    auto issue_step = [&](int flat) {
        const int gi    = flat / steps;
        const int qt    = first_tile + (flat - gi * steps);
        const int stage = (NS == 2) ? (flat & 1) : 0;
        const int qh    = kv_headid * group + gi;
        const long long qrow0 = (static_cast<long long>(batchid) * numQheads + qh) * Sq;

        load_tile_dispatch<Bq, D_PAD, STRIDE, FULL_TILES>(
            Q  + qrow0 * D_PAD, Qs0  + stage * Q_TILE, tid, qt, Sq);
        load_tile_dispatch<Bq, D_PAD, STRIDE, FULL_TILES>(
            dO + qrow0 * D_PAD, dOs0 + stage * Q_TILE, tid, qt, Sq);
        load_stats_tile<Bq, FULL_TILES>(
            L2    + qrow0, L2s0 + stage * Bq * sizeof(float), tid, qt, Sq);
        load_stats_tile<Bq, FULL_TILES>(
            Delta + qrow0, Ds0  + stage * Bq * sizeof(float), tid, qt, Sq);
        asm volatile("cp.async.commit_group;\n");
    };

    load_tile_dispatch<Bkv, D_PAD, STRIDE, FULL_TILES>(Kptr, Ks, tid, tileid, Skv);
    load_tile_dispatch<Bkv, D_PAD, STRIDE, FULL_TILES>(Vptr, Vs, tid, tileid, Skv);
    if (total_steps > 0) {
        issue_step(0);
    } else {
        asm volatile("cp.async.commit_group;\n");
    }
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    const int kvrow0 = kv_start + warp * 16 + (lane >> 2);
    const int kvrow1 = kvrow0 + 8;
    const int warp_kv_last = kv_start + warp * 16 + 15;

    for (int step = 0; step < total_steps; ++step) {
        if constexpr (NS == 2) {

            if (step + 1 < total_steps) issue_step(step + 1);
        }

        const int stage   = (NS == 2) ? (step & 1) : 0;
        const int qt      = first_tile + (step % steps);
        const int q_start = qt * Bq;

        const uint32_t Q_s  = Qs0  + stage * Q_TILE;
        const uint32_t dO_s = dOs0 + stage * Q_TILE;
        const float*   L2s  = L2s_f + stage * Bq;
        const float*   Ds   = Ds_f  + stage * Bq;

        float S_frag[Bk * 4];
        float dP_frag[Bk * 4];
        #pragma unroll
        for (int i = 0; i < Bk * 4; ++i) { S_frag[i] = 0.0f; dP_frag[i] = 0.0f; }

        #pragma unroll
        for (int ks = 0; ks < Dk; ++ks) {
            uint32_t k_a[4], v_a[4];
            ldmatrix_x4(k_a, Ks + a_lane + static_cast<uint32_t>(ks * 16 * sizeof(__half)));
            ldmatrix_x4(v_a, Vs + a_lane + static_cast<uint32_t>(ks * 16 * sizeof(__half)));
            #pragma unroll
            for (int kb = 0; kb < Bk; ++kb) {
                const uint32_t off =
                    k_lane + static_cast<uint32_t>((kb * 8 * STRIDE + ks * 16) * sizeof(__half));
                uint32_t q_b[2], do_b[2];
                ldmatrix_x2(q_b,  Q_s  + off);
                ldmatrix_x2(do_b, dO_s + off);
                mma_m16n8k16(&S_frag[kb * 4],  k_a, q_b);
                mma_m16n8k16(&dP_frag[kb * 4], v_a, do_b);
            }
        }

        if constexpr (!FULL_TILES) {
            if (q_start + Bq > Sq) {
                #pragma unroll
                for (int kb = 0; kb < Bk; ++kb) {
                    const int q0 = q_start + kb * 8 + lane4 * 2;
                    if (q0     >= Sq) { S_frag[kb*4+0] = -INFINITY; S_frag[kb*4+2] = -INFINITY; }
                    if (q0 + 1 >= Sq) { S_frag[kb*4+1] = -INFINITY; S_frag[kb*4+3] = -INFINITY; }
                }
            }
        }

        if constexpr (masked) {
            if (q_start + kv_off < warp_kv_last) {
                #pragma unroll
                for (int kb = 0; kb < Bk; ++kb) {
                    const int q0 = q_start + kb * 8 + lane4 * 2;
                    if (kvrow0 > q0     + kv_off) S_frag[kb*4+0] = -INFINITY;
                    if (kvrow0 > q0 + 1 + kv_off) S_frag[kb*4+1] = -INFINITY;
                    if (kvrow1 > q0     + kv_off) S_frag[kb*4+2] = -INFINITY;
                    if (kvrow1 > q0 + 1 + kv_off) S_frag[kb*4+3] = -INFINITY;
                }
            }
        }

        #pragma unroll
        for (int kb = 0; kb < Bk; kb += 2) {
            const float2 la = *reinterpret_cast<const float2*>(&L2s[ kb      * 8 + lane4 * 2]);
            const float2 lb = *reinterpret_cast<const float2*>(&L2s[(kb + 1) * 8 + lane4 * 2]);
            const float2 da = *reinterpret_cast<const float2*>(&Ds [ kb      * 8 + lane4 * 2]);
            const float2 db = *reinterpret_cast<const float2*>(&Ds [(kb + 1) * 8 + lane4 * 2]);

            const float p0 = exp2f(S_frag[ kb    * 4 + 0] * scale_log2 - la.x);
            const float p1 = exp2f(S_frag[ kb    * 4 + 1] * scale_log2 - la.y);
            const float p2 = exp2f(S_frag[ kb    * 4 + 2] * scale_log2 - la.x);
            const float p3 = exp2f(S_frag[ kb    * 4 + 3] * scale_log2 - la.y);
            const float p4 = exp2f(S_frag[(kb+1) * 4 + 0] * scale_log2 - lb.x);
            const float p5 = exp2f(S_frag[(kb+1) * 4 + 1] * scale_log2 - lb.y);
            const float p6 = exp2f(S_frag[(kb+1) * 4 + 2] * scale_log2 - lb.x);
            const float p7 = exp2f(S_frag[(kb+1) * 4 + 3] * scale_log2 - lb.y);

            uint32_t p_frag[4], ds_frag[4];
            p_frag[0] = pack_float2_to_half2_u32(p0, p1);
            p_frag[1] = pack_float2_to_half2_u32(p2, p3);
            p_frag[2] = pack_float2_to_half2_u32(p4, p5);
            p_frag[3] = pack_float2_to_half2_u32(p6, p7);

            ds_frag[0] = pack_float2_to_half2_u32(p0 * (dP_frag[ kb    * 4 + 0] - da.x),
                                                  p1 * (dP_frag[ kb    * 4 + 1] - da.y));
            ds_frag[1] = pack_float2_to_half2_u32(p2 * (dP_frag[ kb    * 4 + 2] - da.x),
                                                  p3 * (dP_frag[ kb    * 4 + 3] - da.y));
            ds_frag[2] = pack_float2_to_half2_u32(p4 * (dP_frag[(kb+1) * 4 + 0] - db.x),
                                                  p5 * (dP_frag[(kb+1) * 4 + 1] - db.y));
            ds_frag[3] = pack_float2_to_half2_u32(p6 * (dP_frag[(kb+1) * 4 + 2] - db.x),
                                                  p7 * (dP_frag[(kb+1) * 4 + 3] - db.y));

            #pragma unroll
            for (int vs = 0; vs < Dv; ++vs) {
                const uint32_t off = v_lane +
                    static_cast<uint32_t>(((kb / 2) * 16 * STRIDE + vs * 8) * sizeof(__half));
                uint32_t do_t[2], q_t[2];
                ldmatrix_x2_trans(do_t, dO_s + off);
                ldmatrix_x2_trans(q_t,  Q_s  + off);
                mma_m16n8k16(&dV_frag[vs * 4], p_frag,  do_t);
                mma_m16n8k16(&dK_frag[vs * 4], ds_frag, q_t);
            }
        }

        if constexpr (NS == 2) {

            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            __syncthreads();
        } else {
            __syncthreads();
            if (step + 1 < total_steps) {
                issue_step(step + 1);
                asm volatile("cp.async.wait_group 0;\n" ::: "memory");
                __syncthreads();
            }
        }
    }

    store_frag_f16<D_PAD, FULL_TILES>(dK_frag, dKptr, kvrow0, Skv, lane4, scale);
    store_frag_f16<D_PAD, FULL_TILES>(dV_frag, dVptr, kvrow0, Skv, lane4, 1.0f);
}
