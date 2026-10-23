#pragma once

#include "bwd_kernel.cuh"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <limits>
#include <vector>

template<int D_PAD, int Bq, int NS, bool Masked, bool FULL>
static void launch_bwd_dkdv(
    const torch::Tensor& Q, const torch::Tensor& K, const torch::Tensor& V,
    const torch::Tensor& dO, const torch::Tensor& L2, const torch::Tensor& Delta,
    torch::Tensor& dK, torch::Tensor& dV, cudaStream_t stream)
{
    constexpr size_t smem = flashattn_bwd_dkdv_smem_bytes<Bq, NS, D_PAD>();
    const int B   = static_cast<int>(Q.size(0));
    const int H   = static_cast<int>(Q.size(1));
    const int Sq  = static_cast<int>(Q.size(2));
    const int Skv = static_cast<int>(K.size(2));
    const int Hkv = static_cast<int>(K.size(1));

    auto kernel = flashattn_bwd_dkdv<Bq, NS, D_PAD, Masked, FULL>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));

    dim3 grid((Skv + 63) / 64, Hkv, B);
    kernel<<<grid, dim3(128), smem, stream>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(dO.data_ptr<at::Half>()),
        L2.data_ptr<float>(), Delta.data_ptr<float>(),
        reinterpret_cast<__half*>(dK.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(dV.data_ptr<at::Half>()),
        D_PAD, Skv, Sq, Hkv, H);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int D_PAD, int Bc, bool Masked, bool FULL>
static void launch_bwd_dq(
    const torch::Tensor& Q, const torch::Tensor& K, const torch::Tensor& V,
    const torch::Tensor& dO, const torch::Tensor& L2, const torch::Tensor& Delta,
    torch::Tensor& dQ, cudaStream_t stream)
{
    constexpr size_t smem = flashattn_bwd_dq_smem_bytes<Bc, D_PAD>();
    const int B   = static_cast<int>(Q.size(0));
    const int H   = static_cast<int>(Q.size(1));
    const int Sq  = static_cast<int>(Q.size(2));
    const int Skv = static_cast<int>(K.size(2));
    const int Hkv = static_cast<int>(K.size(1));

    auto kernel = flashattn_bwd_dq<Bc, D_PAD, Masked, FULL>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));

    dim3 grid((Sq + 63) / 64, H, B);
    kernel<<<grid, dim3(128), smem, stream>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(dO.data_ptr<at::Half>()),
        L2.data_ptr<float>(), Delta.data_ptr<float>(),
        reinterpret_cast<__half*>(dQ.data_ptr<at::Half>()),
        D_PAD, Skv, Sq, Hkv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int D_PAD, int Bq, int NS, int Bc, bool Masked>
static std::vector<torch::Tensor> launch_bwd_impl(
    const torch::Tensor& Q, const torch::Tensor& K, const torch::Tensor& V,
    const torch::Tensor& O, const torch::Tensor& dO, const torch::Tensor& L)
{
    const int B   = static_cast<int>(Q.size(0));
    const int H   = static_cast<int>(Q.size(1));
    const int Sq  = static_cast<int>(Q.size(2));
    const int Skv = static_cast<int>(K.size(2));
    const int actual_D = static_cast<int>(Q.size(3));
    const int Hkv = static_cast<int>(K.size(1));

    TORCH_CHECK(actual_D == D_PAD, "head dim ", actual_D, " != D_PAD ", D_PAD);
    TORCH_CHECK(Hkv > 0 && H % Hkv == 0, "num_heads must be a multiple of num_kv_heads");
    TORCH_CHECK(B <= 65535, "batch too large for grid.z");
    const int64_t total_rows_64 = static_cast<int64_t>(B) * H * Sq;
    TORCH_CHECK(total_rows_64 <= std::numeric_limits<int>::max(), "B*H*Sq is too large");
    const int total_rows = static_cast<int>(total_rows_64);

    auto dQ = torch::empty_like(Q);
    auto dK = torch::empty_like(K);
    auto dV = torch::empty_like(V);
    auto fopt = Q.options().dtype(torch::kFloat32);
    auto Delta = torch::empty({Q.size(0), Q.size(1), Q.size(2)}, fopt);
    auto L2    = torch::empty({Q.size(0), Q.size(1), Q.size(2)}, fopt);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(Q.get_device());

    const int rows_per_block = 4;
    flashattn_bwd_preprocess<D_PAD>
        <<<(total_rows + rows_per_block - 1) / rows_per_block, dim3(128), 0, stream>>>(
            reinterpret_cast<const __half*>(O.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(dO.data_ptr<at::Half>()),
            L.data_ptr<float>(), Delta.data_ptr<float>(), L2.data_ptr<float>(),
            actual_D, total_rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if ((Skv % 64 == 0) && (Sq % Bq == 0))
        launch_bwd_dkdv<D_PAD, Bq, NS, Masked, true >(Q, K, V, dO, L2, Delta, dK, dV, stream);
    else
        launch_bwd_dkdv<D_PAD, Bq, NS, Masked, false>(Q, K, V, dO, L2, Delta, dK, dV, stream);

    if ((Sq % 64 == 0) && (Skv % Bc == 0))
        launch_bwd_dq<D_PAD, Bc, Masked, true >(Q, K, V, dO, L2, Delta, dQ, stream);
    else
        launch_bwd_dq<D_PAD, Bc, Masked, false>(Q, K, V, dO, L2, Delta, dQ, stream);

    return {dQ, dK, dV};
}

template<bool Masked>
static std::vector<torch::Tensor> dispatch_bwd(
    int d_pad,
    const torch::Tensor& Q, const torch::Tensor& K, const torch::Tensor& V,
    const torch::Tensor& O, const torch::Tensor& dO, const torch::Tensor& L)
{
    switch (d_pad) {
        case  32: return launch_bwd_impl< 32, 64, 2, 32, Masked>(Q, K, V, O, dO, L);
        case  64: return launch_bwd_impl< 64, 32, 2, 32, Masked>(Q, K, V, O, dO, L);
        case  80: return launch_bwd_impl< 80, 32, 2, 32, Masked>(Q, K, V, O, dO, L);
        case  96: return launch_bwd_impl< 96, 16, 2, 32, Masked>(Q, K, V, O, dO, L);
        case 128: return launch_bwd_impl<128, 16, 1, 16, Masked>(Q, K, V, O, dO, L);
        case 160: return launch_bwd_impl<160, 16, 2, 16, Masked>(Q, K, V, O, dO, L);
        case 192: return launch_bwd_impl<192, 16, 2, 16, Masked>(Q, K, V, O, dO, L);
        case 224: return launch_bwd_impl<224, 16, 2, 16, Masked>(Q, K, V, O, dO, L);
        case 256: return launch_bwd_impl<256, 16, 1, 16, Masked>(Q, K, V, O, dO, L);
    }
    TORCH_CHECK(false, "unsupported D_PAD");
    return {};
}
