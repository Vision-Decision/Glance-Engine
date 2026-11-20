// Dense FA2 forward using the split count chosen for a single image. The Python
// dispatcher checks the exact PyTorch build before accessing its internal interface.
#include "params.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <cmath>

namespace cutlass { struct bfloat16_t; }
namespace pytorch_flash {
template<typename T, int HeadDim, bool IsCausal>
void run_mha_fwd_splitkv_dispatch(Flash_fwd_params&, CUstream_st*);
}

static at::Tensor split_attention(const at::Tensor& q, const at::Tensor& k,
                                  const at::Tensor& v, int64_t splits, double scale) {
    TORCH_CHECK(q.is_cuda() && k.device() == q.device() && v.device() == q.device());
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type());
    TORCH_CHECK(q.dim() == 4 && k.sizes() == q.sizes() && v.sizes() == q.sizes());
    TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1);
    const int batch = q.size(0), len = q.size(1), heads = q.size(2), dim = q.size(3);
    TORCH_CHECK(batch > 0 && batch <= 16 && heads == 16 && dim == 64);
    TORCH_CHECK((len == 468 && splits == 2) || (len == 1008 && splits == 4));
    const c10::cuda::CUDAGuard guard(q.device());
    auto out = at::empty(q.sizes(), q.options());
    auto fp32 = q.options().dtype(at::kFloat);
    auto lse = at::empty({batch, heads, len}, fp32);
    auto sums = at::empty({splits, batch, heads, len}, fp32);
    auto partial = at::empty({splits, batch, heads, len, dim}, fp32);
    pytorch_flash::Flash_fwd_params p{};
    p.q_ptr = q.data_ptr(); p.k_ptr = k.data_ptr(); p.v_ptr = v.data_ptr(); p.o_ptr = out.data_ptr();
    p.q_batch_stride = q.stride(0); p.k_batch_stride = k.stride(0);
    p.v_batch_stride = v.stride(0); p.o_batch_stride = out.stride(0);
    p.q_row_stride = q.stride(1); p.k_row_stride = k.stride(1);
    p.v_row_stride = v.stride(1); p.o_row_stride = out.stride(1);
    p.q_head_stride = q.stride(2); p.k_head_stride = k.stride(2);
    p.v_head_stride = v.stride(2); p.o_head_stride = out.stride(2);
    p.b = batch; p.h = p.h_k = heads; p.h_h_k_ratio = 1;
    p.seqlen_q = p.seqlen_k = len; p.seqlen_q_rounded = p.seqlen_k_rounded = (len + 127) / 128 * 128;
    p.d = p.d_rounded = dim; p.scale_softmax = scale; p.scale_softmax_log2 = p.scale_softmax * float(M_LOG2E);
    p.is_bf16 = true; p.is_seqlens_k_cumulative = true; p.num_splits = splits;
    p.window_size_left = p.window_size_right = -1;
    p.p_dropout = p.rp_dropout = 1.0f; p.p_dropout_in_uint8_t = 255;
    p.scale_softmax_rp_dropout = p.scale_softmax;
    p.softmax_lse_ptr = lse.data_ptr(); p.softmax_lseaccum_ptr = sums.data_ptr(); p.oaccum_ptr = partial.data_ptr();
    pytorch_flash::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 64, false>(p, at::cuda::getCurrentCUDAStream().stream());
    return out;
}

TORCH_LIBRARY(glance_vision, m) {
    m.def("split_attention(Tensor q, Tensor k, Tensor v, int splits, float scale) -> Tensor");
}
TORCH_LIBRARY_IMPL(glance_vision, CUDA, m) {
    m.impl("split_attention", TORCH_FN(split_attention));
}
