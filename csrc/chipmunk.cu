#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <Python.h>
#include <cuda_runtime.h>

extern "C" {
  /* Creates a dummy empty cuda module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit_cuda(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "cuda",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace chipmunk {

#ifdef KITTENS_HOPPER
// Sparse MLP
extern void csp_mlp_mm1(at::Tensor a, at::Tensor b_colmajor, at::Tensor c, at::Tensor bias, at::Tensor pa_cache_colmajor, at::Tensor indices, at::Tensor indices_counts);
extern void csp_mlp_mm2_and_scatter_add(at::Tensor packed, at::Tensor unpacked_colmajor, at::Tensor sp_inds, at::Tensor sp_counts, at::Tensor mma_a, at::Tensor mma_b, at::Tensor mma_c, int64_t num_sms_scatter_add, int64_t matmul_kernel);

// Sparse+Dense Attention
extern void csp_attn(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o, at::Tensor indices, at::Tensor indices_counts, int64_t o_scale);
extern std::vector<at::Tensor> dense_attn(at::Tensor q, at::Tensor k, at::Tensor v);
extern std::vector<at::Tensor> dense_colsum_attn(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor p);

// Indexed IO
extern void csp_scatter_add(at::Tensor packed, at::Tensor unpacked_colmajor, at::Tensor sp_inds, at::Tensor sp_counts, int64_t num_sms);
#endif

// Indexed IO
extern void copy_indices(at::Tensor bmfc1, at::Tensor bm_mid_cache, at::Tensor sp_inds, at::Tensor sp_counts);
extern void topk_indices(at::Tensor activation, at::Tensor indices, at::Tensor counts, double sparsity_amount, int64_t multiple_of, double random_amount);
extern std::vector<at::Tensor> mask_to_indices(at::Tensor mask, int64_t multiple_of, int64_t pad_to_multiple_of);


TORCH_LIBRARY(chipmunk, m) {
#ifdef KITTENS_HOPPER
    // Sparse MLP
    m.def("csp_mlp_mm1(Tensor a, Tensor b_colmajor, Tensor(c!) c, Tensor bias, Tensor pa_cache_colmajor, Tensor indices, Tensor indices_counts) -> ()");
    m.def("csp_mlp_mm2_and_scatter_add(Tensor packed, Tensor(unpacked_colmajor!) unpacked_colmajor, Tensor sp_inds, Tensor sp_counts, Tensor mma_a, Tensor mma_b, Tensor mma_c, int num_sms_scatter_add, int matmul_kernel) -> ()");

    // Sparse+Dense Attention
    m.def("csp_attn(Tensor q, Tensor k, Tensor v, Tensor o, Tensor indices, Tensor indices_counts, int o_scale) -> ()");
    m.def("dense_attn(Tensor q, Tensor k, Tensor v) -> Tensor[]");
    m.def("dense_colsum_attn(Tensor q, Tensor k, Tensor v, Tensor p) -> Tensor[]");

    // Indexed IO
    m.def("csp_scatter_add(Tensor packed, Tensor(unpacked_colmajor!) unpacked_colmajor, Tensor sp_inds, Tensor sp_counts, int num_sms) -> ()");
#endif
    // Indexed IO
    m.def("copy_indices(Tensor bmfc1, Tensor(bm_mid_cache!) bm_mid_cache, Tensor sp_inds, Tensor sp_counts) -> ()");
    m.def("topk_indices(Tensor activation, Tensor(indices!) indices, Tensor counts, float sparsity_amount, int multiple_of, float random_amount) -> ()");
    m.def("mask_to_indices(Tensor mask, int multiple_of, int pad_to_multiple_of) -> Tensor[]");
}


TORCH_LIBRARY_IMPL(chipmunk, CUDA, m) {
#ifdef KITTENS_HOPPER
    // Sparse MLP
    m.impl("csp_mlp_mm1", &csp_mlp_mm1);
    m.impl("csp_mlp_mm2_and_scatter_add", &csp_mlp_mm2_and_scatter_add);

    // Sparse+Dense Attention
    m.impl("csp_attn", &csp_attn);
    m.impl("dense_attn", &dense_attn);
    m.impl("dense_colsum_attn", &dense_colsum_attn);

    // Indexed IO
    m.impl("csp_scatter_add", &csp_scatter_add);
#endif    
    // Indexed IO
    m.impl("copy_indices", &copy_indices);
    m.impl("topk_indices", &topk_indices);
    m.impl("mask_to_indices", &mask_to_indices);
}

}