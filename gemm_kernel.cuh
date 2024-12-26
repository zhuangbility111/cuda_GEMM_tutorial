#ifndef GEMM_KERNEL_CUH
#define GEMM_KERNEL_CUH

void launch_gemm_v3_2d_block_tiling_1d_thread_tiling(float *A, float *B, float *C, int M, int N, int K);

#endif // GEMM_KERNEL_CUH