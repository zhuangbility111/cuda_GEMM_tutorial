#include "gemm.h"
#include "gemm_kernel.cuh"

int main() {
    int M, N, K;
    M = 2048;
    N = 2048;
    K = 2048;

    float *A, *B, *C_naive, *C_ref, *C_coalesced, *C_shared_mem_blocking, *C_shared_mem_blocking_1d_thread_blocking;
    A = new float[M * K];
    B = new float[K * N];
    C_naive = new float[M * N];
    C_coalesced = new float[M * N];
    C_ref = new float[M * N];
    C_shared_mem_blocking = new float[M * N];
    C_shared_mem_blocking_1d_thread_blocking = new float[M * N];

    for (int i = 0; i < M * K; i++)
        A[i] = (float)rand() / RAND_MAX;
    
    for (int i = 0; i < K * N; i++)
        // random number between 0 and 1
        B[i] = (float)rand() / RAND_MAX;

    memset(C_naive, 0, M * N * sizeof(float));
    memset(C_ref, 0, M * N * sizeof(float));
    memset(C_coalesced, 0, M * N * sizeof(float));
    memset(C_shared_mem_blocking, 0, M * N * sizeof(float));
    memset(C_shared_mem_blocking_1d_thread_blocking, 0, M * N * sizeof(float));

    dim3 grid(M / 32, N / 32);
    dim3 block(32, 32);

    run_perf_test(A, B, C_ref, M, N, K, 10, 100, grid, block, "cuBLAS", gemm_cublas);
    // run_perf_test(A, B, C_naive, M, N, K, 10, 100, grid, block, "naive", gemm_naive);
    // run_perf_test(A, B, C_coalesced, M, N, K, 10, 100, grid, block, "coalesced", gemm_coalesced);
    // run_perf_test(A, B, C_shared_mem_blocking, M, N, K, 10, 100, grid, block, "shared_mem_blocking", gemm_coalesced);
    run_perf_test_v1(A, B, C_shared_mem_blocking_1d_thread_blocking, M, N, K, 10, 100, "shared_mem_blocking_1d_thread_blocking", launch_gemm_v3_2d_block_tiling_1d_thread_tiling);

    // check_result(C_ref, C_naive, M, N, "naive");
    // check_result(C_ref, C_coalesced, M, N, "coalesced");
    // check_result(C_ref, C_shared_mem_blocking, M, N, "shared_mem_blocking");
    check_result(C_ref, C_shared_mem_blocking_1d_thread_blocking, M, N, "shared_mem_blocking_1d_thread_blocking");

    delete[] A;
    delete[] B;
    delete[] C_naive;
    delete[] C_ref;
    delete[] C_coalesced;
    delete[] C_shared_mem_blocking;
    delete[] C_shared_mem_blocking_1d_thread_blocking;
}