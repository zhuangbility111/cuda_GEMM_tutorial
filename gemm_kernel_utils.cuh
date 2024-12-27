#ifndef GEMM_KERNEL_CUH
#define GEMM_KERNEL_CUH

#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA_ERROR(call)                                      \
do {                                                                \
    cudaError_t err = call;                                         \
    if (err != cudaSuccess) {                                       \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)      \
                  << " in " << __FILE__ << " at line " << __LINE__ \
                  << std::endl;                                     \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while (0)

// load data from global memory to shared memory
template<int BLOCK_TILE_SIZE_X, int BLOCK_TILE_SIZE_Y, int BLOCK_TILE_SIZE_K, int NUM_THREADS_PER_BLOCK>
__device__ void load_data_from_global_mem_to_shared_mem(float *A, float *B, 
                                                        float A_shared_mem_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K],
                                                        float B_shared_mem_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X],
                                                        int shared_mem_tile_idx, int thread_linear_idx, 
                                                        int m, int n, int k) {
    // load A
    // all threads in a thread block will load a (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K) block from memory to shared memory together. 
    // all threads perform a coalesced load (thread0 for elem0, thread1 for elem1, thread2 for elem2, ...)
    constexpr int NUM_CHUNKS_PER_BLOCK_A = (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
    for (int load_idx = 0; load_idx < NUM_CHUNKS_PER_BLOCK_A; load_idx++) {
        int A_shared_mem_tile_row_idx = (thread_linear_idx + load_idx * NUM_THREADS_PER_BLOCK) / BLOCK_TILE_SIZE_K;
        int A_shared_mem_tile_col_idx = (thread_linear_idx + load_idx * NUM_THREADS_PER_BLOCK) % BLOCK_TILE_SIZE_K;

        int A_row_idx = blockIdx.y * BLOCK_TILE_SIZE_Y + A_shared_mem_tile_row_idx;
        int A_col_idx = shared_mem_tile_idx * BLOCK_TILE_SIZE_K + A_shared_mem_tile_col_idx;

        if (A_row_idx < m && A_col_idx < k) {
            A_shared_mem_tile[A_shared_mem_tile_row_idx][A_shared_mem_tile_col_idx] = A[A_row_idx * k + A_col_idx];
        }
    }

    // load B
    // all threads in a thread block will load a (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X) block from memory to shared memory together. 
    // all threads perform a coalesced load (thread0 for elem0, thread1 for elem1, thread2 for elem2, ...)
    constexpr int NUM_CHUNKS_PER_BLOCK_B = (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
    for (int load_idx = 0; load_idx < NUM_CHUNKS_PER_BLOCK_B; load_idx++) {
        int B_shared_mem_tile_row_idx = (thread_linear_idx + load_idx * NUM_THREADS_PER_BLOCK) / BLOCK_TILE_SIZE_X;
        int B_shared_mem_tile_col_idx = (thread_linear_idx + load_idx * NUM_THREADS_PER_BLOCK) % BLOCK_TILE_SIZE_X;

        int B_row_idx = shared_mem_tile_idx * BLOCK_TILE_SIZE_K + B_shared_mem_tile_row_idx;
        int B_col_idx = blockIdx.x * BLOCK_TILE_SIZE_X + B_shared_mem_tile_col_idx;

        if (B_row_idx < k && B_col_idx < n) {
            B_shared_mem_tile[B_shared_mem_tile_row_idx][B_shared_mem_tile_col_idx] = B[B_row_idx * n + B_col_idx];
        }
    }

}

#endif // GEMM_KERNEL_CUH