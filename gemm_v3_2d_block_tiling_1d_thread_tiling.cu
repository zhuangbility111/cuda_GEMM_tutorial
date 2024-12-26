#include "gemm.h"
#include "gemm_kernel.cuh"

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


// version 4, 2d shared memory blocking and 1d thread blocking
template<int BLOCK_TILE_SIZE_X, int BLOCK_TILE_SIZE_Y, 
        int BLOCK_TILE_SIZE_K, int THREAD_TILE_SIZE_Y>
__global__ void gemm_v3_2d_block_tiling_1d_thread_tiling(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float A_shared_mem_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ float B_shared_mem_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    float C_reg_tile[THREAD_TILE_SIZE_Y] = {0.0};

    int num_shared_mem_tiles = (K + BLOCK_TILE_SIZE_K - 1) / BLOCK_TILE_SIZE_K;
    int thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
    constexpr int NUM_THREADS_PER_BLOCK = BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y / THREAD_TILE_SIZE_Y;

    for (int shared_mem_tile_idx = 0; shared_mem_tile_idx < num_shared_mem_tiles; shared_mem_tile_idx++) {

        // load data from global memory to shared memory
        load_data_from_global_mem_to_shared_mem<BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
                                                NUM_THREADS_PER_BLOCK>(A, B, A_shared_mem_tile, B_shared_mem_tile, 
                                                                       shared_mem_tile_idx, thread_linear_idx, M, N, K);
        
        __syncthreads();

        // compute, each thread compute a column, whose size is THREAD_TILE_SIZE_Y
        for (int k_idx = 0; k_idx < BLOCK_TILE_SIZE_K; k_idx++) {
            float B_val = B_shared_mem_tile[k_idx][thread_linear_idx % BLOCK_TILE_SIZE_X];
            for (int reg_tile_row_idx = 0; reg_tile_row_idx < THREAD_TILE_SIZE_Y; reg_tile_row_idx++) {
                float A_val = A_shared_mem_tile[thread_linear_idx / BLOCK_TILE_SIZE_X * THREAD_TILE_SIZE_Y + reg_tile_row_idx][k_idx];
                C_reg_tile[reg_tile_row_idx] += A_val * B_val;
            }
        }

        __syncthreads();

    }

    // write results from register to global memory
    for (int reg_tile_row_idx = 0; reg_tile_row_idx < THREAD_TILE_SIZE_Y; reg_tile_row_idx++) {
        int C_row_idx = blockIdx.y * BLOCK_TILE_SIZE_Y + thread_linear_idx / BLOCK_TILE_SIZE_X * THREAD_TILE_SIZE_Y + reg_tile_row_idx;
        int C_col_idx = blockIdx.x * BLOCK_TILE_SIZE_X + thread_linear_idx % BLOCK_TILE_SIZE_X;
        if (C_row_idx < M && C_col_idx < N) {
            C[C_row_idx * N + C_col_idx] = C_reg_tile[reg_tile_row_idx];
        }
    }
}

void launch_gemm_v3_2d_block_tiling_1d_thread_tiling(float *A, float *B, float *C, int M, int N, int K) {
    constexpr int BLOCK_TILE_SIZE_X = 64;
    constexpr int BLOCK_TILE_SIZE_Y = 64;
    constexpr int BLOCK_TILE_SIZE_K = 8;
    constexpr int THREAD_TILE_SIZE_Y = 8;
    constexpr int NUM_THREADS_PER_BLOCK = BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y / THREAD_TILE_SIZE_Y;
    static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_SIZE_Y == 0);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_K == 0);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_X == 0);
    dim3 const block_dim = {NUM_THREADS_PER_BLOCK, 1, 1};
    dim3 const grid_dim{
        (N + BLOCK_TILE_SIZE_X - 1) /
            BLOCK_TILE_SIZE_X,
        (M + BLOCK_TILE_SIZE_Y - 1) /
            BLOCK_TILE_SIZE_Y,
        1};

    // shared memory block size is BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K + BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X
    // register block size is THREAD_TILE_SIZE_Y * 1
    gemm_v3_2d_block_tiling_1d_thread_tiling<BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, THREAD_TILE_SIZE_Y>
                                            <<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}