#include "gemm.h"
#include "gemm_kernel_utils.cuh"


// version 4, 2d shared memory blocking and 2d thread blocking
template<int BLOCK_TILE_SIZE_X, int BLOCK_TILE_SIZE_Y, int BLOCK_TILE_SIZE_K, 
         int THREAD_TILE_SIZE_X, int THREAD_TILE_SIZE_Y>
__global__ void gemm_v5_2d_block_tiling_2d_thread_tiling_vectorized_transposed(float *A, float *B, float *C, int M, int N, int K) {
    // allocate shared memory
    __shared__ float A_shared_mem_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y];
    __shared__ float B_shared_mem_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    // allocate register 
    float A_reg_tile[THREAD_TILE_SIZE_Y] = {0.0};
    float B_reg_tile[THREAD_TILE_SIZE_X] = {0.0};
    float C_reg_tile[THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {0.0};

    int num_shared_mem_tiles = (K + BLOCK_TILE_SIZE_K - 1) / BLOCK_TILE_SIZE_K;
    int thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
    constexpr int NUM_THREADS_PER_BLOCK = (BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y) / (THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y);
    constexpr int LEN_VECTOR = sizeof(int4) / sizeof(float);
    constexpr int VECTORIZED_THREAD_TILE_SIZE_X = THREAD_TILE_SIZE_X / LEN_VECTOR;
    static_assert(THREAD_TILE_SIZE_X % LEN_VECTOR == 0);

    for (int shared_mem_tile_idx = 0; shared_mem_tile_idx < num_shared_mem_tiles; shared_mem_tile_idx++) {
        // load data from global memory to shared memory by vectorized load, A is transposed, B is normal
        load_data_from_global_mem_to_shared_mem_transposed_vectorized
            <BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, NUM_THREADS_PER_BLOCK>
                (A, B, A_shared_mem_tile_transposed, B_shared_mem_tile, shared_mem_tile_idx, thread_linear_idx, M, N, K);

        __syncthreads();

        for (int k_idx = 0; k_idx < BLOCK_TILE_SIZE_K; k_idx++) {
            // load A from shared memory to register
            int const A_reg_tile_row_idx = thread_linear_idx / (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_Y;
            int const A_reg_tile_col_idx = k_idx;

            for (int reg_tile_row_idx = 0; reg_tile_row_idx < THREAD_TILE_SIZE_Y; reg_tile_row_idx++) {
                A_reg_tile[reg_tile_row_idx] = A_shared_mem_tile_transposed[A_reg_tile_col_idx][A_reg_tile_row_idx + reg_tile_row_idx];
            }

            // load B from shared memory to register, using vectorized load
            int const B_reg_tile_row_idx = k_idx;
            int const B_reg_tile_col_idx = thread_linear_idx % (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_X;

            for (int reg_tile_col_idx = 0; reg_tile_col_idx < VECTORIZED_THREAD_TILE_SIZE_X; reg_tile_col_idx++) {
                // B_reg_tile[reg_tile_col_idx] = B_shared_mem_tile[B_reg_tile_row_idx][B_reg_tile_col_idx + reg_tile_col_idx];
                *reinterpret_cast<int4*>(&B_reg_tile[reg_tile_col_idx * LEN_VECTOR]) = 
                    *reinterpret_cast<int4*>(&B_shared_mem_tile[B_reg_tile_row_idx][B_reg_tile_col_idx + reg_tile_col_idx * LEN_VECTOR]);
            }

            // compute, each thread compute a THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X block, which is stored in register files.
            for (int reg_tile_row_idx = 0; reg_tile_row_idx < THREAD_TILE_SIZE_Y; reg_tile_row_idx++) {
                // get A from register files
                float A_val = A_reg_tile[reg_tile_row_idx];
                for (int reg_tile_col_idx = 0; reg_tile_col_idx < THREAD_TILE_SIZE_X; reg_tile_col_idx++) {
                    // get B from register files
                    float B_val = B_reg_tile[reg_tile_col_idx];
                    C_reg_tile[reg_tile_row_idx][reg_tile_col_idx] += A_val * B_val;
                }
            }
        }

        __syncthreads();
    }

    // vectorized write results from register to global memory
    for (int reg_tile_row_idx = 0; reg_tile_row_idx < THREAD_TILE_SIZE_Y; reg_tile_row_idx++) {
        for (int reg_tile_col_idx = 0; reg_tile_col_idx < VECTORIZED_THREAD_TILE_SIZE_X; reg_tile_col_idx++) {
            int C_row_idx = blockIdx.y * BLOCK_TILE_SIZE_Y + thread_linear_idx / (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_Y + reg_tile_row_idx;
            int C_col_idx = blockIdx.x * BLOCK_TILE_SIZE_X + thread_linear_idx % (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_X + reg_tile_col_idx * LEN_VECTOR;
            if (C_row_idx < M && C_col_idx < N) {
                *reinterpret_cast<int4*>(&C[C_row_idx * N + C_col_idx]) = 
                    *reinterpret_cast<int4*>(&C_reg_tile[reg_tile_row_idx][reg_tile_col_idx * LEN_VECTOR]);
            }
        }
    }
}

void launch_gemm_v5_2d_block_tiling_2d_thread_tiling_vectorized_transposed(float *A, float *B, float *C, int M, int N, int K) {
    constexpr int BLOCK_TILE_SIZE_X = 128;
    constexpr int BLOCK_TILE_SIZE_Y = 128;
    constexpr int BLOCK_TILE_SIZE_K = 16;
    constexpr int THREAD_TILE_SIZE_Y = 8;
    constexpr int THREAD_TILE_SIZE_X = 8;
    // each thread in a thread block will compute a THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X block
    // each thread block will compute a BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_X block
    constexpr int NUM_THREADS_PER_BLOCK = (BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y) / (THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y);
    static_assert(BLOCK_TILE_SIZE_X % THREAD_TILE_SIZE_X == 0);
    static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_SIZE_Y == 0);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_K == 0);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_X == 0);
    static_assert((BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K) % NUM_THREADS_PER_BLOCK == 0);
    static_assert((BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K) % NUM_THREADS_PER_BLOCK == 0);
    dim3 const block_dim = {NUM_THREADS_PER_BLOCK, 1, 1};
    dim3 const grid_dim{
        (N + BLOCK_TILE_SIZE_X - 1) /
            BLOCK_TILE_SIZE_X,
        (M + BLOCK_TILE_SIZE_Y - 1) /
            BLOCK_TILE_SIZE_Y,
        1};

    // shared memory block size is BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K + BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X
    // register block size is THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X
    gemm_v5_2d_block_tiling_2d_thread_tiling_vectorized_transposed
        <BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y>
            <<<grid_dim, block_dim>>>(A, B, C, M, N, K);
    
}