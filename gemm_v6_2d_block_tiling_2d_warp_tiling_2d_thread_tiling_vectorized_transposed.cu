#include "gemm.h"
#include "gemm_kernel_utils.cuh"


// BLOCK_TILE_SIZE is the size of shared memory block tile, WARP_TILE_SIZE is the size of warp tile
// THREAD_TILE_SIZE is the size of regster block tile, NUM_THREAD_TILES_PER_WARP is the number of register block tiles per warp
// each thread will process NUM_THREAD_TILES_PER_WARP number of register block tile
// size of a register block tile is THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y
template <int BLOCK_TILE_SIZE, int WARP_TILE_SIZE, int THREAD_TILE_SIZE, int NUM_THREAD_TILES_PER_WARP>
__device__ void load_data_from_shared_memory_to_register_file_vectorized(float shared_mem_block_tile[BLOCK_TILE_SIZE],
                                                                        float reg_block_tile[NUM_THREAD_TILES_PER_WARP][THREAD_TILE_SIZE],
                                                                        int warp_idx, int thread_idx) {
    static_assert(BLOCK_TILE_SIZE % THREAD_TILE_SIZE == 0);
    constexpr int LEN_VECTOR = sizeof(int4) / sizeof(float);
    static_assert(sizeof(int4) % sizeof(float) == 0);
    constexpr int VECTORIZED_THREAD_TILE_SIZE = THREAD_TILE_SIZE / LEN_VECTOR;
    static_assert(THREAD_TILE_SIZE % LEN_VECTOR == 0);

#pragma unroll
    for (int reg_block_tile_idx = 0; reg_block_tile_idx < NUM_THREAD_TILES_PER_WARP; reg_block_tile_idx++) {
        int row_idx_in_shared_mem = warp_idx * WARP_TILE_SIZE + // warp tile idx
                                    reg_block_tile_idx * (WARP_TILE_SIZE / NUM_THREAD_TILES_PER_WARP) +
                                    thread_idx * THREAD_TILE_SIZE;
#pragma unroll
        for (int reg_block_tile_vector_idx = 0; reg_block_tile_vector_idx < VECTORIZED_THREAD_TILE_SIZE; reg_block_tile_vector_idx++) {
            *reinterpret_cast<int4*>(&reg_block_tile[reg_block_tile_idx][reg_block_tile_vector_idx * LEN_VECTOR]) = 
                *reinterpret_cast<int4*>(&shared_mem_block_tile[row_idx_in_shared_mem + reg_block_tile_vector_idx * LEN_VECTOR]);
        }
    }
}

template<int NUM_THREAD_TILES_PER_WARP_X, int NUM_THREAD_TILES_PER_WARP_Y, int THREAD_TILE_SIZE_X, int THREAD_TILE_SIZE_Y>
__device__ void compute_thread_tile_results(float A_reg_tile[NUM_THREAD_TILES_PER_WARP_Y][THREAD_TILE_SIZE_Y],
                                            float B_reg_tile[NUM_THREAD_TILES_PER_WARP_X][THREAD_TILE_SIZE_X],
                                            float C_reg_tile[NUM_THREAD_TILES_PER_WARP_Y][NUM_THREAD_TILES_PER_WARP_X]
                                                            [THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X]) {
    // Compute NUM_THREAD_TILES_PER_WARP_Y * NUM_THREAD_TILES_PER_WARP_X outer products.
#pragma unroll
    for (int reg_block_tile_row_idx = 0; reg_block_tile_row_idx < NUM_THREAD_TILES_PER_WARP_Y; reg_block_tile_row_idx++) {
#pragma unroll
        for (int reg_block_tile_col_idx = 0; reg_block_tile_col_idx < NUM_THREAD_TILES_PER_WARP_X; reg_block_tile_col_idx++) {
#pragma unroll
            for (int row_idx_in_each_tile = 0; row_idx_in_each_tile < THREAD_TILE_SIZE_Y; row_idx_in_each_tile++) {
#pragma unroll
                for (int col_idx_in_each_tile = 0; col_idx_in_each_tile < THREAD_TILE_SIZE_X; col_idx_in_each_tile++) {
                    C_reg_tile[reg_block_tile_row_idx][reg_block_tile_col_idx][row_idx_in_each_tile][col_idx_in_each_tile] += 
                        A_reg_tile[reg_block_tile_row_idx][row_idx_in_each_tile] * 
                        B_reg_tile[reg_block_tile_col_idx][col_idx_in_each_tile];
                }
            }
        }
    }
}


template<int BLOCK_TILE_SIZE_X, int BLOCK_TILE_SIZE_Y,
         int WARP_TILE_SIZE_X, int WARP_TILE_SIZE_Y,
         int THREAD_TILE_SIZE_X, int THREAD_TILE_SIZE_Y,
         int NUM_THREAD_TILES_PER_WARP_X, int NUM_THREAD_TILES_PER_WARP_Y>
__device__ void write_results_from_register_file_to_global_memory_vectorized(
    float C_reg_tile[NUM_THREAD_TILES_PER_WARP_Y][NUM_THREAD_TILES_PER_WARP_X][THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X],
    float* C, int M, int N, int block_row_idx, int block_col_idx, int warp_row_idx, int warp_col_idx, 
    int thread_row_idx_in_warp, int thread_col_idx_in_warp) {
    constexpr int LEN_VECTOR = sizeof(int4) / sizeof(float);
    static_assert(sizeof(int4) % sizeof(float) == 0);
    static_assert(BLOCK_TILE_SIZE_X % LEN_VECTOR == 0);
    constexpr int VECTORIZED_THREAD_TILE_SIZE_X = THREAD_TILE_SIZE_X / LEN_VECTOR;
    static_assert(THREAD_TILE_SIZE_X % LEN_VECTOR == 0);

#pragma unroll
    for (int reg_block_tile_row_idx = 0; reg_block_tile_row_idx < NUM_THREAD_TILES_PER_WARP_Y; reg_block_tile_row_idx++) {
#pragma unroll
        for (int reg_block_tile_col_idx = 0; reg_block_tile_col_idx < NUM_THREAD_TILES_PER_WARP_X; reg_block_tile_col_idx++) {
#pragma unroll
            for (int row_idx_in_each_tile = 0; row_idx_in_each_tile < THREAD_TILE_SIZE_Y; row_idx_in_each_tile++) {
#pragma unroll
                for (int vector_col_idx_in_each_tile = 0; vector_col_idx_in_each_tile < VECTORIZED_THREAD_TILE_SIZE_X; vector_col_idx_in_each_tile++) {
                    int C_row_idx = block_row_idx * BLOCK_TILE_SIZE_Y + warp_row_idx * WARP_TILE_SIZE_Y + 
                                        reg_block_tile_row_idx * (WARP_TILE_SIZE_Y / NUM_THREAD_TILES_PER_WARP_Y) + 
                                        thread_row_idx_in_warp * THREAD_TILE_SIZE_Y +
                                        row_idx_in_each_tile;
                    int C_col_idx = block_col_idx * BLOCK_TILE_SIZE_X + warp_col_idx * WARP_TILE_SIZE_X +
                                        reg_block_tile_col_idx * (WARP_TILE_SIZE_X / NUM_THREAD_TILES_PER_WARP_X) +
                                        thread_col_idx_in_warp * THREAD_TILE_SIZE_X +
                                        vector_col_idx_in_each_tile * LEN_VECTOR;

                    if (C_row_idx < M && C_col_idx < N) {
                        *reinterpret_cast<int4*>(&C[C_row_idx * N + C_col_idx]) = 
                            *reinterpret_cast<int4*>(&C_reg_tile[reg_block_tile_row_idx][reg_block_tile_col_idx][row_idx_in_each_tile][vector_col_idx_in_each_tile * LEN_VECTOR]);
                    }
                }
            }
        }
    }
}

// version 6, 2d shared memory blocking, 2d warp tiling and 2d thread blocking
// Each thread in the block processes THREAD_TILE_SIZE_Y *
// THREAD_TILE_SIZE_X output values. Number of threads BLOCK_TILE_SIZE_Y *
// BLOCK_TILE_SIZE_X / (THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X)
// BLOCK_TILE --> shared memory tile
// THREAD_TILE --> regster tile
template<int BLOCK_TILE_SIZE_X, int BLOCK_TILE_SIZE_Y, int BLOCK_TILE_SIZE_K,
         int WARP_TILE_SIZE_X, int WARP_TILE_SIZE_Y,
         int THREAD_TILE_SIZE_X, int THREAD_TILE_SIZE_Y,
         int NUM_THREADS_PER_WARP_X, int NUM_THREADS_PER_WARP_Y>
__global__ void gemm_v6_2d_block_tiling_2d_warp_tiling_2d_thread_tiling_vectorized_transposed(float *A, float *B, float *C, int M, int N, int K) {
    static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32);
    constexpr int NUM_WARPS_X = BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X;
    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0);
    constexpr int NUM_WARPS_Y = BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y;
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0);
    constexpr int NUM_THREAD_TILES_PER_WARP_X = WARP_TILE_SIZE_X / THREAD_TILE_SIZE_X / NUM_THREADS_PER_WARP_X;
    constexpr int NUM_THREAD_TILES_PER_WARP_Y = WARP_TILE_SIZE_Y / THREAD_TILE_SIZE_Y / NUM_THREADS_PER_WARP_Y;
    static_assert(WARP_TILE_SIZE_X % (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X) == 0);
    static_assert(WARP_TILE_SIZE_Y % (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y) == 0);

    // Avoid using blockDim.x * blockDim.y as the number of threads per block.
    // Because it is a runtime constant and the compiler cannot optimize the
    // loop unrolling based on that.
    // Use a compile time constant instead.
    constexpr int NUM_THREADS_PER_BLOCK_X = NUM_WARPS_X * NUM_THREADS_PER_WARP_X;
    constexpr int NUM_THREADS_PER_BLOCK_Y = NUM_WARPS_Y * NUM_THREADS_PER_WARP_Y;
    constexpr int NUM_THREADS_PER_BLOCK = NUM_THREADS_PER_BLOCK_X * NUM_THREADS_PER_BLOCK_Y;
    
    // allocate shared memory, A is transposed
    __shared__ float A_shared_mem_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y];
    __shared__ float B_shared_mem_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    // allocate register 
    float A_reg_tile[NUM_THREAD_TILES_PER_WARP_Y][THREAD_TILE_SIZE_Y] = {0.0};
    float B_reg_tile[NUM_THREAD_TILES_PER_WARP_X][THREAD_TILE_SIZE_X] = {0.0};
    // Each thread in the block processes NUM_THREAD_TILES_PER_WARP_Y *
    // NUM_THREAD_TILES_PER_WARP_X * THREAD_TILE_SIZE_Y *
    // THREAD_TILE_SIZE_X output values.
    float C_reg_tile[NUM_THREAD_TILES_PER_WARP_Y][NUM_THREAD_TILES_PER_WARP_X][THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {0.0};

    int thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_linear_idx = thread_linear_idx / 32;
    int warp_row_idx = warp_linear_idx / NUM_WARPS_X;
    int warp_col_dix = warp_linear_idx % NUM_WARPS_X;
    int thread_linear_idx_in_warp = thread_linear_idx % 32;
    int thread_row_idx_in_warp = thread_linear_idx_in_warp / NUM_THREADS_PER_WARP_X;
    int thread_col_idx_in_warp = thread_linear_idx_in_warp % NUM_THREADS_PER_WARP_X;

    int num_shared_mem_tiles = (K + BLOCK_TILE_SIZE_K - 1) / BLOCK_TILE_SIZE_K;


    for (int shared_mem_tile_idx = 0; shared_mem_tile_idx < num_shared_mem_tiles; shared_mem_tile_idx++) {
        // load data from global memory to shared memory by vectorized load, A is transposed, B is normal
        load_data_from_global_mem_to_shared_mem_transposed_vectorized
            <BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, NUM_THREADS_PER_BLOCK>
                (A, B, A_shared_mem_tile_transposed, B_shared_mem_tile, shared_mem_tile_idx, thread_linear_idx, M, N, K);

        __syncthreads();


#pragma unroll
        for (int k_idx = 0; k_idx < BLOCK_TILE_SIZE_K; k_idx++) {
            // load A from shared memory to register
            load_data_from_shared_memory_to_register_file_vectorized<BLOCK_TILE_SIZE_Y, WARP_TILE_SIZE_Y, 
                                                                    THREAD_TILE_SIZE_Y, NUM_THREAD_TILES_PER_WARP_Y>(
                                                                        A_shared_mem_tile_transposed[k_idx],
                                                                        A_reg_tile, 
                                                                        warp_row_idx, 
                                                                        thread_row_idx_in_warp
                                                                    );

            // load B from shared memory to register, using vectorized load
            load_data_from_shared_memory_to_register_file_vectorized<BLOCK_TILE_SIZE_X, WARP_TILE_SIZE_X, 
                                                                    THREAD_TILE_SIZE_X, NUM_THREAD_TILES_PER_WARP_X>(
                                                                        B_shared_mem_tile[k_idx],
                                                                        B_reg_tile, 
                                                                        warp_col_dix,
                                                                        thread_col_idx_in_warp
                                                                    );

            // compute, each thread compute a THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X block, which is stored in register files.
            // Compute a C tile whose size is (NUM_THREAD_TILES_PER_WARP_Y * NUM_THREAD_TILES_PER_WARP_X) (THREAD_TILE_SIZE_Y).
            // 计算C的一个reg tile
            // 这个tile由 NUM_THREAD_TILES_PER_WARP_Y * NUM_THREAD_TILES_PER_WARP_X 个 THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X 的小reg tile组成 
            compute_thread_tile_results<NUM_THREAD_TILES_PER_WARP_X, NUM_THREAD_TILES_PER_WARP_Y, THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y>(
                                            A_reg_tile, B_reg_tile, C_reg_tile
                                        );
        }
        __syncthreads();
    }

    // vectorized write results from register to global memory
    write_results_from_register_file_to_global_memory_vectorized<BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, 
                                                                WARP_TILE_SIZE_X, WARP_TILE_SIZE_Y,
                                                                THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y,
                                                                NUM_THREAD_TILES_PER_WARP_X, NUM_THREAD_TILES_PER_WARP_Y>(
                                                                    C_reg_tile, C, M, N, 
                                                                    blockIdx.y, blockIdx.x, warp_row_idx, warp_col_dix,
                                                                    thread_row_idx_in_warp, thread_col_idx_in_warp);
}

void launch_gemm_v6_2d_block_tiling_2d_warp_tiling_2d_thread_tiling_vectorized_transposed(float *A, float *B, float *C, int M, int N, int K) {
    constexpr int BLOCK_TILE_SIZE_X = 128;
    constexpr int BLOCK_TILE_SIZE_Y = 128;
    constexpr int BLOCK_TILE_SIZE_K = 16;

    constexpr int WARP_TILE_SIZE_X = 32;
    constexpr int WARP_TILE_SIZE_Y = 64;

    constexpr int NUM_WARPS_X = BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X;
    constexpr int NUM_WARPS_Y = BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y;
    // total number of warps in each block is NUM_WARPS_X * NUM_WARPS_Y

    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0);
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0);

    constexpr int THREAD_TILE_SIZE_X = 8;
    constexpr int THREAD_TILE_SIZE_Y = 8;

    constexpr int NUM_THREADS_PER_WARP_X = 4;
    constexpr int NUM_THREADS_PER_WARP_Y = 8;
    static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32);
    static_assert(WARP_TILE_SIZE_X % (NUM_THREADS_PER_WARP_X * THREAD_TILE_SIZE_X) == 0);
    static_assert(WARP_TILE_SIZE_Y % (NUM_THREADS_PER_WARP_Y * THREAD_TILE_SIZE_Y) == 0);

    // each thread in a thread block will compute 
    // (NUM_THREAD_TILES_PER_WARP_Y * NUM_THREAD_TILES_PER_WARP_X) x (THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X)
    // each thread block will compute a BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_X block
    // in preivous versions, each thread must compute 1 THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y tile, which means that 
    // the total number of threads is decided by the size of output matrix C (M * N)
    // if the size of C is too large, there will be too many threads invoked, which will incur resource shortage

    // but in this version, each thread will compute multiple THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y tiles
    // so the total number of threads is independent of the size of output matrix C
    // if the size of C is too large, we could control the total number of warps (NUM_WARPS_X * NUM_WARPS_Y),
    // since the number of threads in each warp is 32 and cannot be changed, we could let each thread in each warp compute
    // (NUM_THREAD_TILES_PER_WARP_X * NUM_THREAD_TILES_PER_WARP_Y) x (THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y) tiles
    // while NUM_THREAD_TILES_PER_WARP_X = WARP_TILE_SIZE_X / THREAD_TILE_SIZE_X / NUM_THREADS_PER_WARP_X
    // NUM_THREAD_TILES_PER_WARP_Y = WARP_TILE_SIZE_Y / THREAD_TILE_SIZE_Y / NUM_THREADS_PER_WARP_Y
    // Therefore, adding a layer of warp tiling here can actually make it easier to group and control threads.
    constexpr int NUM_THREADS_PER_BLOCK = (NUM_WARPS_Y * NUM_THREADS_PER_WARP_Y) * (NUM_WARPS_X * NUM_THREADS_PER_WARP_X);
    dim3 const block_dim = {NUM_THREADS_PER_BLOCK, 1, 1};
    dim3 const grid_dim{
        (N + BLOCK_TILE_SIZE_X - 1) /
            BLOCK_TILE_SIZE_X,
        (M + BLOCK_TILE_SIZE_Y - 1) /
            BLOCK_TILE_SIZE_Y,
        1};
    
    // shared memory block size is BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K + BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X
    // register block size is THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X
    gemm_v6_2d_block_tiling_2d_warp_tiling_2d_thread_tiling_vectorized_transposed
        <BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, WARP_TILE_SIZE_X, WARP_TILE_SIZE_Y, 
        THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y, NUM_THREADS_PER_WARP_X, NUM_THREADS_PER_WARP_Y>
            <<<grid_dim, block_dim>>>(A, B, C, M, N, K);
    
    // CHECK_LAST_CUDA_ERROR();
}