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

#define CHECK_LAST_CUDA_ERROR() check_cuda_last(__FILE__, __LINE__)

void check_cuda_last(const char* const file, const int line);

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

// load data from global memory to shared memory (vectorized load and transposed)
// A is transposed shared memory tile, B is normal shared memory tile
template<int BLOCK_TILE_SIZE_X, int BLOCK_TILE_SIZE_Y, int BLOCK_TILE_SIZE_K, int NUM_THREADS_PER_BLOCK,
        int BLOCK_TILE_SKEW_SIZE_X = 0, int BLOCK_TILE_SKEW_SIZE_Y = 0, typename VECTOR_TYPE = int4>
__device__ void load_data_from_global_mem_to_shared_mem_transposed_vectorized(float *A, float *B, 
                                                        float A_shared_mem_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y],
                                                        float B_shared_mem_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
                                                        int shared_mem_tile_idx, int thread_linear_idx, 
                                                        int m, int n, int k) {
    
    constexpr int LEN_VECTOR = sizeof(VECTOR_TYPE) / sizeof(float);
    static_assert(BLOCK_TILE_SIZE_K % LEN_VECTOR == 0);
    static_assert(BLOCK_TILE_SIZE_X % LEN_VECTOR == 0);

    // The skew size could affect the data alignment in shared memory when we
    // use vectorized load. We need to make sure the data alignment is correct.
    static_assert((BLOCK_TILE_SIZE_Y) * sizeof(float) % sizeof(VECTOR_TYPE) == 0);
    static_assert((BLOCK_TILE_SIZE_X) * sizeof(float) % sizeof(VECTOR_TYPE) == 0);
    static_assert((BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y) * sizeof(float) % sizeof(VECTOR_TYPE) == 0);
    static_assert((BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X) * sizeof(float) % sizeof(VECTOR_TYPE) == 0);
    constexpr int VECTORIZED_BLOCK_TILE_SIZE_K = BLOCK_TILE_SIZE_K / LEN_VECTOR; // for vectorized load A
    constexpr int VECTORIZED_BLOCK_TILE_SIZE_X = BLOCK_TILE_SIZE_X / LEN_VECTOR; // for vectorized load B
    
    // load A
    // all threads in a thread block will load a (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K) block from memory to shared memory together. 
    // all threads perform a coalesced load and vectorized (vector len is 4 float numbers) 
    // load (thread0 for elem0, elem1, elem2, elem3, thread1 for elem4, elem5, elem6, elem7, thread2 for elem8, elem9, elem10, elem11, ...)
    // but have to perform a scalerized write as we need to write data to a transposed array
    constexpr int NUM_CHUNKS_PER_BLOCK_A = (BLOCK_TILE_SIZE_Y * VECTORIZED_BLOCK_TILE_SIZE_K + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
    for (int load_idx = 0; load_idx < NUM_CHUNKS_PER_BLOCK_A; load_idx++) {
        int A_shared_mem_tile_row_idx = (thread_linear_idx + load_idx * NUM_THREADS_PER_BLOCK) / VECTORIZED_BLOCK_TILE_SIZE_K;
        int A_shared_mem_tile_col_idx = (thread_linear_idx + load_idx * NUM_THREADS_PER_BLOCK) % VECTORIZED_BLOCK_TILE_SIZE_K * LEN_VECTOR;

        int A_row_idx = blockIdx.y * BLOCK_TILE_SIZE_Y + A_shared_mem_tile_row_idx;
        int A_col_idx = shared_mem_tile_idx * BLOCK_TILE_SIZE_K + A_shared_mem_tile_col_idx;

        // vector for vectorized load
        int4 A_row_vector_vals{0, 0, 0, 0};
        if (A_row_idx < m && A_col_idx < k) {
            A_row_vector_vals = *reinterpret_cast<int4*>(&A[A_row_idx * k + A_col_idx]);
        }

        // for corner case where col_idx greater than k, zeros the invalid numbers (outside the boundary)
        if (A_col_idx + LEN_VECTOR > k) {
            int num_invalid_numbers = A_col_idx + LEN_VECTOR - k;
            // use a float pointer to handle each elem in int4
            float* A_row_vector_vals_ptr = reinterpret_cast<float*>(&A_row_vector_vals);
            for (int i = 0; i < num_invalid_numbers; i++) {
                A_row_vector_vals_ptr[LEN_VECTOR - 1 - i] = 0;
            }
        }

        // scalarized write for transposed shared memory array
        if (A_shared_mem_tile_row_idx < BLOCK_TILE_SIZE_Y && A_shared_mem_tile_col_idx < BLOCK_TILE_SIZE_K) {
            for (int i = 0; i < LEN_VECTOR; i++) {
                A_shared_mem_tile_transposed[A_shared_mem_tile_col_idx + i][A_shared_mem_tile_row_idx] = 
                    reinterpret_cast<float*>(&A_row_vector_vals)[i];
            }
        }
    }

    // load B
    // all threads in a thread block will load a (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X) block from memory to shared memory together. 
    // all threads perform a coalesced load and vectorized (vector len is 4 float numbers) 
    // load (thread0 for elem0, elem1, elem2, elem3, thread1 for elem4, elem5, elem6, elem7, thread2 for elem8, elem9, elem10, elem11, ...)
    // we could do the vectorized write
    constexpr int NUM_CHUNKS_PER_BLOCK_B = (BLOCK_TILE_SIZE_K * VECTORIZED_BLOCK_TILE_SIZE_X + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
    for (int load_idx = 0; load_idx < NUM_CHUNKS_PER_BLOCK_B; load_idx++) {
        int B_shared_mem_tile_row_idx = (thread_linear_idx + load_idx * NUM_THREADS_PER_BLOCK) / VECTORIZED_BLOCK_TILE_SIZE_X;
        int B_shared_mem_tile_col_idx = (thread_linear_idx + load_idx * NUM_THREADS_PER_BLOCK) % VECTORIZED_BLOCK_TILE_SIZE_X * LEN_VECTOR;

        int B_row_idx = shared_mem_tile_idx * BLOCK_TILE_SIZE_K + B_shared_mem_tile_row_idx;
        int B_col_idx = blockIdx.x * BLOCK_TILE_SIZE_X + B_shared_mem_tile_col_idx;

        // vector for vectorized load
        int4 B_row_vector_vals{0, 0, 0, 0};
        if (B_row_idx < k && B_col_idx < n) {
            B_row_vector_vals = *reinterpret_cast<int4*>(&B[B_row_idx * n + B_col_idx]);
        }

        // for corner case where col_idx greater than n, zeros the invalid numbers (outside the boundary)
        if (B_col_idx + LEN_VECTOR > n) {
            int num_invalid_numbers = B_col_idx + LEN_VECTOR - n;
            // use a float pointer to handle each elem in int4
            float* B_row_vector_vals_ptr = reinterpret_cast<float*>(&B_row_vector_vals);
            for (int i = 0; i < num_invalid_numbers; i++) {
                B_row_vector_vals_ptr[LEN_VECTOR - 1 - i] = 0.0;
            }
        }

        // vectorized write for normal shared memory array
        if (B_shared_mem_tile_row_idx < BLOCK_TILE_SIZE_K && B_shared_mem_tile_col_idx < BLOCK_TILE_SIZE_X) {
            *reinterpret_cast<int4*>(&B_shared_mem_tile[B_shared_mem_tile_row_idx][B_shared_mem_tile_col_idx]) = B_row_vector_vals;
        }
    }
}

#endif // GEMM_KERNEL_CUH