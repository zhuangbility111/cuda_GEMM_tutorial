#include "gemm.h"


void run_perf_test(float *A, float *B, float *C, int M, int N, int K, int warmup, int repeat, dim3 grid, dim3 block, std::string version, gemm_func func) {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // warm up
    if (version == "cuBLAS") {
        for (int i = 0; i < warmup; i++)
            func(d_A, d_B, d_C, M, N, K);
    } else {
        for (int i = 0; i < warmup; i++)
            func<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }

    // performance test
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (version == "cuBLAS") {
        for (int i = 0; i < repeat; i++)
            func(d_A, d_B, d_C, M, N, K);
    } else {
        for (int i = 0; i < repeat; i++)
            func<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_time = ms / repeat;

    printf("Test %s: average time of %d runs %.6f ms, %.6f Tflops, percentage of peak %.6f\n", version.c_str(), repeat, avg_time, 2.0 * M * N * K / avg_time / 1e9, 2.0 * M * N * K / avg_time / 1e9 / 19.5 * 100);

    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


void gemm_cublas(float *d_A, float *d_B, float *d_C, int M, int N, int K) {
    // create cublas handle
    cublasHandle_t handle;
    checkCublasError(cublasCreate(&handle), "Failed to create cublas handle");

    // 调用 cublasSgemm
    float alpha = 1.0f, beta = 0.0f;
    checkCublasError(
        cublasSgemm(handle,
                    CUBLAS_OP_T, CUBLAS_OP_T, // 矩阵 A 和 B 的操作类型：不转置
                    M, N, K,                 // 矩阵 A 的维度 (MxK)，B 的维度 (KxN)，结果 C 的维度 (MxN)
                    &alpha,                  // alpha 系数
                    d_A, K,                  // 矩阵 A 和其主列间距
                    d_B, N,                  // 矩阵 B 和其主列间距
                    &beta,                   // beta 系数
                    d_C, N),                 // 矩阵 C 和其主列间距
        "Failed to call cublasSgemm");
    
}

// version 1: naive implementation
__global__ void gemm_naive(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) 
            sum += A[row * K + k] * B[k * N +col];
        C[row * N + col] = sum;
    }
}


// version 2: coalesced memory access
__global__ void gemm_coalesced(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// version 3: blocking (tiling) on shared memory
// block_size must equals blockDim.x equals blockDim.y
__global__ void gemm_shared_mem_blocking(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float s_A[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE * BLOCK_SIZE];

    int row_in_shared_mem = threadIdx.y;
    int col_in_shared_mem = threadIdx.x;

    float* A_ptr = A + blockIdx.y * BLOCK_SIZE * K;
    float* B_ptr = B + blockIdx.x * BLOCK_SIZE;
    float* C_ptr = C + blockIdx.y * BLOCK_SIZE * N + blockIdx.x * BLOCK_SIZE;

    float tmp = 0.0;
    for (int s_block_idx = 0; s_block_idx < K; s_block_idx += BLOCK_SIZE) {
        // load data from global memory to shared memory
        s_A[row_in_shared_mem * BLOCK_SIZE + col_in_shared_mem] = A_ptr[row_in_shared_mem * K + col_in_shared_mem];
        s_B[row_in_shared_mem * BLOCK_SIZE + col_in_shared_mem] = B_ptr[row_in_shared_mem * N + col_in_shared_mem];

        __syncthreads();

        A_ptr += BLOCK_SIZE;
        B_ptr += BLOCK_SIZE * N;

        // compute
        for (int k = 0; k < BLOCK_SIZE; k++) {
            tmp += s_A[row_in_shared_mem * BLOCK_SIZE + k] * s_B[k * BLOCK_SIZE + col_in_shared_mem];
        }

        __syncthreads();
    }
    
    C_ptr[row_in_shared_mem * BLOCK_SIZE + col_in_shared_mem] = tmp;
}