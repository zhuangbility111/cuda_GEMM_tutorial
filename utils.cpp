#include "gemm.h"

void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void checkCublasError(cublasStatus_t status, const char *msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s: %d\n", msg, status);
        exit(EXIT_FAILURE);
    }
}

void check_result(float *ref, float *my_res, int M, int N, std::string version) {
    printf("ref[0] = %f, ref[1] = %f, ref[2] = %f\n", ref[0], ref[1*M], ref[2*M]);
    printf("my_res[0] = %f, my_res[1] = %f, my_res[2] = %f\n", my_res[0], my_res[1], my_res[2]);

    for (int i = 0; i < M * N; i++) {
        int row = i / N;
        int col = i % N;
        if (abs(ref[col * M + row] - my_res[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test %s passed!\n", version.c_str());
}

void run_perf_test_v1(float *A, float *B, float *C, int M, int N, int K, int warmup, int repeat, std::string version, gemm_func func) {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < warmup; i++) {
        func(d_A, d_B, d_C, M, N, K);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        func(d_A, d_B, d_C, M, N, K);
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