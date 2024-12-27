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
