#ifndef GEMM_H
#define GEMM_H

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>

using gemm_func = void (*)(float *, float *, float *, int, int, int);

void checkCudaError(cudaError_t err, const char *msg);
void checkCublasError(cublasStatus_t status, const char *msg);

void check_result(float *ref, float *gpu, int M, int N, std::string version);

void run_perf_test(float *A, float *B, float *C, int M, int N, int K, int warmup, int repeat, dim3 grid, dim3 block, std::string version, gemm_func func);

void gemm_cublas(float *A, float *B, float *C, int M, int N, int K);

__global__ void gemm_naive(float *A, float *B, float *C, int M, int N, int K);

#endif