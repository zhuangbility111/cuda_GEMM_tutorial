#ifndef GEMM_H
#define GEMM_H

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 32

using gemm_func = void (*)(float *, float *, float *, int, int, int);

void checkCudaError(cudaError_t err, const char *msg);
void checkCublasError(cublasStatus_t status, const char *msg);

void check_result(float *ref, float *gpu, int M, int N, std::string version);

void run_perf_test(float *A, float *B, float *C, int M, int N, int K, int warmup, int repeat, dim3 grid, dim3 block, std::string version, gemm_func func);
void run_perf_test_v1(float *A, float *B, float *C, int M, int N, int K, int warmup, int repeat, std::string version, gemm_func func);

void gemm_cublas(float *A, float *B, float *C, int M, int N, int K);

__global__ void gemm_naive(float *A, float *B, float *C, int M, int N, int K);

__global__ void gemm_coalesced(float *A, float *B, float *C, int M, int N, int K);

__global__ void gemm_shared_mem_blocking(float *A, float *B, float *C, int M, int N, int K);

void launch_gemm_v3_2d_block_tiling_1d_thread_tiling(float *A, float *B, float *C, int M, int N, int K);

void launch_gemm_v4_2d_block_tiling_2d_thread_tiling(float *A, float *B, float *C, int M, int N, int K);

void launch_gemm_v5_2d_block_tiling_2d_thread_tiling_vectorized_transposed(float *A, float *B, float *C, int M, int N, int K);

void launch_gemm_v6_2d_block_tiling_2d_warp_tiling_2d_thread_tiling_vectorized_transposed(float *A, float *B, float *C, int M, int N, int K);
#endif