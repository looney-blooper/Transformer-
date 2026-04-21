#pragma once
#include <cublas_v2.h>
#include "tensor.cuh"

namespace ops {
    // Initialize and destroy the cuBLAS context
    void init_cublas();
    void destroy_cublas();

    // Matrix Multiplication: C = A * B
    void matmul(Tensor* A, Tensor* B, Tensor* C, bool transA = false, bool transB = false);

    // In-place Tensor Addition: A = A + B
    void add_tensors(Tensor* A, Tensor* B);

    void strided_batched_matmul(Tensor* A, Tensor* B, Tensor* C, int b, int h, int m, int n, int k, bool transA, bool transB);
}