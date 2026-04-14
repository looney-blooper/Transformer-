#include "ops.cuh"
#include <iostream>

namespace ops {
    cublasHandle_t handle;

    void init_cublas() {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "CUBLAS initialization failed!" << std::endl;
            exit(EXIT_FAILURE);
        }
        std::cout << "cuBLAS Context Initialized." << std::endl;
    }

    void destroy_cublas() {
        cublasDestroy(handle);
    }

    void matmul(Tensor* A, Tensor* B, Tensor* C, bool transA, bool transB) {
        float alpha = 1.0f;
        float beta = 0.0f; // 0.0 ensures we overwrite C completely

        // The Row-Major Trick: We swap A and B, and their dimensions
        int m = C->shape[1]; // Columns of C
        int n = C->shape[0]; // Rows of C
        int k = transA ? A->shape[0] : A->shape[1]; // The shared inner dimension

        cublasOperation_t cuTransA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t cuTransB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

        // Leading dimensions for memory strides
        int lda = transA ? n : k;
        int ldb = transB ? k : m;
        int ldc = m;

        // Execute optimized GEMM (General Matrix Multiply) on the GPU
        // Note: we pass B first, then A.
        cublasSgemm(handle, 
                    cuTransB, cuTransA, 
                    m, n, k, 
                    &alpha, 
                    B->d_data, ldb, 
                    A->d_data, lda, 
                    &beta, 
                    C->d_data, ldc);
    }
}