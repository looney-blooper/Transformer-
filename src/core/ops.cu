#include "ops.cuh"
#include <iostream>

// --------------------------------------------------------
// ADD TENSORS KERNEL
// --------------------------------------------------------
__global__ void add_tensors_kernel(float* A, float* B, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        A[idx] += B[idx]; // In-place addition
    }
}

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
        float beta = 0.0f;

        // Make it dimension-agnostic (treats [1, 128, 64] identical to [128, 64])
        int A_cols = A->shape.back();
        int A_rows = A->size / A_cols;
        
        int B_cols = B->shape.back();
        //int B_rows = B->size / B_cols;

        int C_cols = C->shape.back();
        int C_rows = C->size / C_cols;

        // The Row-Major Trick
        int m = C_cols;
        int n = C_rows;
        int k = transA ? A_rows : A_cols;

        cublasOperation_t cuTransA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t cuTransB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

        int lda = transA ? n : k;
        int ldb = transB ? k : m;
        int ldc = m;

        cublasSgemm(handle, 
                    cuTransB, cuTransA, 
                    m, n, k, 
                    &alpha, 
                    B->d_data, ldb, 
                    A->d_data, lda, 
                    &beta, 
                    C->d_data, ldc);
    }


    void add_tensors(Tensor* A, Tensor* B) {
        if (A->size != B->size) {
            std::cerr << "Error: Tensor sizes must match for addition!" << std::endl;
            exit(EXIT_FAILURE);
        }

        int threads_per_block = 256;
        int blocks = (A->size + threads_per_block - 1) / threads_per_block;

        add_tensors_kernel<<<blocks, threads_per_block>>>(A->d_data, B->d_data, A->size);
    }

    void strided_batched_matmul(Tensor* A, Tensor* B, Tensor* C, int b, int h, int m, int n, int k, bool transA, bool transB) {
        float alpha = 1.0f;
        float beta = 0.0f;

        // Strides tell cuBLAS how far to jump to reach the next head
        long long int strideA = m * k;
        long long int strideB = k * n;
        long long int strideC = m * n;

        cublasSgemmStridedBatched(
            handle,
            transB ? CUBLAS_OP_T : CUBLAS_OP_N,
            transA ? CUBLAS_OP_T : CUBLAS_OP_N,
            n, m, k,
            &alpha,
            B->d_data, transB ? k : n, strideB,
            A->d_data, transA ? m : k, strideA,
            &beta,
            C->d_data, n, strideC,
            b * h // Total batches = Batch size * Number of heads
        );
    }
    void strided_batched_matmul_kvcache(
        Tensor* A, Tensor* B, Tensor* C, 
        int b, int h, int m, int n, int k, 
        bool transA, bool transB,
        long long int strideA, long long int strideB, long long int strideC) 
    {
        float alpha = 1.0f;
        float beta = 0.0f;

        cublasSgemmStridedBatched(
            handle,
            transB ? CUBLAS_OP_T : CUBLAS_OP_N,
            transA ? CUBLAS_OP_T : CUBLAS_OP_N,
            n, m, k,
            &alpha,
            B->d_data, transB ? k : n, strideB,
            A->d_data, transA ? m : k, strideA,
            &beta,
            C->d_data, n, strideC,
            b * h 
        );
    }
}