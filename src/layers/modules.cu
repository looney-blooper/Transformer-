#include "modules.cuh"
#include "../core/ops.cuh"
#include <cmath>

// --------------------------------------------------------
// 1. THE CUSTOM CUDA KERNEL
// --------------------------------------------------------
// This function runs on the GPU. Every thread calculates its own global index
// and adds the correct bias value to the output matrix Y.
__global__ void add_bias_kernel(float* Y, float* b, int batch_size, int out_features) {
    // Calculate which thread this is
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;

    // Ensure we don't read out of bounds
    if (idx < total_elements) {
        // Find which column this thread is in to apply the correct bias
        int col = idx % out_features; 
        Y[idx] += b[col];
    }
}

namespace layers {

    // --------------------------------------------------------
    // 2. LINEAR LAYER CONSTRUCTOR
    // --------------------------------------------------------
    Linear::Linear(int in_feat, int out_feat) {
        in_features = in_feat;
        out_features = out_feat;

        // Allocate weights [in_features, out_features]
        W = new Tensor({in_features, out_features});
        
        // Allocate bias [1, out_features]
        b = new Tensor({1, out_features});
        
        X_cache = nullptr; // Initialize cache as null

        // TODO: In a real scenario, we must initialize W with Xavier/Kaiming 
        // initialization instead of leaving it as random memory junk.
    }

    Linear::~Linear() {
        delete W;
        delete b;
    }

    // --------------------------------------------------------
    // 3. FORWARD PASS
    // --------------------------------------------------------
    void Linear::forward(Tensor* X, Tensor* Y) {
        // Cache the input. We NEED this later to calculate gradients in the backward pass!
        X_cache = X;

        // Step A: Matrix Multiplication (Y = X * W)
        ops::matmul(X, W, Y);

        // Step B: Launch the custom CUDA kernel to add the bias (Y = Y + b)
        int batch_size = X->shape[0];
        int total_elements = batch_size * out_features;
        
        // CUDA configuration: 256 threads per block, calculate how many blocks we need
        int threads_per_block = 256;
        int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

        // Launch the kernel on the GPU
        add_bias_kernel<<<blocks, threads_per_block>>>(Y->d_data, b->d_data, batch_size, out_features);
        
        // Catch any kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Bias Kernel Failed: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}