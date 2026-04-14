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

// --------------------------------------------------------
// 2. THE BACKWARD BIAS KERNEL
// --------------------------------------------------------
// One thread per output feature. Each thread loops down the batch column
// and sums the gradients to calculate exactly how the bias should update.
__global__ void backward_bias_kernel(float* dY, float* db, int batch_size, int out_features) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < out_features) {
        float sum = 0.0f;
        for (int row = 0; row < batch_size; row++) {
            // Flattened 2D array indexing: [row][col] -> row * width + col
            sum += dY[row * out_features + col];
        }
        db[col] = sum; // Write to the gradient memory
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

    // --------------------------------------------------------
    // 4. BACKWARD PASS (The Chain Rule)
    // --------------------------------------------------------
    void Linear::backward(Tensor* dY, Tensor* dX) {
        if (X_cache == nullptr) {
            std::cerr << "Error: Cannot run backward pass before forward pass!" << std::endl;
            exit(EXIT_FAILURE);
        }

        int batch_size = X_cache->shape[0];

        // 1. Calculate Gradient for Weights (dW = X^T * dY)
        // We temporarily swap the data pointer with the grad pointer to route the 
        // cuBLAS output directly into the gradient memory space.
        float* temp_W_data = W->d_data;
        W->d_data = W->d_grad; 
        
        // transA = true means we transpose X_cache
        ops::matmul(X_cache, dY, W, true, false); 
        
        // Restore the original data pointer
        W->d_data = temp_W_data;

        // 2. Calculate Gradient for Input (dX = dY * W^T)
        // This is the gradient passed backward to the previous layer.
        // transB = true means we transpose W
        if (dX != nullptr) {
            ops::matmul(dY, W, dX, false, true);
        }

        // 3. Calculate Gradient for Bias (db = sum(dY, axis=0))
        int threads_per_block = 256;
        int blocks = (out_features + threads_per_block - 1) / threads_per_block;
        
        backward_bias_kernel<<<blocks, threads_per_block>>>(dY->d_data, b->d_grad, batch_size, out_features);
        
        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Backward Bias Kernel Failed: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}