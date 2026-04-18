#include "loss.cuh"
#include <iostream>
#include <cmath>

// --------------------------------------------------------
// FUSED CROSS-ENTROPY FORWARD & BACKWARD KERNEL
// --------------------------------------------------------
__global__ void cross_entropy_kernel(float* logits, int* targets, float* dLogits, float* d_total_loss, int vocab_size, int total_tokens) {
    // Each thread handles exactly ONE token prediction in the sequence
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < total_tokens) {
        int offset = row * vocab_size;
        int target_idx = targets[row];

        // Step 1: Find Max Logit for numerical stability
        float max_val = -1e20f;
        for (int i = 0; i < vocab_size; i++) {
            if (logits[offset + i] > max_val) {
                max_val = logits[offset + i];
            }
        }

        // Step 2: Sum Exponentials
        float sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            sum_exp += expf(logits[offset + i] - max_val);
        }

        // Step 3: Compute Probabilities, gradients, and isolate the target probability
        float prob_target = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            float prob = expf(logits[offset + i] - max_val) / sum_exp;
            
            // The Gradient Trick: dLoss/dLogit = Prob - Target
            // We scale the gradient by the batch size so the learning rate stays stable
            float grad = prob; 
            if (i == target_idx) {
                prob_target = prob;
                grad -= 1.0f; // Subtract 1 for the correct target!
            }
            dLogits[offset + i] = grad / total_tokens;
        }

        // Step 4: Calculate final Loss and add it to the global counter atomically
        // We add a tiny epsilon (1e-7) to prevent log(0) which causes NaN
        float loss = -logf(prob_target + 1e-7f);
        
        // atomicAdd safely allows thousands of threads to add to a single memory address at once
        atomicAdd(d_total_loss, loss / total_tokens); 
    }
}

namespace layers {

    float CrossEntropyLoss::forward_backward(Tensor* logits, int* d_targets, Tensor* dLogits) {
        int vocab_size = logits->shape.back();
        
        // The rest of the elements make up the total tokens
        int total_tokens = logits->size / vocab_size;

        // Allocate a single float on the GPU to hold the accumulated loss
        float* d_total_loss;
        cudaMalloc((void**)&d_total_loss, sizeof(float));
        cudaMemset(d_total_loss, 0, sizeof(float)); // Set to 0.0

        int threads_per_block = 256;
        int blocks = (total_tokens + threads_per_block - 1) / threads_per_block;

        // Launch the kernel
        cross_entropy_kernel<<<blocks, threads_per_block>>>(
            logits->d_data, 
            d_targets, 
            dLogits->d_data, 
            d_total_loss, 
            vocab_size, 
            total_tokens
        );

        // Catch kernel errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Cross Entropy Kernel Failed: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }

        // Pull the final calculated loss scalar back to the CPU
        float h_total_loss = 0.0f;
        cudaMemcpy(&h_total_loss, d_total_loss, sizeof(float), cudaMemcpyDeviceToHost);

        // Free the temporary GPU float
        cudaFree(d_total_loss);

        return h_total_loss;
    }

}