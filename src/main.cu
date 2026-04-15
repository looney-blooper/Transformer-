#include <iostream>
#include <iomanip>
#include "core/tensor.cuh"
#include "layers/modules.cuh"

int main() {
    int batch_size = 1;
    int seq_len = 2; // Testing 2 separate tokens
    int d_model = 4; // 4 dimensions per token

    std::cout << "Allocating Memory for LayerNorm Test..." << std::endl;
    layers::LayerNorm layernorm(d_model);
    
    // Create Input and Output Tensors
    Tensor X({batch_size, seq_len, d_model});
    Tensor Y({batch_size, seq_len, d_model});

    // Manually load specific values into the Host (CPU) memory
    // Token 1: [1.0, 2.0, 3.0, 4.0]
    X.h_data[0] = 1.0f; X.h_data[1] = 2.0f; X.h_data[2] = 3.0f; X.h_data[3] = 4.0f;
    
    // Token 2: [10.0, 10.0, 10.0, 10.0]
    X.h_data[4] = 10.0f; X.h_data[5] = 10.0f; X.h_data[6] = 10.0f; X.h_data[7] = 10.0f;

    // Push the manually created data to the GPU
    X.to_device(); 

    std::cout << "Running LayerNorm Forward Pass..." << std::endl;
    layernorm.forward(&X, &Y);
    cudaDeviceSynchronize();

    std::cout << "Pulling Normalized Data back to CPU...\n" << std::endl;
    Y.to_host();

    std::cout << "=== NORMALIZED OUTPUT ===" << std::endl;
    float* out = Y.h_data;
    
    for (int token = 0; token < seq_len; token++) {
        std::cout << "Token " << token + 1 << ": [ ";
        for (int dim = 0; dim < d_model; dim++) {
            int idx = token * d_model + dim;
            std::cout << std::fixed << std::setprecision(4) << out[idx] << "   ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "=========================\n" << std::endl;

    return 0;
}