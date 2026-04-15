#include <iostream>
#include <iomanip>
#include "core/tensor.cuh"
#include "core/ops.cuh"
#include "model/transformer.cuh"

int main() {
    std::cout << "Initializing cuBLAS..." << std::endl;
    ops::init_cublas();

    // Model Hyperparameters for local testing
    int batch_size = 1;
    int max_seq_len = 4;
    int d_model = 8;
    int num_heads = 2;
    int d_ff = 32;

    std::cout << "Allocating Memory for Full Decoder Block..." << std::endl;
    model::DecoderBlock block(d_model, num_heads, d_ff, max_seq_len, batch_size);

    // Create Input Tensor
    Tensor X({batch_size, max_seq_len, d_model});

    // Initialize the input tokens to exactly 1.0 so we can track the changes
    for(int i = 0; i < X.size; i++) {
        X.h_data[i] = (float)(rand() % 100) / 100.0f;
    }
    X.to_device();

    std::cout << "Running Full Decoder Block Forward Pass..." << std::endl;
    
    // Remember, X is modified IN-PLACE because of the residual connections!
    block.forward(&X);
    cudaDeviceSynchronize();

    std::cout << "Pulling Data back to CPU...\n" << std::endl;
    X.to_host();

    std::cout << "=== DECODER BLOCK OUTPUT ===" << std::endl;
    for (int token = 0; token < max_seq_len; token++) {
        std::cout << "Token " << token + 1 << ": [ ";
        for (int dim = 0; dim < d_model; dim++) {
            int idx = token * d_model + dim;
            std::cout << std::fixed << std::setprecision(4) << X.h_data[idx] << "   ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "============================\n" << std::endl;

    ops::destroy_cublas();
    return 0;
}