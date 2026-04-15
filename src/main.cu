#include <iostream>
#include <iomanip>
#include "core/tensor.cuh"
#include "core/ops.cuh"
#include "layers/modules.cuh"

// Helper function to initialize memory safely for testing
void init_tensor(Tensor* t) {
    for(int i = 0; i < t->size; i++) {
        // Assign a small random float between 0.0 and 0.1
        t->h_data[i] = ((float)(rand() % 100) / 100.0f) * 0.1f; 
    }
    t->to_device();
}

int main() {
    ops::init_cublas();

    int batch_size = 1;
    int max_seq_len = 2; // 2 tokens
    int d_model = 4;     // Input/Output dimension
    int d_ff = 16;       // Hidden expansion dimension

    std::cout << "Allocating Memory for Feed-Forward Test..." << std::endl;
    layers::FeedForward ffn(d_model, d_ff, max_seq_len, batch_size);
    
    // Safely initialize the FFN weights
    init_tensor(ffn.w1->W);
    init_tensor(ffn.w1->b);
    init_tensor(ffn.w2->W);
    init_tensor(ffn.w2->b);

    Tensor X({batch_size, max_seq_len, d_model});
    Tensor Y({batch_size, max_seq_len, d_model});

    // Initialize the input tokens to 1.0
    for(int i = 0; i < X.size; i++) {
        X.h_data[i] = 1.0f;
    }
    X.to_device();

    std::cout << "Running FFN Forward Pass (Linear -> GELU -> Linear)..." << std::endl;
    ffn.forward(&X, &Y);
    cudaDeviceSynchronize();

    std::cout << "Pulling Data back to CPU...\n" << std::endl;
    Y.to_host();

    std::cout << "=== FFN OUTPUT ===" << std::endl;
    for (int token = 0; token < max_seq_len; token++) {
        std::cout << "Token " << token + 1 << ": [ ";
        for (int dim = 0; dim < d_model; dim++) {
            int idx = token * d_model + dim;
            std::cout << std::fixed << std::setprecision(4) << Y.h_data[idx] << "   ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "==================\n" << std::endl;

    ops::destroy_cublas();
    return 0;
}