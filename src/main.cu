#include <iostream>
#include <iomanip> // For clean matrix formatting
#include "core/tensor.cuh"
#include "core/ops.cuh"
#include "layers/attention.cuh"

int main() {
    ops::init_cublas();

    // Use tiny dimensions so we can visually read the matrix
    int batch_size = 1;
    int max_seq_len = 4; // A simple 4-token sequence
    int d_model = 8;
    int num_heads = 2;

    std::cout << "Allocating Memory for 4x4 Test..." << std::endl;
    layers::MultiHeadAttention mha(d_model, num_heads, max_seq_len, batch_size);
    
    // Dummy input tensor
    Tensor X({batch_size, max_seq_len, d_model});

    // Fill X with random dummy values so the Softmax has actual varying numbers to calculate
    for(int i = 0; i < X.size; i++) {
        X.h_data[i] = (float)(rand() % 100) / 100.0f; 
    }
    X.to_device(); // Push the dummy data to the GPU

    std::cout << "Running Forward Pass (up to Softmax)..." << std::endl;
    
    // We pass nullptr for Y because we haven't written the final output projection yet
    mha.forward(&X, nullptr); 
    cudaDeviceSynchronize();

    std::cout << "Pulling Attention Matrix back to CPU..." << std::endl;
    // CRITICAL: Pull the GPU data back to Host memory
    mha.Attention_Scores->to_host();

    std::cout << "\n=== CAUSAL ATTENTION MATRIX (4x4) ===" << std::endl;
    float* scores = mha.Attention_Scores->h_data;
    
    // Print the matrix row by row
    for (int row = 0; row < max_seq_len; row++) {
        for (int col = 0; col < max_seq_len; col++) {
            // Flattened index calculation
            int idx = row * max_seq_len + col;
            std::cout << std::fixed << std::setprecision(4) << scores[idx] << "   ";
        }
        std::cout << std::endl; // New line for the next row
    }
    std::cout << "=====================================\n" << std::endl;

    ops::destroy_cublas();
    return 0;
}