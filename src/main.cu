#include <iostream>
#include <iomanip>
#include "core/tensor.cuh"
#include "layers/loss.cuh"

int main() {
    int batch_size = 1;
    int seq_len = 2; // Testing 2 tokens
    int vocab_size = 4; // A tiny 4-word vocabulary

    std::cout << "Allocating Memory for Loss Test..." << std::endl;
    layers::CrossEntropyLoss criterion;
    
    // Logits: [batch_size * seq_len, vocab_size] = [2, 4]
    Tensor logits({batch_size * seq_len, vocab_size});
    Tensor dLogits({batch_size * seq_len, vocab_size});

    // --- SETUP SCENARIO ---
    // Token 1 Logits: [4.0, 1.0, 0.5, 0.1] -> Model STRONGLY predicts word 0
    logits.h_data[0] = 4.0f; logits.h_data[1] = 1.0f; logits.h_data[2] = 0.5f; logits.h_data[3] = 0.1f;
    
    // Token 2 Logits: [0.1, 0.5, 3.0, 1.0] -> Model STRONGLY predicts word 2
    logits.h_data[4] = 0.1f; logits.h_data[5] = 0.5f; logits.h_data[6] = 3.0f; logits.h_data[7] = 1.0f;
    
    logits.to_device(); // Push to GPU

    // Define the actual correct answers (Targets)
    // Token 1 target: Word 0 (Model is RIGHT!)
    // Token 2 target: Word 1 (Model is WRONG! It guessed word 2)
    int h_targets[2] = {0, 1}; 
    int* d_targets;
    cudaMalloc((void**)&d_targets, 2 * sizeof(int));
    cudaMemcpy(d_targets, h_targets, 2 * sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "Calculating Forward Loss and Backward Gradients..." << std::endl;
    
    // Run the fused kernel
    float loss = criterion.forward_backward(&logits, d_targets, &dLogits);
    cudaDeviceSynchronize();

    std::cout << "\n=== LOSS SCALAR ===" << std::endl;
    std::cout << "Total Loss: " << loss << std::endl;

    std::cout << "\n=== BACKWARD GRADIENTS (dLogits) ===" << std::endl;
    dLogits.to_host(); // Pull gradients back to CPU
    
    for (int token = 0; token < seq_len; token++) {
        std::cout << "Token " << token + 1 << " dLogits: [ ";
        for (int i = 0; i < vocab_size; i++) {
            std::cout << std::fixed << std::setprecision(4) << dLogits.h_data[token * vocab_size + i] << "   ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "====================================\n" << std::endl;

    cudaFree(d_targets);
    return 0;
}