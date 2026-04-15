#include <iostream>
#include <vector>
#include "core/tensor.cuh"
#include "core/ops.cuh"
#include "model/gpt.cuh"

int main() {
    std::cout << ">>> IGNITING C++ TRANSFORMER ENGINE <<<" << std::endl;
    ops::init_cublas();

    // Hyperparameters for a tiny "Nano-GPT" testing model
    int batch_size = 1;
    int max_seq_len = 4;
    int vocab_size = 1000; // Small vocab for testing
    int d_model = 32;
    int num_heads = 4;
    int d_ff = 128;
    int num_layers = 2; // Stacking 2 full decoder blocks!

    std::cout << "\nAllocating Memory for Master GPT Architecture..." << std::endl;
    model::GPT gpt(vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, batch_size);

    // 1. Create raw input data (Integer Token IDs)
    int h_input_ids[4] = {45, 102, 9, 88}; 
    int* d_input_ids;
    
    // Allocate integer memory directly on the GPU
    cudaMalloc((void**)&d_input_ids, max_seq_len * sizeof(int));
    cudaMemcpy(d_input_ids, h_input_ids, max_seq_len * sizeof(int), cudaMemcpyHostToDevice);

    // 2. Create the output Logits Tensor
    // Shape: [batch_size, max_seq_len, vocab_size]
    Tensor Logits({batch_size, max_seq_len, vocab_size});

    std::cout << "Executing Full Forward Pass..." << std::endl;
    
    // THE MOMENT OF TRUTH
    gpt.forward(d_input_ids, &Logits);
    cudaDeviceSynchronize(); // Force CPU to wait for GPU

    std::cout << "Forward Pass Completed with ZERO Segmentation Faults!" << std::endl;
    std::cout << "\nOutput Logits Shape: [" 
              << Logits.shape[0] << ", " 
              << Logits.shape[1] << ", " 
              << Logits.shape[2] << "]" << std::endl;

    // Clean up our manually allocated integer array
    cudaFree(d_input_ids);
    ops::destroy_cublas();
    
    std::cout << "\nArchitecture complete. Ready for Backpropagation." << std::endl;
    return 0;
}