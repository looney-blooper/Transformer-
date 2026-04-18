#include <iostream>
#include <vector>
#include "core/tensor.cuh"
#include "core/ops.cuh"
#include "model/gpt.cuh"
#include "layers/loss.cuh"

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << ">>> IGNITING FULL C++ TRANSFORMER TRAINING STEP <<<" << std::endl;
    std::cout << "==================================================\n" << std::endl;
    
    ops::init_cublas();

    // Model Architecture Hyperparameters
    int batch_size = 1;
    int max_seq_len = 4;
    int vocab_size = 1000;
    int d_model = 32;
    int num_heads = 4;
    int d_ff = 128;
    int num_layers = 2; // Stacking multiple blocks!

    std::cout << "[1/4] Allocating Master GPT Architecture..." << std::endl;
    model::GPT gpt(vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, batch_size);
    layers::CrossEntropyLoss criterion;

    // --- SETUP SCENARIO ---
    // We pass 4 integer tokens into the model
    int h_input_ids[4] = {45, 102, 9, 88}; 
    int* d_input_ids;
    cudaMalloc((void**)&d_input_ids, max_seq_len * sizeof(int));
    cudaMemcpy(d_input_ids, h_input_ids, max_seq_len * sizeof(int), cudaMemcpyHostToDevice);

    // We tell the model what the NEXT words were supposed to be
    int h_targets[4] = {102, 9, 88, 500}; 
    int* d_targets;
    cudaMalloc((void**)&d_targets, max_seq_len * sizeof(int));
    cudaMemcpy(d_targets, h_targets, max_seq_len * sizeof(int), cudaMemcpyHostToDevice);

    Tensor Logits({batch_size, max_seq_len, vocab_size});
    Tensor dLogits({batch_size, max_seq_len, vocab_size});

    // --- EXECUTE TRAINING STEP ---
    std::cout << "[2/4] Executing Deep Forward Pass..." << std::endl;
    gpt.forward(d_input_ids, &Logits);
    cudaDeviceSynchronize();

    std::cout << "[3/4] Calculating Cross-Entropy Loss..." << std::endl;
    float loss = criterion.forward_backward(&Logits, d_targets, &dLogits);
    cudaDeviceSynchronize();
    
    std::cout << "      -> Current Model Loss: " << loss << std::endl;

    std::cout << "[4/4] Executing Deep Backward Pass (Chain Rule)..." << std::endl;
    gpt.backward(&dLogits);
    cudaDeviceSynchronize();

    std::cout << "\n==================================================" << std::endl;
    std::cout << ">>> END-TO-END TRAINING STEP EXECUTED PERFECTLY <<<" << std::endl;
    std::cout << "==================================================" << std::endl;

    // Cleanup
    cudaFree(d_input_ids);
    cudaFree(d_targets);
    ops::destroy_cublas();
    
    return 0;
}