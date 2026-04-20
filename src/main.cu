#include <iostream>
#include <iomanip>
#include "core/tensor.cuh"
#include "core/ops.cuh"
#include "model/gpt.cuh"
#include "layers/loss.cuh"
#include "core/optimizer.cuh"

// Helper function to seed the neural network
void initialize_model_weights(model::GPT& gpt) {
    for (Tensor* p : gpt.parameters()) {
        for(int i = 0; i < p->size; i++) {
            // Generate a random float between -0.05 and 0.05
            p->h_data[i] = ((float)(rand() % 1000) / 1000.0f - 0.5f) * 0.1f;
        }
        // Push the initialized data to the GPU
        p->to_device(); 
    }
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << ">>> IGNITING C++ TRANSFORMER TRAINING LOOP <<<" << std::endl;
    std::cout << "==================================================\n" << std::endl;
    
    ops::init_cublas();

    // Model Architecture
    int batch_size = 1;
    int max_seq_len = 4;
    int vocab_size = 1000;
    int d_model = 32;
    int num_heads = 4;
    int d_ff = 128;
    int num_layers = 2; 

    std::cout << "Allocating GPT Architecture and AdamW Optimizer..." << std::endl;
    model::GPT gpt(vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, batch_size);
    layers::CrossEntropyLoss criterion;
    
    initialize_model_weights(gpt);
    // We use a high learning rate (0.01) so we can see the loss drop instantly in just 20 steps
    core::AdamW optimizer(gpt.parameters(), 0.01f);

    // --- SETUP SCENARIO ---
    // Input: "The" (45) -> "quick" (102) -> "brown" (9) -> "fox" (88)
    int h_input_ids[4] = {45, 102, 9, 88}; 
    int* d_input_ids;
    cudaMalloc((void**)&d_input_ids, max_seq_len * sizeof(int));
    cudaMemcpy(d_input_ids, h_input_ids, max_seq_len * sizeof(int), cudaMemcpyHostToDevice);

    // Target: "quick" (102) -> "brown" (9) -> "fox" (88) -> "jumps" (500)
    int h_targets[4] = {102, 9, 88, 500}; 
    int* d_targets;
    cudaMalloc((void**)&d_targets, max_seq_len * sizeof(int));
    cudaMemcpy(d_targets, h_targets, max_seq_len * sizeof(int), cudaMemcpyHostToDevice);

    Tensor Logits({batch_size, max_seq_len, vocab_size});
    Tensor dLogits({batch_size, max_seq_len, vocab_size});

    // ... (after allocating the optimizer) ...

    std::cout << "\n=== DEBUG DIAGNOSTIC ===" << std::endl;
    
    // Test 1: Did we actually gather the parameters?
    std::cout << "Total parameter tensors gathered: " << gpt.parameters().size() << std::endl;

    // Test 2: Are the weights actually random, or are they zero?
    gpt.tok_emb->weight->to_host(); // Pull from GPU to check
    std::cout << "First embedding weight: " << gpt.tok_emb->weight->h_data[0] << std::endl;
    
    std::cout << "========================\n" << std::endl;

    std::cout << "\nStarting 20-Step Training Loop...\n" << std::endl;
    
    // --- THE HEARTBEAT OF MACHINE LEARNING ---
    for (int step = 1; step <= 20; step++) {
        
        // 1. Clear old gradients from the previous step
        optimizer.zero_grad();

        // 2. Forward Pass: Generate predictions
        gpt.forward(d_input_ids, &Logits);

        // 3. Calculate Loss and generate dLogits
        float loss = criterion.forward_backward(&Logits, d_targets, &dLogits);

        // 4. Backward Pass: Route the error back through the network
        gpt.backward(&dLogits);

        // 5. Optimizer Step: Physically alter the weights
        optimizer.step();

        // Force CPU to wait for GPU before printing
        cudaDeviceSynchronize(); 

        std::cout << "Step " << std::setw(2) << step << " | Loss: " << std::fixed << std::setprecision(5) << loss << std::endl;
    }

    std::cout << "\n==================================================" << std::endl;
    std::cout << ">>> TRAINING COMPLETE. THE NETWORK HAS LEARNED. <<<" << std::endl;
    std::cout << "==================================================" << std::endl;

    cudaFree(d_input_ids);
    cudaFree(d_targets);
    ops::destroy_cublas();
    
    return 0;
}