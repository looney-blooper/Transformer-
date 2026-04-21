#include <iostream>
#include <iomanip>
#include "core/tensor.cuh"
#include "core/ops.cuh"
#include "model/gpt.cuh"
#include "layers/loss.cuh"
#include "core/optimizer.cuh"

// 1. Initializer (Fixed for LayerNorm Stability!)
void initialize_model_weights(model::GPT& gpt) {
    // Step A: Randomize all weights (Linear, Embeddings, etc.)
    for (Tensor* p : gpt.parameters()) {
        for(int i = 0; i < p->size; i++) {
            p->h_data[i] = ((float)(rand() % 1000) / 1000.0f - 0.5f) * 0.1f;
        }
        p->to_device(); 
    }

    // Step B: WAKE UP THE NETWORK!
    // Overwrite the LayerNorm Gammas to 1.0 and Betas to 0.0
    for (auto block : gpt.blocks) {
        for(int i = 0; i < gpt.d_model; i++) {
            block->ln1->gamma->h_data[i] = 1.0f;
            block->ln1->beta->h_data[i]  = 0.0f;
            block->ln2->gamma->h_data[i] = 1.0f;
            block->ln2->beta->h_data[i]  = 0.0f;
        }
        // Push the fixes to the GPU
        block->ln1->gamma->to_device(); block->ln1->beta->to_device();
        block->ln2->gamma->to_device(); block->ln2->beta->to_device();
    }

    // Fix the final output norm!
    for(int i = 0; i < gpt.d_model; i++) {
        gpt.final_ln->gamma->h_data[i] = 1.0f;
        gpt.final_ln->beta->h_data[i]  = 0.0f;
    }
    gpt.final_ln->gamma->to_device();
    gpt.final_ln->beta->to_device();
}

// 2. Sampler
int greedy_argmax(Tensor* logits, int token_index, int vocab_size) {
    logits->to_host(); 
    int offset = token_index * vocab_size; 
    float max_val = -1e20f;
    int best_id = 0;
    for (int i = 0; i < vocab_size; i++) {
        if (logits->h_data[offset + i] > max_val) {
            max_val = logits->h_data[offset + i];
            best_id = i;
        }
    }
    return best_id;
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << ">>> IGNITING C++ TRANSFORMER ENGINE <<<" << std::endl;
    std::cout << "==================================================\n" << std::endl;
    
    ops::init_cublas();

    int batch_size = 1;
    int max_seq_len = 4;
    int vocab_size = 1000;
    int d_model = 32;
    int num_heads = 4;
    int d_ff = 128;
    int num_layers = 2; 

    model::GPT gpt(vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, batch_size);
    initialize_model_weights(gpt);
    layers::CrossEntropyLoss criterion;
    core::AdamW optimizer(gpt.parameters(), 0.003f, 0.9f, 0.999f, 0.0f);

    initialize_model_weights(gpt);

    std::cout << "\n=== DEBUG: POSITIONAL ENCODING VRAM PROBE ===" << std::endl;
    // 1. Pull the PE matrix back from the GPU to the CPU
    gpt.pos_emb->pe_matrix->to_host(); 
    
    // 2. Print Position 0, Dimension 0 (Should be sin(0) = 0.0)
    std::cout << "PE [Pos 0, Dim 0]: " << gpt.pos_emb->pe_matrix->h_data[0] << std::endl;
    
    // 3. Print Position 1, Dimension 0 (Should be sin(1) = ~0.841)
    // Offset is: (position * d_model) + dimension
    std::cout << "PE [Pos 1, Dim 0]: " << gpt.pos_emb->pe_matrix->h_data[1 * d_model + 0] << std::endl;

    // 4. Print Position 1, Dimension 1 (Should be cos(1) = ~0.540)
    std::cout << "PE [Pos 1, Dim 1]: " << gpt.pos_emb->pe_matrix->h_data[1 * d_model + 1] << std::endl;
    std::cout << "===============================================\n" << std::endl;

    // --- PHASE 1: TRAINING ---
    // Target Sentence: "The quick brown fox jumps"
    // Sequence mappings:
    // 45 ("The") -> 102 ("quick")
    // 102 ("quick") -> 9 ("brown")
    // 9 ("brown") -> 88 ("fox")
    // 88 ("fox") -> 500 ("jumps")
    
    int h_input_ids[4] = {45, 102, 9, 88}; 
    int h_targets[4]   = {102, 9, 88, 500}; 
    
    int* d_input_ids;
    int* d_targets;
    cudaMalloc((void**)&d_input_ids, max_seq_len * sizeof(int));
    cudaMalloc((void**)&d_targets, max_seq_len * sizeof(int));
    cudaMemcpy(d_input_ids, h_input_ids, max_seq_len * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets, max_seq_len * sizeof(int), cudaMemcpyHostToDevice);

    Tensor Logits({batch_size, max_seq_len, vocab_size});
    Tensor dLogits({batch_size, max_seq_len, vocab_size});

    std::cout << "Phase 1: Training the Network (300 Steps)...\n" << std::endl;
    for (int step = 1; step <= 300; step++) {
        optimizer.zero_grad();
        gpt.forward(d_input_ids, &Logits);
        float loss = criterion.forward_backward(&Logits, d_targets, &dLogits);
        gpt.backward(&dLogits);
        optimizer.step();
        cudaDeviceSynchronize(); 
        
        if (step % 5 == 0) {
            std::cout << "Step " << std::setw(2) << step << " | Loss: " << std::fixed << std::setprecision(5) << loss << std::endl;
        }
    }

    // --- PHASE 2: INFERENCE (AUTOREGRESSIVE GENERATION) ---
    std::cout << "\n==================================================" << std::endl;
    std::cout << "Phase 2: Autoregressive Inference" << std::endl;
    std::cout << "==================================================" << std::endl;

    // We give the model exactly ONE word to start with: "The" (45). 
    // We pad the rest of the array with 0s.
    int context[4] = {45, 0, 0, 0}; 
    int current_len = 1;

    std::cout << "\nPrompt: [ 45 ] (\"The\")" << std::endl;
    std::cout << "AI Generating: ";

    // We will ask the model to predict the next 3 words
    for(int step = 0; step < 3; step++) {
        // 1. Push current context to GPU
        cudaMemcpy(d_input_ids, context, max_seq_len * sizeof(int), cudaMemcpyHostToDevice);

        // 2. Run Forward Pass
        gpt.forward(d_input_ids, &Logits);

        // 3. Extract the prediction for the LAST word we just fed it
        // If current_len is 1, we want the prediction from index 0.
        int predicted_id = greedy_argmax(&Logits, current_len - 1, vocab_size);

        // Print the predicted word ID
        std::cout << "[ " << predicted_id << " ] ";

        // 4. Append the prediction to our context for the next loop!
        context[current_len] = predicted_id;
        current_len++;
    }
    std::cout << "\n\n>>> ENGINE COMPLETED SUCCESSFULLY <<<" << std::endl;

    cudaFree(d_input_ids);
    cudaFree(d_targets);
    ops::destroy_cublas();
    
    return 0;
}