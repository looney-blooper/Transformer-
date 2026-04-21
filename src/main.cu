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

int get_argmax(Tensor* logits, int vocab_size) {
    int best_token = 0;
    float max_prob = -1e20f;
    for (int v = 0; v < vocab_size; v++) {
        if (logits->h_data[v] > max_prob) {
            max_prob = logits->h_data[v];
            best_token = v;
        }
    }
    return best_token;
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

   // ==============================================================================
    // PHASE 2: AUTOREGRESSIVE INFERENCE (O(1) KV-Cache)
    // ==============================================================================
    std::cout << "\n==================================================" << std::endl;
    std::cout << "Phase 2: Autoregressive Inference" << std::endl;
    std::cout << "==================================================\n" << std::endl;

    // 1. FLIP THE MASTER SWITCH: Turn on the VRAM state-preservation
    gpt.enable_kv_cache();

    // 2. The Initial Prompt State
    int current_token_id = 45; // "The"
    std::cout << "Prompt: [ " << current_token_id << " ] (\"The\")\nAI Generating: ";

    // 3. Allocate a tiny integer pointer on the GPU for generating ONE token at a time
    int* d_single_input_id;
    cudaMalloc(&d_single_input_id, 1 * sizeof(int));
    
    Tensor single_logits({1, 1, vocab_size});

    // 4. Generate the next 3 tokens
    for (int step = 0; step < 3; step++) {
        
        // Push ONLY the newest integer token to the GPU
        cudaMemcpy(d_single_input_id, &current_token_id, sizeof(int), cudaMemcpyHostToDevice);

        // Forward pass using the raw int pointer:
        // - Calculates Q, K, V for this ONE token.
        // - Appends K and V to the physical VRAM cache.
        // - Matrix Multiplies Q against the ENTIRE historical cache.
        gpt.forward(d_single_input_id, &single_logits);
        single_logits.to_host();

        // Decode the output 
        int next_token_id = get_argmax(&single_logits, vocab_size);
        
        std::cout << "[ " << next_token_id << " ] " << std::flush;

        // The AI's prediction becomes the input reality for the next time step
        current_token_id = next_token_id;
    }

    std::cout << "\n\nInference Complete." << std::endl;

    // ==============================================================================
    // CLEANUP
    // ==============================================================================
    cudaFree(d_input_ids);
    cudaFree(d_targets);
    cudaFree(d_single_input_id);
    gpt.disable_kv_cache();
    ops::destroy_cublas();

    return 0;
}