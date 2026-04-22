#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>

#include "core/tensor.cuh"
#include "core/ops.cuh"
#include "model/gpt.cuh"
#include "layers/loss.cuh"
#include "core/optimizer.cuh"
#include "data/tokenizer.h"
#include "data/dataloader.h"

// --- Helper: Read a text file ---
std::string read_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "CRITICAL ERROR: Could not open " << filepath << std::endl;
        exit(1);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// --- Helper: Decoding Sampler ---
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

// --- Helper: Initialization Fix ---
void initialize_model_weights(model::GPT& gpt) {
    for (Tensor* p : gpt.parameters()) {
        for(int i = 0; i < p->size; i++) p->h_data[i] = ((float)(rand() % 1000) / 1000.0f - 0.5f) * 0.1f;
        p->to_device(); 
    }
    for (auto block : gpt.blocks) {
        for(int i = 0; i < gpt.d_model; i++) {
            block->ln1->gamma->h_data[i] = 1.0f; block->ln1->beta->h_data[i] = 0.0f;
            block->ln2->gamma->h_data[i] = 1.0f; block->ln2->beta->h_data[i] = 0.0f;
        }
        block->ln1->gamma->to_device(); block->ln1->beta->to_device();
        block->ln2->gamma->to_device(); block->ln2->beta->to_device();
    }
    for(int i = 0; i < gpt.d_model; i++) {
        gpt.final_ln->gamma->h_data[i] = 1.0f; gpt.final_ln->beta->h_data[i]  = 0.0f;
    }
    gpt.final_ln->gamma->to_device(); gpt.final_ln->beta->to_device();
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << ">>> IGNITING END-TO-END C++ TRANSFORMER <<<" << std::endl;
    std::cout << "==================================================\n" << std::endl;

    // 1. Setup Environment
    ops::init_cublas();
    std::string text = read_file("src/input.txt"); 

    // 2. Hyperparameters
    int target_vocab_size = 300; // 256 ASCII + 44 learned BPE merges
    int batch_size = 4;
    int max_seq_len = 128;
    int d_model = 64;
    int num_heads = 4;
    int d_ff = 256;
    int num_layers = 2;
    int epochs = 100;
    float learning_rate = 0.001f;

    // 3. Data Pipeline
    std::cout << "--> Training BPE Tokenizer..." << std::endl;
    data::BPETokenizer tokenizer(target_vocab_size);
    tokenizer.train(text);
    std::vector<int> tokens = tokenizer.encode(text);
    
    data::DataLoader dataloader(tokens, batch_size, max_seq_len);
    std::cout << "Dataset encoded into " << tokens.size() << " tokens. Total Batches per Epoch: " << dataloader.total_batches << "\n" << std::endl;

    // 4. Engine Initialization
    model::GPT gpt(target_vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, batch_size);
    initialize_model_weights(gpt);
    layers::CrossEntropyLoss criterion;
    core::AdamW optimizer(gpt.parameters(), learning_rate, 0.9f, 0.999f, 0.01f);

    // Allocate GPU Data Pointers
    int *d_X, *d_Y;
    cudaMalloc(&d_X, batch_size * max_seq_len * sizeof(int));
    cudaMalloc(&d_Y, batch_size * max_seq_len * sizeof(int));
    
    Tensor Logits({batch_size, max_seq_len, target_vocab_size});
    Tensor dLogits({batch_size, max_seq_len, target_vocab_size});

    // ==============================================================================
    // PHASE 1: TRAINING LOOP
    // ==============================================================================
    std::cout << "==================================================" << std::endl;
    std::cout << "Phase 1: Training the Network" << std::endl;
    std::cout << "==================================================" << std::endl;
    gpt.disable_kv_cache();

    for (int epoch = 1; epoch <= epochs; epoch++) {
        dataloader.reset();
        int step = 0;
        float epoch_loss = 0.0f;

        while (dataloader.get_next_batch(d_X, d_Y)) {
            optimizer.zero_grad();
            gpt.forward(d_X, &Logits);
            float loss = criterion.forward_backward(&Logits, d_Y, &dLogits);
            gpt.backward(&dLogits);
            optimizer.step();
            cudaDeviceSynchronize();

            epoch_loss += loss;
            step++;
        }
        std::cout << "Epoch " << epoch << "/" << epochs << " | Avg Loss: " << std::fixed << std::setprecision(5) << (epoch_loss / step) << std::endl;
    }

    // ==============================================================================
    // PHASE 2: INFERENCE (O(1) KV-Cache Generation)
    // ==============================================================================
    std::cout << "\n==================================================" << std::endl;
    std::cout << "Phase 2: Autoregressive Inference" << std::endl;
    std::cout << "==================================================\n" << std::endl;

    gpt.enable_kv_cache();

    // Start the model off with a single seed character (e.g., 'T')
    std::string seed_text = "T";
    std::vector<int> generated_tokens = tokenizer.encode(seed_text);
    int current_token = generated_tokens[0];
    
    std::cout << seed_text << std::flush;

    int* d_single_input;
    cudaMalloc(&d_single_input, 1 * sizeof(int));
    Tensor single_logits({1, 1, target_vocab_size});

    // Generate 100 new tokens
    for (int i = 0; i < 100; i++) {
        cudaMemcpy(d_single_input, &current_token, sizeof(int), cudaMemcpyHostToDevice);
        
        gpt.forward(d_single_input, &single_logits);
        single_logits.to_host();
        
        int next_token = get_argmax(&single_logits, target_vocab_size);
        generated_tokens.push_back(next_token);
        
        // Decode and print the single token as it streams
        std::vector<int> print_vec = {next_token};
        std::cout << tokenizer.decode(print_vec) << std::flush;

        current_token = next_token;
    }
    std::cout << "\n\nInference Complete." << std::endl;

    // Cleanup
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_single_input);
    gpt.disable_kv_cache();
    ops::destroy_cublas();

    return 0;
}