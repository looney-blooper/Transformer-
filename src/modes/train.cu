#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>

#include "../layers/loss.cuh"
#include "../model/gpt.cuh"
#include "../data/tokenizer.h"
#include "../data/dataloader.h"
#include "../core/checkpoint.cuh"
#include "../core/optimizer.cuh"
#include "../config/config.h"


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

__global__ void scale_tensor_kernel(float* d_data, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_data[idx] *= scale;
    }
}

// Host wrapper to call the kernel cleanly
void scale_tensor_gpu(float* d_data, int size, float scale) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    scale_tensor_kernel<<<blocks, threads>>>(d_data, size, scale);
}


void run_train(int argc, char** argv) {
    std::cout << "\n==================================================" << std::endl;
    std::cout << ">>> C++ TRANSFORMER ENGINE: TRAINING MODE <<<" << std::endl;
    std::cout << "==================================================\n" << std::endl;

    
    // Hyperparameters
    config::CONFIG hyper_parameters;
    int target_vocab_size = hyper_parameters.vocab_size;
    int d_model = hyper_parameters.d_model;
    int num_heads = hyper_parameters.num_heads;
    int d_ff = hyper_parameters.d_ff;
    int num_layers = hyper_parameters.num_layers;
    int max_seq_len = hyper_parameters.max_seq_len;
    int batch_size = hyper_parameters.train_batch_size;
    int epochs = hyper_parameters.epochs;
    int save_every_n_steps = hyper_parameters.save_every_n_steps; // Save backup every 500 batches
    int gradient_accumulation_steps = hyper_parameters.gradient_accumulation_steps;

    model::GPT gpt(target_vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, batch_size);
    gpt.print_model_summary();

    layers::CrossEntropyLoss criterion;
    core::AdamW optimizer(gpt.parameters(), 0.001f, 0.9f, 0.999f, 0.01f);
    
    std::cout << "\n[ INITIALIZING RANDOM WEIGHTS ]\n" << std::endl;
    initialize_model_weights(gpt);

    data::BPETokenizer tokenizer(target_vocab_size);
    CheckpointManagerNS::checkpointManager checkpointer("checkpoints");
    
    int start_epoch = 1, start_step = 0;
    std::string text = read_file("input.txt");
    
    // 1. Recover State OR Start Fresh
    bool is_resuming = checkpointer.load_latest(gpt, optimizer, start_epoch, start_step);
    
    if (is_resuming) {
        std::cout << "--> Loading existing BPE Vocabulary..." << std::endl;
        tokenizer.load("vocab.bin");
    } else {
        std::cout << "--> Training new BPE Tokenizer..." << std::endl;
        tokenizer.train(text);
        tokenizer.save("vocab.bin");
    }

    // 2. Setup Dataloader
    std::vector<int> tokens = tokenizer.encode(text);
    data::DataLoader dataloader(tokens, batch_size, max_seq_len);

    if (is_resuming) {
        dataloader.fast_forward_to_step(start_step);
    }

    // 3. VRAM Allocation
    int *d_X, *d_Y;
    cudaMalloc(&d_X, batch_size * max_seq_len * sizeof(int));
    cudaMalloc(&d_Y, batch_size * max_seq_len * sizeof(int));
    Tensor Logits({batch_size, max_seq_len, target_vocab_size});
    Tensor dLogits({batch_size, max_seq_len, target_vocab_size});

    gpt.disable_kv_cache();

    std::cout << "\n==================================================" << std::endl;
    std::cout << ">>> PRE-FLIGHT CHECK: OVERFITTING SINGLE BATCH <<<" << std::endl;
    std::cout << "==================================================\n" << std::endl;

    // 1. Fetch exactly ONE batch
    if (!dataloader.get_next_batch(d_X, d_Y)) {
        std::cerr << "Failed to fetch test batch!" << std::endl;
        return;
    }

    // 2. Force the engine to memorize it
    int test_steps = 250;
    for (int step = 1; step <= test_steps; step++) {
        
        optimizer.zero_grad();
        
        gpt.forward(d_X, &Logits);
        float loss = criterion.forward_backward(&Logits, d_Y, &dLogits);
        
        gpt.backward(&dLogits);
        optimizer.step();
        cudaDeviceSynchronize();

        // Print every 10 steps so we can watch the math stabilize
        if (step % 10 == 0 || step == 1) {
            std::cout << "Step " << step << "/" << test_steps 
                      << " | Loss: " << std::fixed << std::setprecision(5) << loss << std::endl;
        }
    }

    std::cout << "\n[ PRE-FLIGHT COMPLETE. DO NOT USE THESE WEIGHTS. ]\n" << std::endl;
    
    // --> Do NOT save these weights! They are garbage and only know one sentence.
    
    cudaFree(d_X); 
    cudaFree(d_Y);
}