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

    // 4. The Master Training Loop
    for (int epoch = start_epoch; epoch <= epochs; epoch++) {

        // If resuming, pick up where we left off. Otherwise, start at 0.
        int step = (epoch == start_epoch) ? start_step : 0; 

        float epoch_loss = 0.0f;
        int batch_count = 0;

        while (dataloader.get_next_batch(d_X, d_Y)) {
            optimizer.zero_grad();
            // 1. Forward Pass
            gpt.forward(d_X, &Logits);

            float loss = criterion.forward_backward(&Logits, d_Y, &dLogits);
            // 2. Calculate the raw loss for telemetry
            float raw_loss = criterion.forward_backward(&Logits, d_Y, &dLogits);

            gpt.backward(&dLogits);
            optimizer.step();
            cudaDeviceSynchronize();
            // 3. SCALE THE GRADIENTS (CRITICAL)
            // If you don't scale dLogits down by 32, accumulating 32 gradients will cause an Infinity/NaN explosion!
            // Note: You will need to implement a simple CUDA kernel or CPU loop to multiply dLogits.d_data by (1.0f / cfg.gradient_accumulation_steps) here.
            float scale_factor = 1.0f / gradient_accumulation_steps;
            scale_tensor_gpu(dLogits.d_data, dLogits.size, scale_factor); 

            epoch_loss += loss;
            // 4. Backward Pass (Accumulates gradients into memory without updating weights)
            gpt.backward(&dLogits);
            
            epoch_loss += raw_loss;
            step++;
            batch_count++;

            // 5. THE ACCUMULATION TRIGGER
            if (step % gradient_accumulation_steps == 0) {
                // We have now collected 32 sequences worth of gradients. 
                // Now we finally step the optimizer and clear the buffer!
                optimizer.step();
                optimizer.zero_grad();
                cudaDeviceSynchronize();
            }

            // --> TRIGGER THE BACKUP CHECKPOINT
            if (step % save_every_n_steps == 0) {
                checkpointer.save_checkpoint(gpt, optimizer, epoch, step, loss, optimizer.lr);
                checkpointer.save_checkpoint(gpt, optimizer, epoch, step, raw_loss, optimizer.lr);
            }
        }

        // Epoch Complete
        float avg_loss = epoch_loss / batch_count;
        std::cout << "Epoch " << epoch << "/" << epochs << " | Avg Loss: " << std::fixed << std::setprecision(5) << avg_loss << std::endl;

        // Let the checkpointer handle telemetry safely
        checkpointer.log_telemetry(epoch, step, avg_loss, optimizer.lr);

        // --> RESET THE DATALOADER FOR THE NEXT EPOCH
        dataloader.reset(); 
    }

    // 5. Final Artifact Extraction
    std::cout << "\n[ TRAINING COMPLETE. EXTRACTING BRAIN... ]\n" << std::endl;
    gpt.save_pretrained("gpt2_weights.bin");

    cudaFree(d_X); 
    cudaFree(d_Y);
}