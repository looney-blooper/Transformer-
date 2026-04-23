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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./gpt_engine [train | infer]" << std::endl;
        return 1;
    }
    std::string mode = argv[1];

    std::cout << "==================================================" << std::endl;
    std::cout << ">>> C++ TRANSFORMER ENGINE: " << (mode == "train" ? "TRAINING MODE" : "INFERENCE MODE") << " <<<" << std::endl;
    std::cout << "==================================================\n" << std::endl;

    ops::init_cublas();
    std::string text = read_file("input.txt"); 

    // Hyperparameters
    int target_vocab_size = 300; 
    int batch_size = 4;
    int max_seq_len = 128; 
    int d_model = 64;
    int num_heads = 4;
    int d_ff = 256;
    int num_layers = 2;

    // Data Pipeline
    data::BPETokenizer tokenizer(target_vocab_size);
    tokenizer.train(text);

    // Engine Initialization
    model::GPT gpt(target_vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, batch_size);
    gpt.print_model_summary();

    if (mode == "train") {
        std::cout << "\n[ INITIALIZING RANDOM WEIGHTS ]\n" << std::endl;
        initialize_model_weights(gpt);
        
        std::vector<int> tokens = tokenizer.encode(text);
        data::DataLoader dataloader(tokens, batch_size, max_seq_len);
        
        layers::CrossEntropyLoss criterion;
        core::AdamW optimizer(gpt.parameters(), 0.001f, 0.9f, 0.999f, 0.01f);

        int *d_X, *d_Y;
        cudaMalloc(&d_X, batch_size * max_seq_len * sizeof(int));
        cudaMalloc(&d_Y, batch_size * max_seq_len * sizeof(int));
        Tensor Logits({batch_size, max_seq_len, target_vocab_size});
        Tensor dLogits({batch_size, max_seq_len, target_vocab_size});

        gpt.disable_kv_cache();
        int epochs = 150;
        
        // Save the learned vocabulary immediately
        tokenizer.save("vocab.bin");

        // OPEN TELEMETRY FILE
        std::ofstream history_file("training_history.csv");
        history_file << "Epoch,Loss\n"; // CSV Header

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

            float avg_loss = epoch_loss / step;
            std::cout << "Epoch " << epoch << "/" << epochs << " | Avg Loss: " << std::fixed << std::setprecision(5) << avg_loss << std::endl;
            
            // LOG TELEMETRY
            history_file << epoch << "," << avg_loss << "\n";
        }

        // --- THE ARTIFACT EXTRACTION ---
        std::cout << "\n[ TRAINING COMPLETE. EXTRACTING BRAIN... ]\n" << std::endl;
        gpt.save_pretrained("gpt2_weights.bin");

        history_file.close(); // Close the log safely

        cudaFree(d_X); cudaFree(d_Y);

    } else if (mode == "infer") {
        // ... loading weights ...
        std::cout << "\n[ LOADING PRE-TRAINED ARTIFACTS ]\n" << std::endl;
        
        // Load BOTH the weights and the vocabulary. No input.txt required!
        gpt.load_pretrained("gpt2_weights.bin");
        tokenizer.load("vocab.bin");

        gpt.enable_kv_cache();

        // THE PATCH: Read the prompt from the command line, default to "The" if missing
        std::string seed_text = (argc >= 3) ? argv[2] : "The";
        
        std::vector<int> generated_tokens = tokenizer.encode(seed_text);
        int current_token = generated_tokens.back(); // Feed the last token to the GPU
        
        // Output a clean JSON-friendly string for the web server to read
        std::cout << seed_text;

        int* d_single_input;
        cudaMalloc(&d_single_input, 1 * sizeof(int));
        Tensor single_logits({1, 1, target_vocab_size});

        for (int i = 0; i < 100; i++) {
            cudaMemcpy(d_single_input, &current_token, sizeof(int), cudaMemcpyHostToDevice);
            
            gpt.forward(d_single_input, &single_logits);
            single_logits.to_host();
            
            int next_token = get_argmax(&single_logits, target_vocab_size);
            generated_tokens.push_back(next_token);
            
            std::vector<int> print_vec = {next_token};
            std::cout << tokenizer.decode(print_vec); // Print directly without extra formatting

            current_token = next_token;
        }
        std::cout << std::endl; // One clean newline at the end
        cudaFree(d_single_input);
    } else {
        std::cerr << "Invalid mode. Use 'train' or 'infer'." << std::endl;
    }

    ops::destroy_cublas();
    return 0;
}