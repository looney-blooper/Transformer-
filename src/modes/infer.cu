#include <iostream>
#include <string>
#include "../model/gpt.cuh"
#include "../data/tokenizer.h"


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


void run_infer(int argc, char** argv) {
    std::cout << "\n==================================================" << std::endl;
    std::cout << ">>> C++ TRANSFORMER ENGINE: INFERENCE MODE <<<" << std::endl;
    std::cout << "==================================================\n" << std::endl;

    // Grab the seed text safely
    std::string seed_text = (argc >= 3) ? argv[2] : "The";

    // Ensure hyperparams match training!
    int target_vocab_size = 300;
    int d_model = 64;
    int num_heads = 4;
    int d_ff = 256;
    int num_layers = 2;
    int max_seq_len = 128;
    int batch_size = 1;

    model::GPT gpt(target_vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, batch_size);
    data::BPETokenizer tokenizer(target_vocab_size);

    std::cout << "\n[ LOADING INT8 COMPRESSED ARTIFACTS ]\n" << std::endl;
    gpt.load_int8("gpt2_weights_int8.bin"); 
    tokenizer.load("vocab.bin");

    gpt.enable_kv_cache();

    std::cout << "Starting generation with prompt: '" << seed_text << "'\n" << std::endl;    
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
}