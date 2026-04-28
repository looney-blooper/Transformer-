#include <iostream>
#include "../model/gpt.cuh"

void run_compress() {
    std::cout << "\n==================================================" << std::endl;
    std::cout << ">>> C++ TRANSFORMER ENGINE: INT8 QUANTIZATION <<<" << std::endl;
    std::cout << "==================================================\n" << std::endl;

    // Ensure hyperparams match
    int target_vocab_size = 300;
    int d_model = 64;
    int num_heads = 4;
    int d_ff = 256;
    int num_layers = 2;
    int max_seq_len = 128;
    int batch_size = 1;

    model::GPT gpt(target_vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, batch_size);

    std::cout << "\n[ INITIATING INT8 COMPRESSION ]\n" << std::endl;
    
    gpt.load_pretrained("gpt2_weights.bin");
    gpt.save_int8("gpt2_weights_int8.bin");
    
    std::cout << "Compression complete. Check your hard drive for the new file sizes." << std::endl;
}