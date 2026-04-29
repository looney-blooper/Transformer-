#include <iostream>
#include "../model/gpt.cuh"
#include "../config/config.h"

void run_compress(int argc, char** argv) {
    std::cout << "\n==================================================" << std::endl;
    std::cout << ">>> C++ TRANSFORMER ENGINE: INT8 QUANTIZATION <<<" << std::endl;
    std::cout << "==================================================\n" << std::endl;

    config::CONFIG hyper_parameters;
    int target_vocab_size = hyper_parameters.vocab_size;
    int d_model = hyper_parameters.d_model;
    int num_heads = hyper_parameters.num_heads;
    int d_ff = hyper_parameters.d_ff;
    int num_layers = hyper_parameters.num_layers;
    int max_seq_len = hyper_parameters.max_seq_len;
    int batch_size = hyper_parameters.train_batch_size;

    model::GPT gpt(target_vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, batch_size);

    std::cout << "\n[ INITIATING INT8 COMPRESSION ]\n" << std::endl;
    
    gpt.load_pretrained("gpt2_weights.bin");
    gpt.save_int8("gpt2_weights_int8.bin");
    
    std::cout << "Compression complete. Check your hard drive for the new file sizes." << std::endl;
}