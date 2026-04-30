#include "../config/config.h"
#include "../model/gpt.cuh"

void run_get_model_summary(int argc, char** argv){

    config::CONFIG hyper_parameters;
    int target_vocab_size = hyper_parameters.vocab_size;
    int d_model = hyper_parameters.d_model;
    int num_heads = hyper_parameters.num_heads;
    int d_ff = hyper_parameters.d_ff;
    int num_layers = hyper_parameters.num_layers;
    int max_seq_len = hyper_parameters.max_seq_len;
    int batch_size = hyper_parameters.train_batch_size;

    model::GPT gpt(target_vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, batch_size);

    gpt.print_model_summary();

}