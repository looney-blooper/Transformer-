#pragma once
#include <vector>
#include "../core/tensor.cuh"
#include "../layers/modules.cuh"
#include "../layers/attention.cuh"
#include "transformer.cuh"


namespace model {
    class GPT {
    public:
        int vocab_size;
        int d_model;
        int num_heads;
        int num_layers;
        int max_seq_len;
        int batch_size;

        //input layer
        layers::Embedding* tok_emb;
        layers::PositionalEncoding* pos_emb;

        //transofmer backbone
        std::vector<DecoderBlock*> blocks;

        //output layers
        layers::LayerNorm* final_ln;
        layers::Linear* lm_head;

        // The main sequence memory buffer that flows through the network
        Tensor* hidden_state;

        GPT(int vocab_size, int d_model, int num_heads, int d_ff, int num_layers, int max_seq_len, int batch_size);
        ~GPT();


        // X_ids is an array of integer tokens on the GPU.
        // Logits is the final un-normalized probability output.
        void forward(int* d_input_ids, Tensor* logits);
    };
}