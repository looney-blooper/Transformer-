#pragma once
#include "../core/tensor.cuh"
#include "../layers/attention.cuh"
#include "../layers/modules.cuh"

namespace model {
    class DecoderBlock {
    public:
        layers::LayerNorm* ln1;
        layers::MultiHeadAttention* mha;
        layers::LayerNorm* ln2;
        layers::FeedForward* ffn;
        
        //Intermidiate Caches
        Tensor* norm_cache;
        Tensor* attn_out;
        Tensor* ffn_out;

        //Gradient caches
        Tensor* d_norm_cache;
        Tensor* d_attn_out;
        Tensor* d_ffn_out;


        DecoderBlock(int d_model, int num_heads, int d_ff, int max_seq_len, int batch_size);

        ~DecoderBlock();

        void forward(Tensor* X);
        void backward(Tensor* dX);
    };
}