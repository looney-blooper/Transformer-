#pragma once
#include "../core/tensor.cuh"
#include "modules.cuh"


namespace layers {
    class MultiHeadAttention {
    public:
        int d_model;
        int num_heads;
        int d_k;

        Linear* W_q;
        Linear* W_k;
        Linear* W_v;
        Linear* W_o;


        Tensor* Q;
        Tensor* K;
        Tensor* V;
        Tensor* Attention_Scores;
        Tensor* Context;

        MultiHeadAttention(int d_model, int num_heads, int max_seq_len, int batch_size);
        ~MultiHeadAttention();



        void forward(Tensor* X, Tensor* Y);

    };
}

