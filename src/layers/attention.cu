#include "attention.cuh"
#include "../core/ops.cuh"
#include <iostream>
#include <cmath>

namespace layers {
    MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads, int max_seq_len, int batch_size){
        this->d_model = d_model;
        this->num_heads = num_heads;
        this->d_k = d_model / num_heads;

        W_q = new Linear(d_model, d_model);
        W_k = new Linear(d_model, d_model);
        W_v = new Linear(d_model, d_model);
        W_o = new Linear(d_model, d_model);


        std::vector<int> qkv_shape = {batch_size, d_model};
        Q = new Tensor(qkv_shape);
        K = new Tensor(qkv_shape);
        V = new Tensor(qkv_shape);

        std::vector<int> attn_shape = {batch_size, max_seq_len};
        Attention_Scores = new Tensor(attn_shape);
        Context = new Tensor(qkv_shape);
    }

    MultiHeadAttention::~MultiHeadAttention(){
        delete W_q;
        delete W_k;
        delete W_v;
        delete W_o;
        delete Q;
        delete K;
        delete V;
        delete Attention_Scores;
        delete Context;
    }

    void MultiHeadAttention::forward(Tensor* X, Tensor* Y){
        W_q->forward(X, Q);
        W_k->forward(X, K);
        W_v->forward(X, V);
    
    }
}