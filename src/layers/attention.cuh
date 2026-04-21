#pragma once
#include "../core/tensor.cuh"
#include "modules.cuh"

namespace layers {
    class MultiHeadAttention {
    public:
        int d_model;
        int num_heads;
        int d_k;
        int max_seq_len;

        Linear* W_q;
        Linear* W_k;
        Linear* W_v;
        Linear* W_o;

        // 1. Interleaved Tensors (Token-Major)
        Tensor* Q;
        Tensor* K;
        Tensor* V;
        Tensor* Context;

        // 2. Transposed Tensors (Head-Major) -> NEW
        Tensor* Q_transposed;
        Tensor* K_transposed;
        Tensor* V_transposed;
        Tensor* Context_transposed;

        // 3. The Attention Matrix
        Tensor* Attention_Scores;

        // 4. Interleaved Gradients (Token-Major)
        Tensor* dQ;
        Tensor* dK;
        Tensor* dV;
        Tensor* dContext;
        Tensor* dAttention_Scores;

        // 5. Transposed Gradients (Head-Major) -> NEW
        Tensor* dQ_transposed;
        Tensor* dK_transposed;
        Tensor* dV_transposed;
        Tensor* dContext_transposed;

        // 6. Gradients for the input X
        Tensor* dX_q;
        Tensor* dX_k;
        Tensor* dX_v;

        MultiHeadAttention(int d_model, int num_heads, int max_seq_len, int batch_size);
        ~MultiHeadAttention();

        void forward(Tensor* X, Tensor* Y);
        void backward(Tensor* dY, Tensor* dX);


        // ==========================================
        // KV-CACHE MEMORY BUFFERS
        // ==========================================
        bool use_kv_cache;      // Toggle switch for Training vs Inference
        int current_cache_len;  // Tracks how many tokens are currently stored

        // Persistent pre-allocated memory for the historical sequence
        // Shape: [batch_size, num_heads, max_seq_len, d_k]
        Tensor* K_cache;
        Tensor* V_cache;

        void enable_kv_cache();
        void disable_kv_cache();
        void clear_kv_cache();

        Tensor* Inference_Scores;
    };
}