#pragma once
#include "../core/tensor.cuh"
#include <random>

namespace layers {
    class Linear {
    public:
        int in_features;
        int out_features;

        Tensor* W;    //weight matrix
        Tensor* b;    //bais matrix
        Tensor* X_cache;     //cahce used for back prop

        Linear (int in_feat, int out_feat);

        ~Linear();

        //forward pass : Y = WX + b
        void forward(Tensor* X, Tensor* Y);

        //backward pass : 
        void backward(Tensor* dY, Tensor* dX);
    };

    class LayerNorm {
    public:
        int d_model;
        float eps;

        Tensor* gamma;
        Tensor* beta;
        
        // Cache for backprop
        Tensor* X_cache; 
        Tensor* normalized_cache;

        LayerNorm(int d_model, float eps = 1e-5f);

        ~LayerNorm();

        void forward(Tensor* X, Tensor* Y);

        void backward(Tensor* dY, Tensor* dX);
    };


    class GELU {
    public:
        // Cache for backprop
        Tensor* X_cache;

        GELU() = default;
        ~GELU() = default;

        void forward(Tensor* X, Tensor* Y);
        void backward(Tensor* dY, Tensor* dX);
    };


    class FeedForward {
    public:
        Linear* w1;
        Linear* w2;
        GELU* activation;

        Tensor* hidden_cache;    // Holds pre-GELU values
        Tensor* activated_cache; // NEW: Holds post-GELU values

        FeedForward(int d_model, int d_ff, int max_seq_len, int batch_size);

        ~FeedForward();

        void forward(Tensor* X, Tensor* Y);
        void backward(Tensor* dY, Tensor* dX);
    };


    class Embedding {
    public:
        int vocab_size;
        int d_model;

        Tensor* weight; // Positional Encoding matrix // Shape: [vocab_size, d_model]
        Tensor* dWeight;

        int* ids_cache; // Remembers the input tokens for backprop

        Embedding(int vocab_size, int d_model);

        ~Embedding();

        // X is an array of integer Token IDs. Y is the output float Tensor.
        void forward(int* X_ids, Tensor* Y, int total_tokens);
        void backward(Tensor* dY, int total_tokens);
    };


    class PositionalEncoding {
    public:
        int max_seq_len;
        int d_model;
        Tensor* pe_matrix; // Fixed matrix of shape [max_seq_len, d_model]

        PositionalEncoding(int max_seq_len, int d_model);
        ~PositionalEncoding();

        // Adds the positional encodings IN-PLACE to the embedded tensor X
        void forward(Tensor* X);
    };
}
