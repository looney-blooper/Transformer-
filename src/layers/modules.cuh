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
        

        LayerNorm(int d_model, float eps = 1e-5f);

        ~LayerNorm();

        void forward(Tensor* X, Tensor* Y);

        void Backward(Tensor* dY, Tensor* dX);
    };


    class GELU {
    public:
        GELU() = default;
        ~GELU() = default;

        void forward(Tensor* X, Tensor* Y);
    };


    class FeedForward {
    public:
        Linear* w1;
        Linear* w2;
        GELU* activation;
        Tensor* hidden_cache;

        FeedForward(int d_model, int d_ff, int max_seq_len, int batch_size);

        ~FeedForward();

        void forward(Tensor* X, Tensor* Y);

    };
}
