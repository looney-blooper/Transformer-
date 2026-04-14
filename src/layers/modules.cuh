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


        //lets add backprop next
    };
}
