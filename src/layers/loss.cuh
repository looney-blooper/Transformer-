#pragma once
#include"../core/tensor.cuh"


namespace layers {
    class CrossEntropyLoss{
    public:
        CrossEntropyLoss() = default;
        ~CrossEntropyLoss() = default;

        // Logits: [batch_size * seq_len, vocab_size]
        // Targets: [batch_size * seq_len] array of integer IDs
        // dLogits: Output tensor to hold the gradients for backprop
        // Returns: The scalar loss value (float)
        float forward_backward(Tensor* Logits, int* d_targets, Tensor* dLogits);
    };
}