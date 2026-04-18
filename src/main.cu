#include <iostream>
#include <iomanip>
#include "core/tensor.cuh"
#include "core/ops.cuh"
#include "layers/attention.cuh"

// Helper to initialize weights safely so Softmax doesn't explode
void init_tensor(Tensor* t) {
    for(int i = 0; i < t->size; i++) {
        t->h_data[i] = ((float)(rand() % 100) / 100.0f) * 0.1f;
    }
    t->to_device();
}

int main() {
    std::cout << ">>> IGNITING MHA BACKPROP TEST <<<" << std::endl;
    ops::init_cublas();

    int batch_size = 1;
    int max_seq_len = 4;
    int d_model = 8;
    int num_heads = 2;

    std::cout << "Allocating Memory for Multi-Head Attention..." << std::endl;
    layers::MultiHeadAttention mha(d_model, num_heads, max_seq_len, batch_size);
    
    // Safely initialize the projection weights
    init_tensor(mha.W_q->W); init_tensor(mha.W_q->b);
    init_tensor(mha.W_k->W); init_tensor(mha.W_k->b);
    init_tensor(mha.W_v->W); init_tensor(mha.W_v->b);
    init_tensor(mha.W_o->W); init_tensor(mha.W_o->b);

    Tensor X({batch_size, max_seq_len, d_model});
    Tensor Y({batch_size, max_seq_len, d_model});
    Tensor dY({batch_size, max_seq_len, d_model}); // Simulated loss gradient
    Tensor dX({batch_size, max_seq_len, d_model}); // The final output gradient

    // Initialize inputs
    for(int i = 0; i < X.size; i++) {
        X.h_data[i] = 1.0f;
        dY.h_data[i] = 0.5f; // Pretend the loss sent a flat 0.5 gradient
    }
    X.to_device();
    dY.to_device();

    std::cout << "Executing Forward Pass (Building Q, K, V, P)..." << std::endl;
    mha.forward(&X, &Y);
    cudaDeviceSynchronize();

    std::cout << "Executing Backward Pass (Unraveling the Calculus)..." << std::endl;
    mha.backward(&dY, &dX);
    cudaDeviceSynchronize();

    std::cout << ">>> MHA BACKWARD PASS COMPLETED WITHOUT SEGFAULTS! <<<" << std::endl;

    ops::destroy_cublas();
    return 0;
}