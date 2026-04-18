#include <iostream>
#include "core/tensor.cuh"
#include "core/ops.cuh"
#include "layers/modules.cuh"

// Helper to initialize weights safely
void init_tensor(Tensor* t) {
    for(int i = 0; i < t->size; i++) {
        t->h_data[i] = ((float)(rand() % 100) / 100.0f) * 0.1f;
    }
    t->to_device();
}

int main() {
    ops::init_cublas();

    int batch_size = 1;
    int max_seq_len = 2;
    int d_model = 4;
    int d_ff = 16;

    std::cout << "Allocating Memory for FFN Backprop Test..." << std::endl;
    layers::FeedForward ffn(d_model, d_ff, max_seq_len, batch_size);
    
    init_tensor(ffn.w1->W); init_tensor(ffn.w1->b);
    init_tensor(ffn.w2->W); init_tensor(ffn.w2->b);

    Tensor X({batch_size, max_seq_len, d_model});
    Tensor Y({batch_size, max_seq_len, d_model});
    Tensor dY({batch_size, max_seq_len, d_model}); // Simulated loss gradient
    Tensor dX({batch_size, max_seq_len, d_model});

    // Initialize inputs
    for(int i = 0; i < X.size; i++) {
        X.h_data[i] = 1.0f;
        dY.h_data[i] = 0.5f; // Pretend the loss function sent a flat 0.5 gradient back
    }
    X.to_device();
    dY.to_device();

    std::cout << "Executing FFN Forward Pass..." << std::endl;
    ffn.forward(&X, &Y);
    cudaDeviceSynchronize();

    std::cout << "Executing FFN Backward Pass..." << std::endl;
    ffn.backward(&dY, &dX);
    cudaDeviceSynchronize();

    std::cout << ">>> FFN BACKWARD PASS COMPLETED WITHOUT SEGFAULTS! <<<" << std::endl;

    ops::destroy_cublas();
    return 0;
}