#include <iostream>
#include "core/tensor.cuh"
#include "core/ops.cuh"
#include "layers/modules.cuh"

int main() {
    ops::init_cublas();

    int batch_size = 2;
    int d_model = 64;   // Input features
    int d_ff = 128;     // Output features

    std::cout << "Allocating memory..." << std::endl;
    Tensor X({batch_size, d_model});
    Tensor Y({batch_size, d_ff});
    Tensor dY({batch_size, d_ff}); // Simulated loss gradient from the future
    Tensor dX({batch_size, d_model}); // Where we will store the gradient for X

    layers::Linear linear(d_model, d_ff);

    std::cout << "Executing Forward Pass (Y = XW + b)..." << std::endl;
    linear.forward(&X, &Y);
    cudaDeviceSynchronize(); 

    std::cout << "Executing Backward Pass (dW, db, dX)..." << std::endl;
    linear.backward(&dY, &dX);
    cudaDeviceSynchronize();

    std::cout << "Backpropagation completed without VRAM faults!" << std::endl;

    ops::destroy_cublas();
    return 0;
}