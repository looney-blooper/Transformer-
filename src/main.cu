#include <iostream>
#include "core/tensor.cuh"
#include "core/ops.cuh"
#include "layers/attention.cuh"

int main() {
    std::cout << "Initializing cuBLAS..." << std::endl;
    ops::init_cublas();

    // The hyperparameters we locked in
    int batch_size = 2;
    int max_seq_len = 128;
    int d_model = 64;
    int num_heads = 2;

    std::cout << "\nAllocating Attention Memory on RTX 3050..." << std::endl;
    
    // Create the Attention block
    layers::MultiHeadAttention mha(d_model, num_heads, max_seq_len, batch_size);

    std::cout << "Multi-Head Attention instantiated successfully!" << std::endl;
    std::cout << "- Q, K, V buffers allocated." << std::endl;
    std::cout << "- Attention Scores buffer allocated: " << mha.Attention_Scores->size << " elements." << std::endl;
    
    // Create a dummy input tensor and test a partial forward pass
    Tensor X({batch_size, d_model});
    Tensor Y({batch_size, d_model});

    std::cout << "\nTesting Q, K, V Projections (Forward Pass Step 1)..." << std::endl;
    // We only call the projection layers since the Softmax isn't written yet
    mha.W_q->forward(&X, mha.Q);
    mha.W_k->forward(&X, mha.K);
    mha.W_v->forward(&X, mha.V);

    cudaDeviceSynchronize();
    std::cout << "Projections completed without VRAM faults!" << std::endl;

    std::cout << "\nCleaning up VRAM..." << std::endl;
    ops::destroy_cublas();
    // Destructors will automatically fire here and free everything.
    
    std::cout << "Environment clean. Ready for Softmax." << std::endl;
    return 0;
}