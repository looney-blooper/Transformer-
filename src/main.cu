#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "data/tokenizer.h"
#include "data/dataloader.h"

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << ">>> PHASE 1: CPU BPE TOKENIZATION" << std::endl;
    std::cout << "==================================================\n" << std::endl;

    // 1. Train a small custom vocabulary
    // Standard ASCII is 256. Let's learn 24 custom merges.
    data::BPETokenizer tokenizer(280); 
    
    std::string text = "to be or not to be, that is the question. "
                       "whether 'tis nobler in the mind to suffer "
                       "the slings and arrows of outrageous fortune, "
                       "or to take arms against a sea of troubles";
                       
    tokenizer.train(text);
    
    // 2. Encode the string into integers
    std::vector<int> tokens = tokenizer.encode(text);
    
    std::cout << "\nOriginal Text Length : " << text.length() << " characters." << std::endl;
    std::cout << "Encoded Array Length : " << tokens.size() << " tokens." << std::endl;
    std::cout << "Compression Ratio    : " << (float)text.length() / tokens.size() << "x\n" << std::endl;

    std::cout << "==================================================" << std::endl;
    std::cout << ">>> PHASE 2: GPU DATALOADER BATCHING" << std::endl;
    std::cout << "==================================================\n" << std::endl;

    // 3. Configure the DataLoader
    int batch_size = 2;
    int max_seq_len = 4;
    int elements_per_batch = batch_size * max_seq_len;
    
    data::DataLoader dataloader(tokens, batch_size, max_seq_len);
    
    std::cout << "Total batches available in this dataset: " << dataloader.total_batches << "\n\n";

    // 4. Allocate GPU Pointers (The Destination)
    int *d_X, *d_Y;
    cudaMalloc(&d_X, elements_per_batch * sizeof(int));
    cudaMalloc(&d_Y, elements_per_batch * sizeof(int));

    // 5. Allocate CPU Pointers (For verifying the GPU data)
    int *h_X = new int[elements_per_batch];
    int *h_Y = new int[elements_per_batch];

    // 6. The Training Loop Simulation
    int step = 1;
    while (dataloader.get_next_batch(d_X, d_Y)) {
        
        // Pull the data back from the GPU to verify it arrived safely
        cudaMemcpy(h_X, d_X, elements_per_batch * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Y, d_Y, elements_per_batch * sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << "--- Batch " << step << " ---" << std::endl;
        
        for (int b = 0; b < batch_size; b++) {
            std::cout << "  Row " << b << " | X: [ ";
            for (int s = 0; s < max_seq_len; s++) std::cout << h_X[b * max_seq_len + s] << " ";
            
            std::cout << "]  -->  Y: [ ";
            for (int s = 0; s < max_seq_len; s++) std::cout << h_Y[b * max_seq_len + s] << " ";
            std::cout << "]" << std::endl;
        }
        std::cout << std::endl;
        step++;
    }

    // Cleanup
    cudaFree(d_X);
    cudaFree(d_Y);
    delete[] h_X;
    delete[] h_Y;

    return 0;
}