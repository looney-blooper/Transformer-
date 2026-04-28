#include <iostream>
#include "dataloader.h"
#include <cuda_runtime.h>

namespace data {

    DataLoader::DataLoader(const std::vector<int>& tokens, int batch_size, int max_seq_len) {
        this->tokens = tokens;
        this->batch_size = batch_size;
        this->max_seq_len = max_seq_len;
        this->current_position = 0;
        
        // Calculate how many full batches we can process safely
        // We need (max_seq_len + 1) tokens per sequence to safely grab the shifted Y target
        int tokens_per_batch = batch_size * max_seq_len;
        this->total_batches = (tokens.size() - 1) / tokens_per_batch;
    }

    DataLoader::~DataLoader() {}

    bool DataLoader::get_next_batch(int* d_X, int* d_Y) {
        // Check if we have enough tokens left to form a full batch + target shift
        if (current_position + (batch_size * max_seq_len) + 1 > tokens.size()) {
            return false; // The Epoch is complete
        }

        // Temporary CPU buffers to build the chunks
        std::vector<int> h_X(batch_size * max_seq_len);
        std::vector<int> h_Y(batch_size * max_seq_len);

        for (int b = 0; b < batch_size; b++) {
            for (int s = 0; s < max_seq_len; s++) {
                int data_idx = current_position + (b * max_seq_len) + s;
                int flat_idx = b * max_seq_len + s;
                
                h_X[flat_idx] = tokens[data_idx];
                h_Y[flat_idx] = tokens[data_idx + 1]; // The target is always shifted 1 into the future
            }
        }

        // Blast the buffers straight into VRAM
        cudaMemcpy(d_X, h_X.data(), h_X.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Y, h_Y.data(), h_Y.size() * sizeof(int), cudaMemcpyHostToDevice);

        // Slide the window forward
        current_position += batch_size * max_seq_len;
        
        return true;
    }

    void DataLoader::reset() {
        current_position = 0;
    }


    void DataLoader::fast_forward_to_step(int step) {
        // Calculate exactly how many tokens were consumed to reach this step
        current_position = step * batch_size * max_seq_len;
        
        // Safety check to prevent segfaults if the step count is somehow higher than the dataset
        if (current_position >= tokens.size() - 1) {
            current_position = 0; 
        }
        std::cout << "[DATALOADER] Buffer fast-forwarded to token index " << current_position << std::endl;
    }

}