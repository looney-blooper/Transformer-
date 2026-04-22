#pragma once
#include <vector>

namespace data {
    class DataLoader {
    public:
        std::vector<int> tokens;
        int batch_size;
        int max_seq_len;
        int current_position;
        int total_batches;
        
        DataLoader(const std::vector<int>& tokens, int batch_size, int max_seq_len);
        ~DataLoader();

        // Fetches the next chunk and streams it directly to the GPU pointers
        bool get_next_batch(int* d_X, int* d_Y);

        // Rewinds the dataset back to the beginning for the next Epoch
        void reset();
    };

}