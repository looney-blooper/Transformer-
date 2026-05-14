#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include "../data/tokenizer.h"
#include "../config/config.h"

void run_preprocess() {
    std::cout << "\n==================================================" << std::endl;
    std::cout << ">>> DATA PIPELINE: BINARY ENCODING <<<" << std::endl;
    std::cout << "==================================================\n" << std::endl;

    config::CONFIG cfg;
    data::BPETokenizer tokenizer(cfg.vocab_size);

    std::cout << "[1] Loading Vocabulary..." << std::endl;
    tokenizer.load("vocab.bin");

    std::cout << "[2] Opening input.txt..." << std::endl;
    std::ifstream file("input.txt");
    if (!file.is_open()) {
        std::cerr << "CRITICAL: Could not find input.txt. Did you download it?" << std::endl;
        exit(1);
    }

    std::ofstream out("dataset.bin", std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "CRITICAL: Could not create dataset.bin" << std::endl;
        exit(1);
    }

    std::cout << "[3] Encoding via Line-Batches to defeat O(N^2) complexity..." << std::endl;

    std::string line;
    std::string batch = "";
    
    // 50 KB micro-batches (Fast enough for CPU, large enough for I/O efficiency)
    const size_t BATCH_LIMIT = 50 * 1024; 
    
    size_t total_tokens = 0;
    size_t bytes_processed = 0;
    int batch_count = 0;

    // Read line-by-line to prevent slicing words in half!
    while (std::getline(file, line)) {
        batch += line + "\n";
        
        // When the batch hits 50KB, encode it and clear it
        if (batch.size() >= BATCH_LIMIT) {
            std::vector<int> tokens = tokenizer.encode(batch);
            out.write(reinterpret_cast<const char*>(tokens.data()), tokens.size() * sizeof(int));
            
            total_tokens += tokens.size();
            bytes_processed += batch.size();
            batch_count++;
            batch.clear(); // Reset for the next batch

            // Print progress every 10 batches
            if (batch_count % 10 == 0) {
                float mb_processed = (float)bytes_processed / (1024.0f * 1024.0f);
                std::cout << "   -> Processed " << std::fixed << std::setprecision(2) 
                          << mb_processed << " MB. Total Tokens: " << total_tokens << "\r" << std::flush;
            }
        }
    }

    // Flush whatever is left in the final batch
    if (!batch.empty()) {
        std::vector<int> tokens = tokenizer.encode(batch);
        out.write(reinterpret_cast<const char*>(tokens.data()), tokens.size() * sizeof(int));
        total_tokens += tokens.size();
    }

    file.close();
    out.close();
    std::cout << "\n\n[ SUCCESS ] Pipeline Complete! Saved " << total_tokens << " tokens to dataset.bin." << std::endl;
}