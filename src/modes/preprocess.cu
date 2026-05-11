#include <iostream>
#include <fstream>
#include <vector>
#include <string>
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

    // Chunking parameters (2MB chunks to bypass Big O complexity)
    const size_t CHUNK_SIZE = 2 * 1024 * 1024;
    std::string chunk;
    chunk.resize(CHUNK_SIZE);

    size_t total_tokens = 0;
    int chunk_count = 0;

    std::cout << "[3] Encoding in chunks to bypass O(N^2) complexity..." << std::endl;
    while (file) {
        file.read(&chunk[0], CHUNK_SIZE);
        size_t bytes_read = file.gcount();
        if (bytes_read == 0) break;

        // Trim to actual read size (critical for the final chunk)
        std::string actual_chunk = chunk.substr(0, bytes_read);

        // Encode just this chunk
        std::vector<int> tokens = tokenizer.encode(actual_chunk);

        // Write raw integer bytes directly to disk
        out.write(reinterpret_cast<const char*>(tokens.data()), tokens.size() * sizeof(int));

        total_tokens += tokens.size();
        chunk_count++;

        if (chunk_count % 5 == 0) {
            std::cout << "   -> Processed " << (chunk_count * 2) << " MB. Total Tokens: " << total_tokens << "\r" << std::flush;
        }
    }

    file.close();
    out.close();
    std::cout << "\n\n[ SUCCESS ] Pipeline Complete! Saved " << total_tokens << " tokens to dataset.bin." << std::endl;
}