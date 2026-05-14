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

    std::cout << "[3] Encoding line-by-line to completely eliminate memory shifting..." << std::endl;

    std::string line;
    size_t total_tokens = 0;
    size_t lines_processed = 0;

    // Read strictly line-by-line. 'N' is never larger than ~100 characters!
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        // Add the newline character back (since getline strips it)
        line += "\n"; 
        
        // Encode this tiny string instantly
        std::vector<int> tokens = tokenizer.encode(line);
        
        // Blast the integers to disk
        out.write(reinterpret_cast<const char*>(tokens.data()), tokens.size() * sizeof(int));
        
        total_tokens += tokens.size();
        lines_processed++;

        // Print progress every 10,000 lines so you know it isn't frozen
        if (lines_processed % 10000 == 0) {
            std::cout << "   -> Processed " << lines_processed << " lines. Total Tokens: " << total_tokens << "\r" << std::flush;
        }
    }

    file.close();
    out.close();
    std::cout << "\n\n[ SUCCESS ] Pipeline Complete! Saved " << total_tokens << " tokens to dataset.bin." << std::endl;
}