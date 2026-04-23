#include "tokenizer.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdexcept>

namespace data {

    // ==========================================
    // 1. INITIALIZATION
    // ==========================================
    BPETokenizer::BPETokenizer(int target_vocab_size) {
        this->target_vocab_size = target_vocab_size;
        this->current_vocab_size = 256; // We start with the 256 standard bytes

        // Initialize the base vocabulary (0-255 map directly to their byte values)
        for (int i = 0; i < 256; i++) {
            std::string ch = "";
            ch += (char)i;
            vocab[ch] = i;
            inverse_vocab[i] = ch;
        }
    }

    BPETokenizer::~BPETokenizer() {}

    // ==========================================
    // 2. HELPER: Find the most frequent pair
    // ==========================================
    std::pair<int, int> BPETokenizer::get_most_frequent_pair(const std::vector<int>& tokens) {
        std::unordered_map<std::pair<int, int>, int, PairHash> counts;
        
        // Slide a window of size 2 across the token array and count
        for (size_t i = 0; i < tokens.size() - 1; i++) {
            std::pair<int, int> pair = {tokens[i], tokens[i+1]};
            counts[pair]++;
        }

        // Find the pair with the absolute highest frequency
        int max_count = -1;
        std::pair<int, int> best_pair = {-1, -1};
        
        for (const auto& kv : counts) {
            if (kv.second > max_count) {
                max_count = kv.second;
                best_pair = kv.first;
            }
        }
        
        return best_pair;
    }

    // ==========================================
    // 3. HELPER: Apply the merge to the array
    // ==========================================
    std::vector<int> BPETokenizer::apply_merge(const std::vector<int>& tokens, std::pair<int, int> pair_to_merge, int new_token_id) {
        std::vector<int> new_tokens;
        new_tokens.reserve(tokens.size()); // Pre-allocate to prevent CPU reallocation lag
        
        size_t i = 0;
        while (i < tokens.size()) {
            // If we find the exact pair, merge it!
            if (i < tokens.size() - 1 && tokens[i] == pair_to_merge.first && tokens[i+1] == pair_to_merge.second) {
                new_tokens.push_back(new_token_id);
                i += 2; // Skip the next token since we absorbed it
            } else {
                // Otherwise, leave it alone
                new_tokens.push_back(tokens[i]);
                i += 1;
            }
        }
        
        return new_tokens;
    }

    // ==========================================
    // 4. THE MASTER TRAINING LOOP
    // ==========================================
    void BPETokenizer::train(const std::string& text) {
        std::cout << "Starting BPE Training. Target Vocab Size: " << target_vocab_size << std::endl;

        // 1. Convert raw text into initial byte tokens (0-255)
        std::vector<int> tokens;
        tokens.reserve(text.length());
        for (unsigned char c : text) {
            tokens.push_back((int)c); 
        }

        // 2. Iteratively compress the sequence
        while (current_vocab_size < target_vocab_size) {
            
            // Step A: Count and find the best target
            std::pair<int, int> best_pair = get_most_frequent_pair(tokens);
            
            // Safety trigger: If no pairs repeat, we cannot compress further.
            if (best_pair.first == -1) break; 

            // Step B: Create the new token ID
            int new_token_id = current_vocab_size;

            // Step C: Update the Master Dictionaries
            std::string token_str = inverse_vocab[best_pair.first] + inverse_vocab[best_pair.second];
            vocab[token_str] = new_token_id;
            inverse_vocab[new_token_id] = token_str;
            
            // Step D: Record the rule (so we can encode new text later)
            merge_rules.push_back(best_pair);

            // Step E: Physically rewrite the token array, compressing it
            tokens = apply_merge(tokens, best_pair, new_token_id);

            current_vocab_size++;

            // Print progress
            if (current_vocab_size % 100 == 0 || current_vocab_size == target_vocab_size) {
                std::cout << "Vocab Size: " << std::setw(5) << current_vocab_size 
                          << " | Merged: (" << best_pair.first << ", " << best_pair.second 
                          << ") -> '" << token_str << "'" << std::endl;
            }
        }
        
        std::cout << "BPE Training Complete. Final sequence length: " << tokens.size() << std::endl;
    }

    // ==========================================
    // 5. THE ENCODER (String -> Tokens)
    // ==========================================
    std::vector<int> BPETokenizer::encode(const std::string& text) {
        // 1. Initial byte encoding (0-255)
        std::vector<int> tokens;
        tokens.reserve(text.length());
        for (unsigned char c : text) {
            tokens.push_back((int)c);
        }

        // 2. Iteratively merge based on learned rules
        while (tokens.size() >= 2) {
            int best_rank = 9999999;
            std::pair<int, int> best_pair = {-1, -1};
            int new_token_id = -1;

            for (size_t i = 0; i < tokens.size() - 1; i++) {
                std::pair<int, int> current_pair = {tokens[i], tokens[i+1]};
                
                // ==========================================
                // THE FIX: Raw C++ Index Lookup
                // ==========================================
                int rank = -1;
                for (size_t r = 0; r < merge_rules.size(); r++) {
                    // Manually check both halves of the pair
                    if (merge_rules[r].first == current_pair.first && merge_rules[r].second == current_pair.second) {
                        rank = r; // The index is the chronologic rank!
                        break; 
                    }
                }
                
                // If we found a valid rule, see if it has the highest priority (lowest rank)
                if (rank != -1) {
                    if (rank < best_rank) {
                        best_rank = rank;
                        best_pair = current_pair;
                        
                        // Rule 0 becomes Token 256. Rule 1 becomes Token 257.
                        new_token_id = 256 + rank; 
                    }
                }
            }

            // If no pairs in the current string match any of our learned rules, we are done
            if (best_pair.first == -1) {
                break;
            }

            // Apply the best merge and repeat the scan
            tokens = apply_merge(tokens, best_pair, new_token_id);
        }

        return tokens;
    }

    // ==========================================
    // 6. THE DECODER (Tokens -> String)
    // ==========================================
    std::string BPETokenizer::decode(const std::vector<int>& tokens) {
        std::string text = "";
        
        for (int token : tokens) {
            if (inverse_vocab.find(token) != inverse_vocab.end()) {
                text += inverse_vocab[token];
            } else {
                // Safety fallback if the GPU hallucinates an untrained token
                text += " [UNK] "; 
            }
        }
        
        return text;
    }

    void BPETokenizer::save(const std::string& filepath) {
        std::ofstream out(filepath, std::ios::binary);
        if (!out.is_open()) {
            throw std::runtime_error("Could not open vocab file for saving: " + filepath);
        }

        // 1. Write the target and current vocab sizes to maintain state
        out.write(reinterpret_cast<const char*>(&target_vocab_size), sizeof(int));
        out.write(reinterpret_cast<const char*>(&current_vocab_size), sizeof(int));

        // 2. Write the total number of merge rules
        int num_merges = merge_rules.size();
        out.write(reinterpret_cast<const char*>(&num_merges), sizeof(int));

        // 3. Write the sequential rules (Left Token, Right Token)
        for (const auto& pair : merge_rules) {
            int left = pair.first;
            int right = pair.second;
            
            out.write(reinterpret_cast<const char*>(&left), sizeof(int));
            out.write(reinterpret_cast<const char*>(&right), sizeof(int));
        }
        
        out.close();
        std::cout << "Vocabulary saved to " << filepath << " (" << num_merges << " merges)" << std::endl;
    }

    void BPETokenizer::load(const std::string& filepath) {
        std::ifstream in(filepath, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("Could not open vocab file for loading: " + filepath);
        }

        // 1. Clear any existing state
        merge_rules.clear();
        vocab.clear();
        inverse_vocab.clear();
        
        // 2. Re-initialize the base ASCII dictionary (IDs 0-255)
        for (int i = 0; i < 256; i++) {
            std::string ch = std::string(1, (char)i);
            vocab[ch] = i;
            inverse_vocab[i] = ch;
        }

        // 3. Read the state variables
        in.read(reinterpret_cast<char*>(&target_vocab_size), sizeof(int));
        in.read(reinterpret_cast<char*>(&current_vocab_size), sizeof(int));

        int num_merges = 0;
        in.read(reinterpret_cast<char*>(&num_merges), sizeof(int));

        // 4. Read the rules and reconstruct the master dictionaries
        for (int i = 0; i < num_merges; i++) {
            int left, right;
            in.read(reinterpret_cast<char*>(&left), sizeof(int));
            in.read(reinterpret_cast<char*>(&right), sizeof(int));
            
            // Add the rule back to the ordered list
            merge_rules.push_back({left, right});

            // Reconstruct the New Token ID (Sequential, starting from 256)
            int new_token_id = 256 + i;
            
            // Reconstruct the physical string by looking up the left and right components
            std::string new_token_str = inverse_vocab[left] + inverse_vocab[right];
            
            // Re-map the dictionaries
            vocab[new_token_str] = new_token_id;
            inverse_vocab[new_token_id] = new_token_str;
        }
        
        in.close();
        std::cout << "Vocabulary loaded from " << filepath << " (Total Vocab Size: " << current_vocab_size << ")" << std::endl;
    }
}