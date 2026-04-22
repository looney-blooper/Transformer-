#pragma once
#include <string>
#include <vector>
#include <unordered_map>

// A custom hash function so we can use std::pair as a key in unordered_map
struct PairHash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1); 
    }
};

namespace data {

    class BPETokenizer {
    public:
        int target_vocab_size;
        int current_vocab_size;

        // The master dictionaries
        std::unordered_map<std::string, int> vocab;
        std::unordered_map<int, std::string> inverse_vocab;

        // The ordered list of merge rules: (Token A, Token B) -> New Token C
        std::vector<std::pair<int, int>> merge_rules;

        BPETokenizer(int target_vocab_size);
        ~BPETokenizer();

        // Phase 1: The Trainer
        void train(const std::string& text);

        // Phase 2: The Encoder/Decoder
        std::vector<int> encode(const std::string& text);
        std::string decode(const std::vector<int>& tokens);

    private:
        // Helper function to find the most frequent adjacent pairs in an array of tokens
        std::pair<int, int> get_most_frequent_pair(const std::vector<int>& tokens);
        
        // Helper function to replace all instances of a pair with the new merged token
        std::vector<int> apply_merge(const std::vector<int>& tokens, std::pair<int, int> pair_to_merge, int new_token_id);
    };

}