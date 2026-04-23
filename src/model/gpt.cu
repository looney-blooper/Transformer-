#include "gpt.cuh"
#include <iostream>
#include <fstream>
#include <iomanip>

namespace model {
    GPT::GPT(int vocab_size, int d_model, int num_heads, int d_ff, int num_layers, int max_seq_len, int batch_size){
        this->vocab_size = vocab_size;
        this->d_model = d_model;
        this->num_heads = num_heads;
        this->num_layers = num_layers;
        this->batch_size = batch_size;
        this->max_seq_len = max_seq_len;
        this->d_ff = d_ff;

        //1. Input Archeitecture
        tok_emb = new layers::Embedding(vocab_size, d_model);
        pos_emb = new layers::PositionalEncoding(max_seq_len, d_model);

        //2.transformer block
        for(int i=0;i<num_heads;i++){
            blocks.push_back(new DecoderBlock(d_model, num_heads, d_ff, max_seq_len, batch_size));
        }

        //3. Output archeitecture
        final_ln = new layers::LayerNorm(d_model);
        lm_head = new layers::Linear(d_model, vocab_size);      // The LM Head projects from d_model back to the vocabulary size

        std::vector<int> hidden_shape = {batch_size, max_seq_len, d_model};
        hidden_state = new Tensor(hidden_shape);

        d_hidden_state = new Tensor(hidden_shape);
    }

    GPT::~GPT(){
        delete tok_emb;
        delete pos_emb;
        for(auto block : blocks) delete block;
        delete final_ln;
        delete lm_head;
        delete hidden_state;
        delete d_hidden_state;
    }

    void GPT::forward(int* d_input_ids, Tensor* logits) {
        int batch_size = logits->shape[0];
        int seq_len = logits->shape[1];
        int total_tokens = batch_size * seq_len;

        // ==========================================
        // THE ENGINE THROTTLE
        // ==========================================
        int original_seq = hidden_state->shape[1];
        int original_size = hidden_state->size;
        
        // Shrink the physical dimensions for inference
        hidden_state->shape[1] = seq_len;
        hidden_state->size = total_tokens * tok_emb->d_model; // Assuming your emb layer holds d_model

        // 1. Embeddings
        tok_emb->forward(d_input_ids, hidden_state, total_tokens);

        // 2. Time-Shift
        int start_pos = 0;
        if (blocks[0]->mha->use_kv_cache) {
            start_pos = blocks[0]->mha->current_cache_len;
        }
        pos_emb->forward(hidden_state, start_pos);

        // 3. Deep Neural Network
        for (auto block : blocks) {
            block->forward(hidden_state);
        }

        // 4. Output
        final_ln->forward(hidden_state, hidden_state);
        lm_head->forward(hidden_state, logits); // Will now safely write exactly 1 row!

        // ==========================================
        // RESTORE DIMENSIONS
        // ==========================================
        hidden_state->shape[1] = original_seq;
        hidden_state->size = original_size;
    }

    void GPT::backward(Tensor* dLogits){
        int total_tokens = batch_size * max_seq_len;

        // Step 1: Backprop through the Language Model Head
        // Takes dLogits [batch, seq, vocab] -> Outputs d_hidden_state [batch, seq, d_model]
        lm_head->backward(dLogits, d_hidden_state);

        // Step 2: Backprop through the Final Layer Norm
        // We do this in-place to save memory!
        final_ln->backward(d_hidden_state, d_hidden_state);

        // Step 3: Backprop through the Decoder Blocks IN REVERSE ORDER
        for (int i = num_layers - 1; i >= 0; i--) {
            // DecoderBlock modifies d_hidden_state in-place via residual addition
            blocks[i]->backward(d_hidden_state);
        }

        // Note: Positional Encodings have no learnable weights. The gradients 
        // simply flow straight past them, unchanged.

        // Step 4: Backprop into the Token Embeddings
        // Maps the d_hidden_state vectors back to specific vocabulary IDs
        tok_emb->backward(d_hidden_state, total_tokens);
    }

    // --------------------------------------------------------
    // PARAMETER GATHERING (For the Optimizer)
    // --------------------------------------------------------
    std::vector<Tensor*> GPT::parameters() {
        std::vector<Tensor*> params;

        // 1. Embeddings
        params.push_back(tok_emb->weight);
        // Note: Positional Encodings are fixed math, NOT learnable parameters!

        // 2. Decoder Blocks
        for (auto block : blocks) {
            // LayerNorm 1
            params.push_back(block->ln1->gamma);
            params.push_back(block->ln1->beta);
            
            // Attention Projections
            params.push_back(block->mha->W_q->W); params.push_back(block->mha->W_q->b);
            params.push_back(block->mha->W_k->W); params.push_back(block->mha->W_k->b);
            params.push_back(block->mha->W_v->W); params.push_back(block->mha->W_v->b);
            params.push_back(block->mha->W_o->W); params.push_back(block->mha->W_o->b);
            
            // LayerNorm 2
            params.push_back(block->ln2->gamma);
            params.push_back(block->ln2->beta);

            // Feed-Forward
            params.push_back(block->ffn->w1->W);  params.push_back(block->ffn->w1->b);
            params.push_back(block->ffn->w2->W);  params.push_back(block->ffn->w2->b);
        }

        // 3. Final Output Layers
        params.push_back(final_ln->gamma);
        params.push_back(final_ln->beta);
        params.push_back(lm_head->W);
        params.push_back(lm_head->b);

        return params;
    }

    void GPT::enable_kv_cache() {
        for (auto block : blocks) {
            block->enable_kv_cache();
        }
    }
    void GPT::disable_kv_cache() {
        for (auto block : blocks) {
            block->disable_kv_cache();
        }
    }
    void GPT::clear_kv_cache() {
        for (auto block : blocks) {
            block->clear_kv_cache();
        }
    }

    void model::GPT::save_pretrained(const std::string& filepath) {
        std::ofstream out(filepath, std::ios::binary);
        if (!out.is_open()) {
            std::cerr << "Failed to open " << filepath << " for saving." << std::endl;
            return;
        }

        std::vector<Tensor*> params = this->parameters();
        for (Tensor* p : params) {
            p->save(out);
        }
        
        out.close();
        std::cout << "Model weights successfully saved to: " << filepath << std::endl;
    }

    void model::GPT::load_pretrained(const std::string& filepath) {
        std::ifstream in(filepath, std::ios::binary);
        if (!in.is_open()) {
            std::cerr << "Failed to open " << filepath << " for loading." << std::endl;
            return;
        }

        std::vector<Tensor*> params = this->parameters();
        for (Tensor* p : params) {
            p->load(in);
        }
        
        in.close();
        std::cout << "Model weights successfully loaded from: " << filepath << std::endl;
    }


    size_t model::GPT::get_parameter_count() {
        size_t total_params = 0;
        std::vector<Tensor*> params = this->parameters();
        for (Tensor* p : params) {
            total_params += p->size;
        }
        return total_params;
    }

    void model::GPT::print_model_summary() {
        size_t total_params = get_parameter_count();
        float memory_mb = (float)(total_params * sizeof(float)) / (1024.0f * 1024.0f);

        std::cout << "\n==================================================" << std::endl;
        std::cout << "             MODEL ARCHITECTURE SUMMARY           " << std::endl;
        std::cout << "==================================================" << std::endl;
        std::cout << std::left << std::setw(20) << "Vocab Size:" << vocab_size << std::endl;
        std::cout << std::left << std::setw(20) << "Context Window:" << max_seq_len << " tokens" << std::endl;
        std::cout << std::left << std::setw(20) << "Embedding Dim:" << d_model << std::endl;
        std::cout << std::left << std::setw(20) << "Attention Heads:" << num_heads << std::endl;
        std::cout << std::left << std::setw(20) << "FeedForward Dim:" << d_ff << std::endl;
        std::cout << std::left << std::setw(20) << "Transformer Layers:" << num_layers << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << std::left << std::setw(20) << "Total Parameters:" << total_params << std::endl;
        std::cout << std::left << std::setw(20) << "Physical Size:" << std::fixed << std::setprecision(2) << memory_mb << " MB (FP32)" << std::endl;
        std::cout << "==================================================\n" << std::endl;
    }

    void model::GPT::save_int8(const std::string& filepath) {
        std::ofstream out(filepath, std::ios::binary);
        std::vector<Tensor*> params = this->parameters();
        for (Tensor* p : params) p->save_int8(out);
        out.close();
        std::cout << "INT8 Compressed Brain saved to: " << filepath << std::endl;
    }

    void model::GPT::load_int8(const std::string& filepath) {
        std::ifstream in(filepath, std::ios::binary);
        std::vector<Tensor*> params = this->parameters();
        for (Tensor* p : params) p->load_int8(in);
        in.close();
        std::cout << "INT8 Compressed Brain loaded from: " << filepath << std::endl;
    }
}