#include "gpt.cuh"
#include <iostream>

namespace model {
    GPT::GPT(int vocab_size, int d_model, int num_heads, int d_ff, int num_layers, int max_seq_len, int batch_size){
        this->vocab_size = vocab_size;
        this->d_model = d_model;
        this->num_heads = num_heads;
        this->num_layers = num_layers;
        this->batch_size = batch_size;
        this->max_seq_len = max_seq_len;

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

    void GPT::forward(int* d_input_ids, Tensor* logits){
        int total_tokens = batch_size * max_seq_len;

        // Step 1: Token Embeddings (Vocab IDs -> Float Vectors)
        tok_emb->forward(d_input_ids, hidden_state, total_tokens);

        // Step 2: Add Positional Encodings (In-place addition)
        pos_emb->forward(hidden_state);

        // Step 3: Pass through all N Decoder Blocks
        for (int i = 0; i < num_layers; i++) {
            blocks[i]->forward(hidden_state);
        }

        // Step 4: Final Layer Normalization (In-place to save memory)
        final_ln->forward(hidden_state, hidden_state);

        // Step 5: Language Model Head Projection (Hidden -> Vocab Size)
        lm_head->forward(hidden_state, logits);
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
}