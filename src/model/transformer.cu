#include "transformer.cuh"
#include "../core/ops.cuh"

namespace model {
    
    DecoderBlock::DecoderBlock(int d_model, int num_heads, int d_ff, int max_seq_len, int batch_size){
        ln1 = new layers::LayerNorm(d_model);
        mha = new layers::MultiHeadAttention(d_model, num_heads, max_seq_len, batch_size);
        ln2 = new layers::LayerNorm(d_model);
        ffn = new layers::FeedForward(d_model, d_ff, max_seq_len, batch_size);

        std::vector<int> shape = {batch_size, max_seq_len, d_model};
        norm_cache_1 = new Tensor(shape);
        norm_cache_2 = new Tensor(shape);
        attn_out = new Tensor(shape);
        ffn_out = new Tensor(shape);

        d_norm_cache = new Tensor(shape);
        d_attn_out = new Tensor(shape);
        d_ffn_out = new Tensor(shape);
    }

    DecoderBlock::~DecoderBlock(){
        delete ln1;
        delete mha;
        delete ln2;
        delete ffn;
        delete norm_cache_1;
        delete norm_cache_2;
        delete ffn_out;
        delete attn_out;
        delete d_norm_cache;
        delete d_ffn_out;
        delete d_attn_out;
    }

    void DecoderBlock::forward(Tensor* X) {
        // 1. Synchronize the internal caches to the input throttle!
        int seq_len = X->shape[1];
        int dynamic_size = X->size;

        norm_cache_1->shape[1] = seq_len; norm_cache_1->size = dynamic_size;
        attn_out->shape[1] = seq_len;     attn_out->size = dynamic_size;
        norm_cache_2->shape[1] = seq_len; norm_cache_2->size = dynamic_size;
        ffn_out->shape[1] = seq_len;      ffn_out->size = dynamic_size;

        // 2. Standard Forward Pass
        ln1->forward(X, norm_cache_1);
        mha->forward(norm_cache_1, attn_out);
        ops::add_tensors(X, attn_out);

        ln2->forward(X, norm_cache_2);
        ffn->forward(norm_cache_2, ffn_out);
        ops::add_tensors(X, ffn_out);
    }


    void DecoderBlock::backward(Tensor* dX){
        // --- REVERSE SUBLAYER 2: FEED-FORWARD ---
        // 1. FFN Backward: Calculates gradients for w1/w2 and outputs d_norm_cache
        ffn->backward(dX, d_norm_cache);
        
        // 2. LayerNorm 2 Backward: Calculates gradients for gamma/beta and outputs d_ffn_out
        ln2->backward(d_norm_cache, d_ffn_out);
        
        // 3. Residual 2: Add the sublayer gradient back into the main stream
        ops::add_tensors(dX, d_ffn_out);

        // --- REVERSE SUBLAYER 1: ATTENTION ---
        // 4. Attention Backward: Calculates gradients for Q/K/V/O and outputs d_norm_cache
        mha->backward(dX, d_norm_cache);
        
        // 5. LayerNorm 1 Backward: Calculates gradients for gamma/beta and outputs d_attn_out
        ln1->backward(d_norm_cache, d_attn_out);
        
        // 6. Residual 1: Add the sublayer gradient back into the main stream
        ops::add_tensors(dX, d_attn_out);
    }

    void DecoderBlock::enable_kv_cache() { 
        mha->enable_kv_cache(); 
    }
    void DecoderBlock::disable_kv_cache() {
         mha->disable_kv_cache(); 
    }
    void DecoderBlock::clear_kv_cache() { 
        mha->clear_kv_cache();
    }
}