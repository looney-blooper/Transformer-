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

    void DecoderBlock::forward(Tensor* X){
        // --- SUBLAYER 1: ATTENTION ---
        // 1. Pre-Norm: norm_cache = LayerNorm(X)
        ln1->forward(X, norm_cache_1);

        // 2. Attention: attn_out = MHA(norm_cache)
        mha->forward(norm_cache_1, attn_out);

        // 3. Residual 1: X = X + attn_out
        ops::add_tensors(X, attn_out);

        // --- SUBLAYER 2: FEED-FORWARD ---
        // 4. Pre-Norm: norm_cache = LayerNorm(X) 
        // (We safely overwrite the old norm_cache to save VRAM!)
        ln2->forward(X, norm_cache_2);

        // 5. FFN: ffn_out = FFN(norm_cache)
        ffn->forward(norm_cache_2, ffn_out);

        // 6. Residual 2: X = X + ffn_out
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
}