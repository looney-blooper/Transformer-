#include "attention.cuh"
#include "../core/ops.cuh"
#include <iostream>
#include <cmath>

// [Batch, Seq, Heads, Dk] -> [Batch, Heads, Seq, Dk]
__global__ void transpose_qkv_kernel(float* src, float* dst, int batch, int seq, int heads, int d_k) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int s = blockIdx.x * blockDim.x + threadIdx.x;

    if (s < seq) {
        for (int d = 0; d < d_k; d++) {
            // Original index in [B, S, H, Dk]
            int src_idx = (b * seq * heads * d_k) + (s * heads * d_k) + (h * d_k) + d;
            // Target index in [B, H, S, Dk]
            int dst_idx = (b * heads * seq * d_k) + (h * seq * d_k) + (s * d_k) + d;
            dst[dst_idx] = src[src_idx];
        }
    }
}

// [Batch, Heads, Seq, Dk] -> [Batch, Seq, Heads, Dk]
__global__ void untranspose_output_kernel(float* src, float* dst, int batch, int heads, int seq, int d_k) {
    int b = blockIdx.z;
    int s = blockIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;

    if (h < heads) {
        for (int d = 0; d < d_k; d++) {
            // Original index in [B, H, S, Dk]
            int src_idx = (b * heads * seq * d_k) + (h * seq * d_k) + (s * d_k) + d;
            // Target index in [B, S, H, Dk]
            int dst_idx = (b * seq * heads * d_k) + (s * heads * d_k) + (h * d_k) + d;
            dst[dst_idx] = src[src_idx];
        }
    }
}

// --------------------------------------------------------
// FUSED CAUSAL-MASKED SCALED-SOFTMAX KERNEL
// --------------------------------------------------------
__global__ void masked_scaled_softmax_kernel(float* attn_scores, int seq_len, float scale, int batch_size, int num_heads) {
    
    // We treat (batch * head) as a single flat "z-axis" dimension
    int batch_head_idx = blockIdx.y; 
    int row = blockIdx.x * blockDim.x + threadIdx.x; 
    
    int total_matrices = batch_size * num_heads;

    if (batch_head_idx < total_matrices && row < seq_len) {
        // Calculate the starting index of this specific row, accounting for WHICH head we are in!
        int row_offset = (batch_head_idx * seq_len * seq_len) + (row * seq_len);

        // Step 1: Scale, Mask, and find Max
        float max_val = -1e20f; 
        for (int col = 0; col < seq_len; col++) {
            float val;
            // THE CAUSAL MASK: If the Key (col) is ahead of the Query (row), block it.
            if (col > row) {
                val = -1e20f; // effectively -infinity
                attn_scores[row_offset + col] = val; // Overwrite memory
            } else {
                // Otherwise, scale the valid score
                val = attn_scores[row_offset + col] * scale;
                attn_scores[row_offset + col] = val; 
            }

            if (val > max_val) { max_val = val; }
        }

        // Step 2: Exponents and Sum (Only need to loop up to 'row' because the rest are 0 after exp!)
        float sum_exp = 0.0f;
        for (int col = 0; col <= row; col++) {
            float exp_val = expf(attn_scores[row_offset + col] - max_val);
            attn_scores[row_offset + col] = exp_val;
            sum_exp += exp_val;
        }

        // The masked tokens become exactly 0.0
        for (int col = row + 1; col < seq_len; col++) {
            attn_scores[row_offset + col] = 0.0f;
        }

        // Step 3: Divide by sum to get final probabilities
        for (int col = 0; col <= row; col++) {
            attn_scores[row_offset + col] /= sum_exp;
        }
    }
}

// --------------------------------------------------------
// FUSED SCALED-SOFTMAX KERNEL
// --------------------------------------------------------
__global__ void scaled_softmax_kernel(float* attn_scores, int seq_len, float scale, int total_rows) {
    // Each thread handles exactly one row of the Attention matrix
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < total_rows) {
        int row_offset = row * seq_len;

        // Step 1: Scale the row and find the maximum value for numerical stability
        float max_val = -1e20f; // Start with a very small number
        for (int i = 0; i < seq_len; i++) {
            // Apply the 1/sqrt(d_k) scale here
            attn_scores[row_offset + i] *= scale; 
            
            float val = attn_scores[row_offset + i];
            if (val > max_val) {
                max_val = val;
            }
        }

        // Step 2: Calculate exponents and their sum
        float sum_exp = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            // Subtract max_val before exponentiating!
            float exp_val = expf(attn_scores[row_offset + i] - max_val);
            attn_scores[row_offset + i] = exp_val; // Store temporarily
            sum_exp += exp_val;
        }

        // Step 3: Divide by the sum to get final probabilities (0.0 to 1.0)
        for (int i = 0; i < seq_len; i++) {
            attn_scores[row_offset + i] /= sum_exp;
        }
    }
}

// --------------------------------------------------------
// MASKED SCALED-SOFTMAX BACKWARD KERNEL
// --------------------------------------------------------
__global__ void masked_softmax_backward_kernel(float* dP, float* P, int seq_len, float scale, int batch_size, int num_heads) {
    // Treat (batch * head) as a single flat "y-axis" dimension
    int batch_head_idx = blockIdx.y; 
    int row = blockIdx.x * blockDim.x + threadIdx.x; 
    
    int total_matrices = batch_size * num_heads;

    if (batch_head_idx < total_matrices && row < seq_len) {
        int row_offset = (batch_head_idx * seq_len * seq_len) + (row * seq_len);

        // Step 1: Calculate the dot product of (P * dP) for this row
        float sum_p_dp = 0.0f;
        for (int col = 0; col <= row; col++) {
            sum_p_dp += P[row_offset + col] * dP[row_offset + col];
        }

        // Step 2: Calculate the final gradient and overwrite dP in-place
        for (int col = 0; col <= row; col++) {
            float p_val = P[row_offset + col];
            float dp_val = dP[row_offset + col];
            
            // The Softmax derivative + scaling factor
            dP[row_offset + col] = scale * p_val * (dp_val - sum_p_dp);
        }

        // Step 3: Zero out the causally masked future
        for (int col = row + 1; col < seq_len; col++) {
            dP[row_offset + col] = 0.0f;
        }
    }
}

// --------------------------------------------------------
// KV-CACHE APPEND KERNEL
// --------------------------------------------------------
// Takes a new vector of shape [Batch, Heads, 1, Dk]
// Injects it into Cache of shape [Batch, Heads, MaxSeq, Dk] at the specified 'step' index
__global__ void append_kv_cache_kernel(float* cache, float* new_vec, int max_seq_len, int d_k, int step) {
    // b = batch, h = head, d = dimension inside the vector
    int b = blockIdx.z;
    int h = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (d < d_k) {
        // 1. Where to read from the newly generated token
        // Shape is [B, H, 1, Dk], so we skip across heads
        int src_idx = (b * gridDim.y * d_k) + (h * d_k) + d;
        
        // 2. Where to write it in the massive historical cache
        // Shape is [B, H, MaxSeq, Dk]. We offset by 'step' to drop it in the correct row!
        int dst_idx = (b * gridDim.y * max_seq_len * d_k) + (h * max_seq_len * d_k) + (step * d_k) + d;
        
        cache[dst_idx] = new_vec[src_idx];
    }
}

// --------------------------------------------------------
// INFERENCE SOFTMAX KERNEL (Respects Cache Strides)
// --------------------------------------------------------
__global__ void inference_softmax_kernel(float* attn_scores, int active_len, int max_seq_len, float scale, int total_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < total_rows) {
        int row_offset = row * max_seq_len; // Physical memory jump to the next Head
        
        // 1. Scale and find Max
        float max_val = -1e20f;
        for (int i = 0; i < active_len; i++) {
            attn_scores[row_offset + i] *= scale;
            if (attn_scores[row_offset + i] > max_val) {
                max_val = attn_scores[row_offset + i];
            }
        }
        
        // 2. Exponentiate and Sum
        float sum_exp = 0.0f;
        for (int i = 0; i < active_len; i++) {
            float exp_val = expf(attn_scores[row_offset + i] - max_val);
            attn_scores[row_offset + i] = exp_val;
            sum_exp += exp_val;
        }
        
        // 3. Normalize
        for (int i = 0; i < active_len; i++) {
            attn_scores[row_offset + i] /= sum_exp;
        }
    }
}// --------------------------------------------------------
// CAUSAL MASK KERNEL (Prevents Cheating in Phase 1)
// --------------------------------------------------------
__global__ void causal_mask_kernel(float* scores, int max_seq_len, int total_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < total_rows) {
        // scores is shape [Batch, Heads, max_seq_len, max_seq_len]
        // We find which exact token row we are on within the sequence
        int seq_row = row % max_seq_len; 
        int offset = row * max_seq_len;
        
        // Loop through the Keys (columns)
        for (int col = 0; col < max_seq_len; col++) {
            // If the Key is in the future relative to the Query, obliterate it!
            if (col > seq_row) {
                scores[offset + col] = -1e20f;
            }
        }
    }
}

namespace layers {
    MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads, int max_seq_len, int batch_size) {
        this->d_model = d_model;
        this->num_heads = num_heads;
        this->d_k = d_model / num_heads;
        this->max_seq_len = max_seq_len;

        W_q = new Linear(d_model, d_model);
        W_k = new Linear(d_model, d_model);
        W_v = new Linear(d_model, d_model);
        W_o = new Linear(d_model, d_model);

        // --- 1. QKV Interleaved (Token-Major) ---
        std::vector<int> qkv_shape = {batch_size, max_seq_len, d_model};
        
        Q = new Tensor(qkv_shape);
        K = new Tensor(qkv_shape);
        V = new Tensor(qkv_shape);
        Context = new Tensor(qkv_shape);

        dQ = new Tensor(qkv_shape);
        dK = new Tensor(qkv_shape);
        dV = new Tensor(qkv_shape);
        dContext = new Tensor(qkv_shape);

        dX_q = new Tensor(qkv_shape);
        dX_k = new Tensor(qkv_shape);
        dX_v = new Tensor(qkv_shape);

        // --- 2. QKV Transposed (Head-Major) ---
        // CRITICAL: If any of these are missing, the kernels will Segfault!
        Q_transposed = new Tensor(qkv_shape);
        K_transposed = new Tensor(qkv_shape);
        V_transposed = new Tensor(qkv_shape);
        Context_transposed = new Tensor(qkv_shape);

        dQ_transposed = new Tensor(qkv_shape);
        dK_transposed = new Tensor(qkv_shape);
        dV_transposed = new Tensor(qkv_shape);
        dContext_transposed = new Tensor(qkv_shape);

        // --- 3. Attention Matrices (Batched across Heads) ---
        // CRITICAL: This shape MUST include num_heads to hold the 64 floats!
        std::vector<int> attn_shape = {batch_size, num_heads, max_seq_len, max_seq_len};
        
        Attention_Scores = new Tensor(attn_shape);
        dAttention_Scores = new Tensor(attn_shape); // Backward pass needs room too!

        ////////////
        //KV CACHE//
        ////////////
        use_kv_cache = false;
        current_cache_len = 0;

        std::vector<int> cache_shape = {batch_size, num_heads, max_seq_len, this->d_k};
        K_cache = new Tensor(cache_shape);
        V_cache = new Tensor(cache_shape);

        Inference_Scores = new Tensor({batch_size, num_heads, 1, max_seq_len});
    }

    MultiHeadAttention::~MultiHeadAttention(){
        delete W_q;
        delete W_k;
        delete W_v;
        delete W_o;
        delete Q;
        delete K;
        delete V;
        delete Attention_Scores;
        delete Context;

        delete dQ;
        delete dK;
        delete dV;
        delete dX_q;
        delete dX_k;
        delete dX_v;
        delete dAttention_Scores;
        delete dContext;

        delete Q_transposed;
        delete K_transposed;
        delete V_transposed;
        delete Context_transposed;

        delete dQ_transposed;
        delete dK_transposed;
        delete dV_transposed;
        delete dContext_transposed;

        delete K_cache;
        delete V_cache;

        delete Inference_Scores;
    }

    void MultiHeadAttention::forward(Tensor* X, Tensor* Y) {
        int B = X->shape[0];
        int S = X->shape[1];
        int H = num_heads;
        int Dk = d_model / H;



        if (use_kv_cache) {
            // ==========================================
            // KV-CACHE INFERENCE (O(1) Memory)
            // ==========================================
            
            // 1. Projections for the SINGLE token
            W_q->forward(X, Q);
            W_k->forward(X, K);
            W_v->forward(X, V);

            // 2. Transpose to Head-Major [B, H, 1, Dk]
            dim3 grid_trans( 1, H, B );
            transpose_qkv_kernel<<<grid_trans, 32>>>(Q->d_data, Q_transposed->d_data, B, 1, H, Dk);
            transpose_qkv_kernel<<<grid_trans, 32>>>(K->d_data, K_transposed->d_data, B, 1, H, Dk);
            transpose_qkv_kernel<<<grid_trans, 32>>>(V->d_data, V_transposed->d_data, B, 1, H, Dk);

            // 3. Append the new K and V into the Persistent Cache
            dim3 grid_append( (Dk + 255)/256, H, B );
            append_kv_cache_kernel<<<grid_append, 256>>>(
                K_cache->d_data, K_transposed->d_data, max_seq_len, Dk, current_cache_len);
            append_kv_cache_kernel<<<grid_append, 256>>>(
                V_cache->d_data, V_transposed->d_data, max_seq_len, Dk, current_cache_len);
            
            current_cache_len++; 
            int active_len = current_cache_len;

            // 4. Q * K_cache^T -> [B, H, 1, max_seq_len]
            long long int strideQ = 1 * Dk; 
            long long int strideK = max_seq_len * Dk;     // Physical gap in K_cache
            long long int strideScores = 1 * max_seq_len; // Physical gap in Inference_Scores

            ops::strided_batched_matmul_kvcache(
                Q_transposed, K_cache, Inference_Scores, B, H, 
                1, active_len, Dk, false, true, 
                strideQ, strideK, strideScores
            );

            // 5. Unmasked Inference Softmax
            int total_rows = B * H;
            int threads = 256;
            int blocks = (total_rows + threads - 1) / threads;
            float scale = 1.0f / sqrtf((float)Dk);
            
            inference_softmax_kernel<<<blocks, threads>>>(
                Inference_Scores->d_data, active_len, max_seq_len, scale, total_rows
            );

            // 6. Scores * V_cache -> [B, H, 1, Dk]
            long long int strideV = max_seq_len * Dk; // Physical gap in V_cache
            long long int strideCtx = 1 * Dk;         // Physical gap in Context_transposed

            ops::strided_batched_matmul_kvcache(
                Inference_Scores, V_cache, Context_transposed, B, H, 
                1, Dk, active_len, false, false, 
                strideScores, strideV, strideCtx
            );

            // 7. Untranspose back to Interleaved [B, 1, H, Dk]
            dim3 grid_untrans( (H + 31) / 32, 1, B );
            untranspose_output_kernel<<<grid_untrans, 32>>>(
                Context_transposed->d_data, Context->d_data, B, H, 1, Dk);

            // 8. Final Output Projection
            if (Y != nullptr) W_o->forward(Context, Y);

        } else {

            // 1. Projections
            W_q->forward(X, Q);
            W_k->forward(X, K);
            W_v->forward(X, V);

            // 2. Transpose to Head-Major [B, H, S, Dk]
            dim3 grid_trans( (S + 31) / 32, H, B );
            transpose_qkv_kernel<<<grid_trans, 32>>>(Q->d_data, Q_transposed->d_data, B, S, H, Dk);
            transpose_qkv_kernel<<<grid_trans, 32>>>(K->d_data, K_transposed->d_data, B, S, H, Dk);
            transpose_qkv_kernel<<<grid_trans, 32>>>(V->d_data, V_transposed->d_data, B, S, H, Dk);

            // 3. Batched QK^T -> Scores [B, H, S, S]
            ops::strided_batched_matmul(Q_transposed, K_transposed, Attention_Scores, B, H, S, S, Dk, false, true);

            // ==========================================
            // 3.5. THE CAUSAL MASK (No more cheating!)
            // ==========================================
            int total_rows = B * H * S; 
            int mask_threads = 256;
            int mask_blocks = (total_rows + mask_threads - 1) / mask_threads;
            
            causal_mask_kernel<<<mask_blocks, mask_threads>>>(
                Attention_Scores->d_data, S, total_rows
            );

            // 4. Fused Masked Softmax (Now actually masked!)
            int threads_per_block = 256;
            int blocks_x = (S + threads_per_block - 1) / threads_per_block;
            int blocks_y = B * H; // ONE BLOCK PER HEAD!
            
            dim3 grid(blocks_x, blocks_y);
            dim3 block(threads_per_block);
            float scale = 1.0f / sqrtf((float)Dk);
            
            masked_scaled_softmax_kernel<<<grid, block>>>(Attention_Scores->d_data, S, scale, B, H);

            // 5. Batched Scores * V -> Context [B, H, S, Dk]
            ops::strided_batched_matmul(Attention_Scores, V_transposed, Context_transposed, B, H, S, Dk, S, false, false);

            // 6. Untranspose back to Token-Major [B, S, H, Dk]
            dim3 grid_untrans( (H + 31) / 32, S, B );
            untranspose_output_kernel<<<grid_untrans, 32>>>(Context_transposed->d_data, Context->d_data, B, H, S, Dk);

            // 7. Final Output Projection
            if (Y != nullptr) {
                W_o->forward(Context, Y);
            }
        }        
    }


    void MultiHeadAttention::backward(Tensor* dY, Tensor* dX) {
        int B = Q->shape[0];
        int S = Q->shape[1];
        int H = num_heads;
        int Dk = d_model / H;

        // 1. Backprop through the Output Projection (W_o) -> Outputs interleaved dContext
        W_o->backward(dY, dContext);

        // 2. Transpose dContext into Head-Major so it aligns with our batched math!
        dim3 grid_trans( (S + 31) / 32, H, B );
        transpose_qkv_kernel<<<grid_trans, 32>>>(dContext->d_data, dContext_transposed->d_data, B, S, H, Dk);

        // 3. Backprop through the (Scores * V) multiplication (Batched!)
        // dV_transposed = Scores^T * dContext_transposed
        ops::strided_batched_matmul(Attention_Scores, dContext_transposed, dV_transposed, B, H, S, Dk, S, true, false);

        // dScores = dContext_transposed * V_transposed^T
        ops::strided_batched_matmul(dContext_transposed, V_transposed, dAttention_Scores, B, H, S, S, Dk, false, true);

        // 4. Backprop through the Causal Softmax (Batched!)
        float scale = 1.0f / sqrtf((float)Dk);
        int threads_per_block = 128;
        int blocks_x = (S + threads_per_block - 1) / threads_per_block;
        int blocks_y = B * H; // ONE BLOCK PER HEAD!

        dim3 gridDim(blocks_x, blocks_y);
        dim3 blockDim(threads_per_block);

        masked_softmax_backward_kernel<<<gridDim, blockDim>>>(
            dAttention_Scores->d_data, Attention_Scores->d_data, S, scale, B, H
        );

        // 5. Backprop through the (Q * K^T) multiplication (Batched!)
        // dQ_transposed = dScores * K_transposed
        ops::strided_batched_matmul(dAttention_Scores, K_transposed, dQ_transposed, B, H, S, Dk, S, false, false);
        
        // dK_transposed = dScores^T * Q_transposed
        ops::strided_batched_matmul(dAttention_Scores, Q_transposed, dK_transposed, B, H, S, Dk, S, true, false);

        // 6. Untranspose the gradients back to Token-Major (Interleaved)
        dim3 grid_untrans( (H + 31) / 32, S, B );
        untranspose_output_kernel<<<grid_untrans, 32>>>(dQ_transposed->d_data, dQ->d_data, B, H, S, Dk);
        untranspose_output_kernel<<<grid_untrans, 32>>>(dK_transposed->d_data, dK->d_data, B, H, S, Dk);
        untranspose_output_kernel<<<grid_untrans, 32>>>(dV_transposed->d_data, dV->d_data, B, H, S, Dk);

        // 7. Backprop through the Linear Projections
        W_q->backward(dQ, dX_q);
        W_k->backward(dK, dX_k);
        W_v->backward(dV, dX_v);

        // 8. Accumulate final gradient
        cudaMemset(dX->d_data, 0, dX->size * sizeof(float));
        ops::add_tensors(dX, dX_q);
        ops::add_tensors(dX, dX_k);
        ops::add_tensors(dX, dX_v);
    }

    void MultiHeadAttention::enable_kv_cache(){
        use_kv_cache = true;
        current_cache_len = 0;
    }

    void MultiHeadAttention::disable_kv_cache(){
        use_kv_cache = false;
        current_cache_len = 0;
    }

    void MultiHeadAttention::clear_kv_cache(){
        current_cache_len = 0;
    }
}