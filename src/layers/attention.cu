#include "attention.cuh"
#include "../core/ops.cuh"
#include <iostream>
#include <cmath>

// --------------------------------------------------------
// FUSED CAUSAL-MASKED SCALED-SOFTMAX KERNEL
// --------------------------------------------------------
__global__ void masked_scaled_softmax_kernel(float* attn_scores, int seq_len, float scale, int batch_size) {
    // Each thread block handles one sequence in the batch
    // Each thread handles one row (Query) inside that sequence's N x N matrix
    int b = blockIdx.y; 
    int row = blockIdx.x * blockDim.x + threadIdx.x; 

    if (b < batch_size && row < seq_len) {
        // Calculate the starting index of this specific row in the flattened 1D memory array
        int row_offset = (b * seq_len * seq_len) + (row * seq_len);

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
__global__ void masked_softmax_backward_kernel(float* dP, float* P, int seq_len, float scale, int batch_size) {
    // b = batch index, row = token query index
    int b = blockIdx.y; 
    int row = blockIdx.x * blockDim.x + threadIdx.x; 

    if (b < batch_size && row < seq_len) {
        int row_offset = (b * seq_len * seq_len) + (row * seq_len);

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

        // Step 3: Zero out the causally masked future just to be perfectly safe
        for (int col = row + 1; col < seq_len; col++) {
            dP[row_offset + col] = 0.0f;
        }
    }
}

namespace layers {
    MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads, int max_seq_len, int batch_size) {
        this->d_model = d_model;
        this->num_heads = num_heads;
        this->d_k = d_model / num_heads;

        W_q = new Linear(d_model, d_model);
        W_k = new Linear(d_model, d_model);
        W_v = new Linear(d_model, d_model);
        W_o = new Linear(d_model, d_model);

        // FIX 1: Q, K, V must hold the entire sequence! 
        // Shape: [batch_size, max_seq_len, d_model]
        std::vector<int> qkv_shape = {batch_size, max_seq_len, d_model};
        Q = new Tensor(qkv_shape);
        K = new Tensor(qkv_shape);
        V = new Tensor(qkv_shape);
        Context = new Tensor(qkv_shape); // Context output has the same shape


        //Gradients
        dQ = new Tensor(qkv_shape);
        dK = new Tensor(qkv_shape);
        dV = new Tensor(qkv_shape);
        dContext = new Tensor(qkv_shape);

        dX_q = new Tensor(qkv_shape);
        dX_k = new Tensor(qkv_shape);
        dX_v = new Tensor(qkv_shape);


        // FIX 2: Attention Scores must be an N x N matrix per batch
        // Shape: [batch_size, max_seq_len, max_seq_len]
        std::vector<int> attn_shape = {batch_size, max_seq_len, max_seq_len};
        Attention_Scores = new Tensor(attn_shape);

        dAttention_Scores = new Tensor(attn_shape);
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
    }

    void MultiHeadAttention::forward(Tensor* X, Tensor* Y){
        int batch_size = X->shape[0];
        
        // Step 1: Calculate Q, K, V using our Linear layers
        W_q->forward(X, Q);
        W_k->forward(X, K);
        W_v->forward(X, V);

        // Step 2: Calculate Attention Scores (Q * K^T)
        // transB = true tells cuBLAS to transpose K on the fly
        ops::matmul(Q, K, Attention_Scores, false, true);

        /// Setup Grid for [batch_size, seq_len, seq_len]
        int seq_len = Attention_Scores->shape[1]; // e.g., 128
        float scale = 1.0f / sqrtf((float)d_k);

        int threads_per_block = 128; // One thread per row in the seq_len
        int blocks_x = (seq_len + threads_per_block - 1) / threads_per_block;
        int blocks_y = batch_size; // One block grid per batch

        dim3 gridDim(blocks_x, blocks_y);
        dim3 blockDim(threads_per_block);

        masked_scaled_softmax_kernel<<<gridDim, blockDim>>>(
            Attention_Scores->d_data, 
            seq_len, 
            scale, 
            batch_size
        );

        // Check for Softmax kernel errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Softmax Kernel Failed: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }

        // Step 5: Multiply by V (Context = Attention_Scores * V)
        // Attention_Scores is [seq_len, seq_len], V is [seq_len, d_model]
        // Result Context is [seq_len, d_model]
        ops::matmul(Attention_Scores, V, Context);
        
        // Step 6: Final output projection (Y = Context * W_o)
        // Context is [seq_len, d_model], W_o is [d_model, d_model]
        // Result Y is [seq_len, d_model]
        if (Y != nullptr) {
            W_o->forward(Context, Y);
        }
    }


    void MultiHeadAttention::backward(Tensor* dY, Tensor* dX){
        int batch_size = Q->shape[0];
        int seq_len = Q->shape[1];

        // 1. Backprop through the Output Projection (W_o)
        // dContext = dY * W_o^T
        W_o->backward(dY, dContext);

        // 2. Backprop through the (Scores * V) multiplication
        // dV = Scores^T * dContext (Note: Transpose A!)
        ops::matmul(Attention_Scores, dContext, dV, true, false);

        // dScores = dContext * V^T (Note: Transpose B!)
        ops::matmul(dContext, V, dAttention_Scores, false, true);

        // 3. Backprop through the Causal Softmax
        float scale = 1.0f / sqrtf((float)d_k);
        int threads_per_block = 128;
        int blocks_x = (seq_len + threads_per_block - 1) / threads_per_block;
        int blocks_y = batch_size;

        dim3 gridDim(blocks_x, blocks_y);
        dim3 blockDim(threads_per_block);

        masked_softmax_backward_kernel<<<gridDim, blockDim>>>(
            dAttention_Scores->d_data, // acts as dP input, overwritten to become dS output
            Attention_Scores->d_data,  // forward probabilities P
            seq_len, 
            scale, 
            batch_size
        );

        // 4. Backprop through the (Q * K^T) multiplication
        // dQ = dScores * K
        ops::matmul(dAttention_Scores, K, dQ, false, false);
        
        // dK = dScores^T * Q (Note: Transpose A!)
        ops::matmul(dAttention_Scores, Q, dK, true, false);

        // 5. Backprop through the Q, K, V Linear Projections
        W_q->backward(dQ, dX_q);
        W_k->backward(dK, dX_k);
        W_v->backward(dV, dX_v);

        // 6. Accumulate the final gradient for X
        // Because the input X split into three paths (Q, K, V), the chain rule 
        // dictates we must sum their gradients together to get the total dX.
        
        // First, zero out dX since we are going to accumulate into it
        cudaMemset(dX->d_data, 0, dX->size * sizeof(float));

        ops::add_tensors(dX, dX_q);
        ops::add_tensors(dX, dX_k);
        ops::add_tensors(dX, dX_v);
    }
}