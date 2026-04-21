#include "modules.cuh"
#include "../core/ops.cuh"
#include <cmath>

// --------------------------------------------------------
// 1. THE CUSTOM CUDA KERNEL
// --------------------------------------------------------
// This function runs on the GPU. Every thread calculates its own global index
// and adds the correct bias value to the output matrix Y.
__global__ void add_bias_kernel(float* Y, float* b, int total_elements, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int col = idx % out_features; 
        Y[idx] += b[col];
    }
}

// --------------------------------------------------------
// 2. THE BACKWARD BIAS KERNEL
// --------------------------------------------------------
// One thread per output feature. Each thread loops down the batch column
// and sums the gradients to calculate exactly how the bias should update.
__global__ void backward_bias_kernel(float* dY, float* db, int rows, int out_features) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < out_features) {
        float sum = 0.0f;
        for (int row = 0; row < rows; row++) {
            sum += dY[row * out_features + col];
        }
        db[col] = sum; 
    }
}

// --------------------------------------------------------
// LAYERNORM FORWARD KERNEL
// --------------------------------------------------------
__global__ void layernorm_forward_kernel(
    float* X, float* Y, float* gamma, float* beta, 
    float* mean_out, float* inv_std_out, 
    int d_model, float eps, int total_tokens) 
{
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (token_idx < total_tokens) {
        int offset = token_idx * d_model;
        
        // 1. Calculate Mean
        float sum = 0.0f;
        for(int i = 0; i < d_model; i++) {
            sum += X[offset + i];
        }
        float m = sum / d_model;
        mean_out[token_idx] = m; // SAVE THE MEAN
        
        // 2. Calculate Variance & Inverse StdDev
        float var_sum = 0.0f;
        for(int i = 0; i < d_model; i++) {
            float diff = X[offset + i] - m;
            var_sum += diff * diff;
        }
        float var = var_sum / d_model;
        float istd = rsqrtf(var + eps); // 1.0 / sqrt(var + eps)
        inv_std_out[token_idx] = istd; // SAVE THE INV_STD
        
        // 3. Normalize and Apply Gamma/Beta
        for(int i = 0; i < d_model; i++) {
            float x_hat = (X[offset + i] - m) * istd;
            Y[offset + i] = gamma[i] * x_hat + beta[i];
        }
    }
}

// --------------------------------------------------------
// LAYERNORM BACKWARD KERNEL (Calculates dX)
// --------------------------------------------------------
__global__ void layernorm_backward_kernel_dx(
    float* dX, float* dY, float* X_saved, float* gamma, 
    float* mean, float* inv_std, int d_model, int total_tokens) 
{
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (token_idx < total_tokens) {
        int offset = token_idx * d_model;
        float m = mean[token_idx];
        float istd = inv_std[token_idx];
        
        float sum_dy_gamma = 0.0f;
        float sum_dy_gamma_xhat = 0.0f;
        
        for(int i = 0; i < d_model; i++) {
            float dy_val = dY[offset + i];
            float g_val = gamma[i];
            float x_hat = (X_saved[offset + i] - m) * istd;
            
            sum_dy_gamma += dy_val * g_val;
            sum_dy_gamma_xhat += dy_val * g_val * x_hat;
        }
        
        float N = (float)d_model;
        for(int i = 0; i < d_model; i++) {
            float dy_val = dY[offset + i];
            float g_val = gamma[i];
            float x_hat = (X_saved[offset + i] - m) * istd;
            
            // The exact chain rule derivative
            float dx_val = (N * dy_val * g_val - sum_dy_gamma - x_hat * sum_dy_gamma_xhat) / N;
            dX[offset + i] = dx_val * istd;
        }
    }
}

// --------------------------------------------------------
// GELU KERNEL
// --------------------------------------------------------
__global__ void gelu_kernel(float* X, float* Y, int total_elements){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        float x = X[idx];
        // The GPT GELU approximation math
        // sqrt(2/pi) is approximately 0.7978845608
        float cube = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * cube);
        float tanh_val = tanhf(inner); // tanhf is the CUDA built-in fast tanh
        
        Y[idx] = 0.5f * x * (1.0f + tanh_val);
    }
}

// --------------------------------------------------------
// GELU BACKWARD KERNEL
// --------------------------------------------------------
__global__ void gelu_backward_kernel(float* dY, float* X, float* dX, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        float x = X[idx];
        
        // The derivative of the GPT GELU approximation
        float cube = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * cube);
        float tanh_val = tanhf(inner);
        
        float sech2 = 1.0f - tanh_val * tanh_val;
        float derivative = 0.5f * (1.0f + tanh_val) + 
                           0.5f * x * sech2 * 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);
        
        // Chain rule: Multiply incoming gradient (dY) by local derivative
        dX[idx] = dY[idx] * derivative;
    }
}

// --------------------------------------------------------
// EMBEDDING LOOKUP KERNEL
// --------------------------------------------------------
__global__ void embedding_forward_kernel(int* input_ids, float* weights, float* output, int d_model, int total_tokens) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (token_idx < total_tokens) {
        int vocab_id = input_ids[token_idx];
        
        // Calculate memory offsets
        int weight_offset = vocab_id * d_model;
        int output_offset = token_idx * d_model;

        // Copy the specific row from the weight matrix into the sequence
        for (int i = 0; i < d_model; i++) {
            output[output_offset + i] = weights[weight_offset + i];
        }
    }
}

// --------------------------------------------------------
// EMBEDDING BACKWARD KERNEL
// --------------------------------------------------------
__global__ void embedding_backward_kernel(float* dY, float* dWeight, int* input_ids, int d_model, int total_tokens) {
    // Each thread handles exactly ONE token in the sequence
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (token_idx < total_tokens) {
        int vocab_id = input_ids[token_idx];
        
        // Calculate memory offsets
        int weight_offset = vocab_id * d_model;
        int dy_offset = token_idx * d_model;

        // Safely accumulate the gradients into the specific vocabulary row
        for (int i = 0; i < d_model; i++) {
            // atomicAdd guarantees no memory collisions if multiple tokens have the same vocab_id!
            atomicAdd(&dWeight[weight_offset + i], dY[dy_offset + i]);
        }
    }
}

// --------------------------------------------------------
// ADD POSITIONAL ENCODING KERNEL
// --------------------------------------------------------
__global__ void add_pe_kernel(float* X, float* PE, int seq_len, int d_model, int batch_size) {
    // Each thread handles one token's entire embedding vector
    int b = blockIdx.y; 
    int pos = blockIdx.x * blockDim.x + threadIdx.x; 

    if (b < batch_size && pos < seq_len) {
        int x_offset = (b * seq_len * d_model) + (pos * d_model);
        int pe_offset = pos * d_model; // PE is identical for every sequence in the batch

        for (int i = 0; i < d_model; i++) {
            X[x_offset + i] += PE[pe_offset + i]; // In-place addition
        }
    }
}


namespace layers {

    // --------------------------------------------------------
    // LINEAR LAYER CONSTRUCTOR
    // --------------------------------------------------------
    Linear::Linear(int in_feat, int out_feat) {
        in_features = in_feat;
        out_features = out_feat;

        // Allocate weights [in_features, out_features]
        W = new Tensor({in_features, out_features});
        
        // Allocate bias [1, out_features]
        b = new Tensor({1, out_features});
        
        X_cache = nullptr; // Initialize cache as null

        // TODO: In a real scenario, we must initialize W with Xavier/Kaiming 
        // initialization instead of leaving it as random memory junk.
    }

    Linear::~Linear() {
        delete W;
        delete b;
    }

    
    void Linear::forward(Tensor* X, Tensor* Y) {
        // Cache the input. We NEED this later to calculate gradients in the backward pass!
        X_cache = X;

        // Step A: Matrix Multiplication (Y = X * W)
        ops::matmul(X, W, Y);

        // Step B: Launch the custom CUDA kernel to add the bias (Y = Y + b)
        int total_elements = Y->size;
        
        // CUDA configuration: 256 threads per block, calculate how many blocks we need
        int threads_per_block = 256;
        int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

        // Launch the kernel on the GPU
        add_bias_kernel<<<blocks, threads_per_block>>>(Y->d_data, b->d_data, total_elements, out_features);
        
        // Catch any kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Bias Kernel Failed: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    
    void Linear::backward(Tensor* dY, Tensor* dX) {
        if (X_cache == nullptr) {
            std::cerr << "Error: Cannot run backward pass before forward pass!" << std::endl;
            exit(EXIT_FAILURE);
        }

        int batch_size = X_cache->shape[0];

        // 1. Calculate Gradient for Weights (dW = X^T * dY)
        // We temporarily swap the data pointer with the grad pointer to route the 
        // cuBLAS output directly into the gradient memory space.
        float* temp_W_data = W->d_data;
        W->d_data = W->d_grad; 
        
        // transA = true means we transpose X_cache
        ops::matmul(X_cache, dY, W, true, false); 
        
        // Restore the original data pointer
        W->d_data = temp_W_data;

        // 2. Calculate Gradient for Input (dX = dY * W^T)
        // This is the gradient passed backward to the previous layer.
        // transB = true means we transpose W
        if (dX != nullptr) {
            ops::matmul(dY, W, dX, false, true);
        }

        // 3. Calculate Gradient for Bias (db = sum(dY, axis=0))
        int rows = dY->size / out_features;
        int threads_per_block = 256;
        int blocks = (out_features + threads_per_block - 1) / threads_per_block;
        
        backward_bias_kernel<<<blocks, threads_per_block>>>(dY->d_data, b->d_grad, rows, out_features);
        
        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Backward Bias Kernel Failed: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // --------------------------------------------------------
    // LayerNorm IMPLEMENTATION
    // --------------------------------------------------------
    LayerNorm::LayerNorm(int d_model, float eps) {
        this->d_model = d_model;
        this->eps = eps;

        gamma = new Tensor({1, d_model});
        beta = new Tensor({1, d_model});

        X_save = nullptr;
        mean = nullptr;
        inv_std = nullptr;

        for(int i = 0; i < d_model; i++) {
            gamma->h_data[i] = 1.0f;
            beta->h_data[i] = 0.0f;
        }
        gamma->to_device();
        beta->to_device();
    }

    LayerNorm::~LayerNorm(){
        delete gamma;
        delete beta;
        if (X_save != nullptr) delete X_save;
        if (mean != nullptr) delete mean;
        if (inv_std != nullptr) delete inv_std;
    }

    void LayerNorm::forward(Tensor* X, Tensor* Y){
        int total_tokens = X->size / d_model;

        // Allocate deep copies and statistic caches on the very first pass
        if (X_save == nullptr) {
            X_save = new Tensor(X->shape);
            mean = new Tensor({X->shape[0], X->shape[1], 1}); // 1 value per token
            inv_std = new Tensor({X->shape[0], X->shape[1], 1}); 
        }

        // DEEP COPY to protect against the Residual Trap
        cudaMemcpy(X_save->d_data, X->d_data, X->size * sizeof(float), cudaMemcpyDeviceToDevice);

        int threads_per_block = 256;
        int blocks = (total_tokens + threads_per_block - 1)/ threads_per_block;

        layernorm_forward_kernel<<<blocks, threads_per_block>>>(
            X->d_data, Y->d_data, gamma->d_data, beta->d_data, 
            mean->d_data, inv_std->d_data, // Pass the new caches!
            d_model, eps, total_tokens
        );
    }

    void LayerNorm::backward(Tensor* dY, Tensor* dX){
        int total_tokens = dY->size / d_model;
        int threads_per_block = 256;
        int blocks = (total_tokens + threads_per_block - 1) / threads_per_block;

        // Launch the true calculus kernel!
        layernorm_backward_kernel_dx<<<blocks, threads_per_block>>>(
            dX->d_data, dY->d_data, X_save->d_data, gamma->d_data, 
            mean->d_data, inv_std->d_data, d_model, total_tokens
        );
        
        // (Note: If you have a separate kernel to calculate dGamma/dBeta, launch it here)
    }

    // --------------------------------------------------------
    // GELU IMPLEMENTATION
    // --------------------------------------------------------
    void GELU::forward(Tensor* X, Tensor* Y){
        X_cache = X;

        int total_elements = X->size;
        int threads_per_block = 256;
        int blocks = (total_elements + threads_per_block - 1)/ threads_per_block;

        gelu_kernel<<<blocks, threads_per_block>>>(X->d_data, Y->d_data, total_elements);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "GELU Kernel Failed: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    void GELU::backward(Tensor* dY, Tensor* dX){
        int total_elements = X_cache->size;
        int threads_per_block = 256;
        int blocks = (total_elements + threads_per_block - 1)/ threads_per_block;

        gelu_backward_kernel<<<blocks, threads_per_block>>>(dY->d_data, X_cache->d_data, dX->d_data, total_elements);
    }

    // --------------------------------------------------------
    // FeedForward IMPLEMENTATION
    // --------------------------------------------------------
    FeedForward::FeedForward(int d_model, int d_ff, int max_seq_len, int batch_size){
        w1 = new Linear(d_model, d_ff);
        w2 = new Linear(d_ff, d_model);

        activation = new GELU();

        std::vector<int> hidden_shape = {batch_size, max_seq_len, d_ff};
        hidden_cache = new Tensor(hidden_shape);
        activated_cache = new Tensor(hidden_shape);
    }

    FeedForward::~FeedForward(){
        delete w1;
        delete w2;
        delete activation;
        delete hidden_cache;
        delete activated_cache;
    }

    void FeedForward::forward(Tensor* X, Tensor* Y){
        // Step 1: Linear projection to hidden dimension (X * w1) -> hidden_cache
        w1->forward(X, hidden_cache);

        activation->forward(hidden_cache, activated_cache);

        // Step 3: Linear projection back to original dimension (activation_cache * w2) -> Y
        w2->forward(activated_cache, Y);
    }

    void FeedForward::backward(Tensor* dY, Tensor* dX) {
        // 1. w2 backward: It uses 'activated_cache' to calculate dW2. 
        // Then, it writes the gradient (dX2) directly OVER the 'activated_cache' memory!
        w2->backward(dY, activated_cache);

        // 2. GELU backward: It reads the incoming gradient from 'activated_cache', 
        // uses 'hidden_cache' to calculate the derivative, and writes the new gradient 
        // directly OVER the 'hidden_cache' memory!
        activation->backward(activated_cache, hidden_cache);

        // 3. w1 backward: It reads the gradient from 'hidden_cache', calculates dW1,
        // and finally writes the ultimate gradient out to dX.
        w1->backward(hidden_cache, dX);
    }


    // --------------------------------------------------------
    // EMBEDDING IMPLEMENTATION
    // --------------------------------------------------------
    Embedding::Embedding(int vocab_size, int d_model) {
        this->vocab_size = vocab_size;
        this->d_model = d_model;
        weight = new Tensor({vocab_size, d_model});
        
        dWeight = new Tensor({vocab_size, d_model});
        ids_cache = nullptr;
        // TODO: In a real model, we initialize this with random normal distribution
    }

    Embedding::~Embedding() {
        delete weight;
        delete dWeight;
    }

    void Embedding::forward(int* X_ids, Tensor* Y, int total_tokens) {
        ids_cache = X_ids;

        int threads_per_block = 256;
        int blocks = (total_tokens + threads_per_block - 1) / threads_per_block;

        embedding_forward_kernel<<<blocks, threads_per_block>>>(
            X_ids, weight->d_data, Y->d_data, d_model, total_tokens
        );
    }

    void Embedding::backward(Tensor* dY, int total_tokens){
       // 1. Zero out the old gradients in the NATIVE d_grad array
        cudaMemset(weight->d_grad, 0, weight->size * sizeof(float));

        int threads_per_block = 256;
        int blocks = (total_tokens + threads_per_block - 1) / threads_per_block;

        // 2. Route the atomic additions directly into weight->d_grad
        embedding_backward_kernel<<<blocks, threads_per_block>>>(
            dY->d_data, weight->d_grad, ids_cache, d_model, total_tokens
        );
    }

    // --------------------------------------------------------
    // POSITIONAL ENCODING IMPLEMENTATION
    // --------------------------------------------------------
    PositionalEncoding::PositionalEncoding(int max_seq_len, int d_model) {
        this->max_seq_len = max_seq_len;
        this->d_model = d_model;
        pe_matrix = new Tensor({max_seq_len, d_model}, false); // requires_grad = false!

        // Generate the sine/cosine grid on the CPU
        for (int pos = 0; pos < max_seq_len; pos++) {
            for (int i = 0; i < d_model; i += 2) {
                // Calculate the frequency denominator
                float div_term = powf(10000.0f, (float)i / (float)d_model);
                
                // Even dimensions get Sine
                pe_matrix->h_data[pos * d_model + i] = sinf((float)pos / div_term);
                
                // Odd dimensions get Cosine (make sure we don't go out of bounds)
                if (i + 1 < d_model) {
                    pe_matrix->h_data[pos * d_model + i + 1] = cosf((float)pos / div_term);
                }
            }
        }
        
        // Push the calculated grid to the GPU VRAM
        pe_matrix->to_device(); 
    }

    PositionalEncoding::~PositionalEncoding() {
        delete pe_matrix;
    }

    void PositionalEncoding::forward(Tensor* X) {
        int batch_size = X->shape[0];
        int seq_len = X->shape[1];

        int threads_per_block = 128;
        int blocks_x = (seq_len + threads_per_block - 1) / threads_per_block;
        int blocks_y = batch_size;

        dim3 grid(blocks_x, blocks_y);
        dim3 block(threads_per_block);

        add_pe_kernel<<<grid, block>>>(X->d_data, pe_matrix->d_data, seq_len, d_model, batch_size);
    }
}