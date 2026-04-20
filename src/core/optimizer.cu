#include "optimizer.cuh"
#include <cmath>
#include <iostream>


// --------------------------------------------------------
// FUSED ADAM-W UPDATE KERNEL
// --------------------------------------------------------
__global__ void adamw_kernel(float* W, float* grad, float* m, float* v, 
                             float lr, float beta1, float beta2, 
                             float eps, float weight_decay, 
                             int step, int total_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        float g = grad[idx];
        float weight = W[idx];

        // 1. Update biased first moment estimate (Momentum)
        float m_new = beta1 * m[idx] + (1.0f - beta1) * g;
        m[idx] = m_new;

        // 2. Update biased second raw moment estimate (Variance)
        float v_new = beta2 * v[idx] + (1.0f - beta2) * (g * g);
        v[idx] = v_new;

        // 3. Compute bias-corrected estimates
        // powf is the CUDA math function for exponents
        float m_hat = m_new / (1.0f - powf(beta1, (float)step));
        float v_hat = v_new / (1.0f - powf(beta2, (float)step));

        // 4. AdamW Weight Decay (Decoupled from the gradient!)
        weight = weight - (lr * weight_decay * weight);

        // 5. Final Parameter Update
        weight = weight - lr * (m_hat / (sqrtf(v_hat) + eps));
        
        // Write back to main memory
        W[idx] = weight;
    }
}


namespace core {
    AdamW::AdamW(std::vector<Tensor*> params, float lr, float beta1 , float beta2 , float weight_decay ){
        this->parameters = params;
        this->lr = lr;
        this->beta1 = beta1;
        this->beta2 = beta2;
        this->weight_decay = weight_decay;
        this->eps = 1e-8f;
        this->step_count = 0;


        for(Tensor* p : parameters){
            Tensor* m = new Tensor(p->shape);
            Tensor* v = new Tensor(p->shape);

            // Initialize memory to absolute zero!
            cudaMemset(m->d_data, 0, m->size * sizeof(float));
            cudaMemset(v->d_data, 0, v->size * sizeof(float));

            m_state.push_back(m);
            v_state.push_back(v);
        }
    }

    AdamW::~AdamW(){
        for(int i=0;i<m_state.size();i++){
            delete m_state[i];
            delete v_state[i];
        }
    }

    void AdamW::step(){
        step_count++;

        for(int i=0;i<parameters.size();i++){
            Tensor* p = parameters[i];
            Tensor* m = m_state[i];
            Tensor* v = v_state[i];

            int total_elements = p->size;
            int threads_per_block = 256;
            int blocks = (total_elements + threads_per_block - 1)/threads_per_block;

            // Fire the kernel to update this specific tensor
            adamw_kernel<<<blocks, threads_per_block>>>(
                p->d_data, p->d_grad, m->d_data, v->d_data,
                lr, beta1, beta2, eps, weight_decay,
                step_count, total_elements
            );
        }
    }

    void AdamW::zero_grad(){
        //zeroing all grads
        for(Tensor* p : parameters){
            cudaMemset(p->d_grad, 0, p->size * sizeof(float));
        }
    }
}