#pragma once
#include <vector>
#include "tensor.cuh"


namespace core {
    class AdamW {
    public:
        std::vector<Tensor*> parameters;
        
        //memory state
        std::vector<Tensor*> m_state; // momentum state
        std::vector<Tensor*> v_state; // varience state


        //Hyper parameters
        float lr;
        float weight_decay;
        float beta1;
        float beta2;
        float eps;
        int step_count;

        AdamW(std::vector<Tensor*> params, float lr = 3e-4f, float beta1 = 0.9f, float beta2 = 0.999f, float weight_decay = 0.01f);
        ~AdamW();

        //Updates the weights on the GPU
        void step();


        //Zeros out the d_grads before moving to next step; 
        void zero_grad();

        //Serialization for checkpointing
        void save_state(const std::string& filepath);
        void load_state(const std::string& filepath);
    
    };
}