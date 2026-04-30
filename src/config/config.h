#pragma once

namespace config {
    struct CONFIG {
        // 1. Architecture Hyperparameters (GPT-2 Small)
        int vocab_size = 50257;
        int d_model = 768;
        int num_heads = 12;
        int d_ff = 3072; // Set to 4 * d_model
        int num_layers = 12;
        int max_seq_len = 1024;

        // 2. Training Hyperparameters (Standard Pre-training/Replication Guidelines)
        int train_batch_size = 64; 
        int epochs = 1; // 1 epoch over a massive dataset (or ~9.6B tokens / steps)
        int save_every_n_steps = 1000;

        // 3. Optimizer Hyperparameters (AdamW)
        float lr = 0.0006f; // 6e-4 learning rate
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float weight_decay = 0.1f;
    };
}