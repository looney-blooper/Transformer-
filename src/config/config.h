#pragma once

namespace config {
    struct CONFIG {
        // 1. Architecture Hyperparameters
        int vocab_size = 300;
        int d_model = 64;
        int num_heads = 4;
        int d_ff = 256;
        int num_layers = 2;
        int max_seq_len = 128;

        // 2. Training Hyperparameters
        int train_batch_size = 1;
        int epochs = 150;
        int save_every_n_steps = 500;

        // 3. Optimizer Hyperparameters
        float lr = 0.001f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float weight_decay = 0.01f;
    };
}