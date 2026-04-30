#pragma once

namespace config {
    struct CONFIG {
        /*
        // 1. Architecture Hyperparameters (GPT-Nano)
        int vocab_size = 10000;    // Compact vocabulary (e.g., specific character or small BPE)
        int d_model = 256;         // Hidden dimension
        int num_heads = 4;         // Attention heads
        int d_ff = 1024;           // Feed-forward dimension (d_model * 4)
        int num_layers = 4;        // Transformer blocks
        int max_seq_len = 512;     // Context window length

        // 2. Training Hyperparameters
        int train_batch_size = 32; // Low memory footprint
        int epochs = 10;           // Fast convergence over smaller datasets
        int save_every_n_steps = 100;

        // 3. Optimizer Hyperparameters
        float lr = 0.001f;         // Standard learning rate
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float weight_decay = 0.01f;
        */

        // 1. Architecture Hyperparameters (GPT-Mini Variant)
        int vocab_size = 50257;    // Standard byte-level BPE vocabulary size
        int d_model = 512;         // Hidden dimension size
        int num_heads = 8;         // Attention heads
        int d_ff = 2048;           // Feed-forward dimension (4 * d_model)
        int num_layers = 6;        // Transformer blocks
        int max_seq_len = 512;     // Context window length

        // 2. Training Hyperparameters
        int train_batch_size = 32; // Scaled for efficient memory usage
        int epochs = 5;            // Scaled for shorter experiments
        int save_every_n_steps = 250;

        // 3. Optimizer Hyperparameters
        float lr = 0.0005f;        // 5e-4 learning rate
        float beta1 = 0.9f;
        float beta2 = 0.98f;       // Adjusted beta2 for stability
        float weight_decay = 0.1f;
    };
}