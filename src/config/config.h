#pragma once

namespace config {
    struct CONFIG {
        
        // 1. Architecture Hyperparameters (GPT-Nano) -- 8M params
        int vocab_size = 10000;    // Compact vocabulary (e.g., specific character or small BPE)
        int d_model = 256;         // Hidden dimension
        int num_heads = 4;         // Attention heads
        int d_ff = 1024;           // Feed-forward dimension (d_model * 4)
        int num_layers = 4;        // Transformer blocks
        int max_seq_len = 512;     // Context window length

        // 2. Training Hyperparameters
        int train_batch_size = 32; // Low memory footprint
        int gradient_accumulation_steps = 1;
        int epochs = 1;           // Fast convergence over smaller datasets
        int save_every_n_steps = 100;

        // 3. Optimizer Hyperparameters
        float lr = 0.001f;         // Standard learning rate
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float weight_decay = 0.01f;
        
        /*
        // 1. Architecture Hyperparameters (GPT-Mini) -- 36M params
        int vocab_size = 50257;    // Standard byte-level BPE vocabulary size
        int d_model = 512;         // Hidden dimension size
        int num_heads = 8;         // Attention heads
        int d_ff = 2048;           // Feed-forward dimension (4 * d_model)
        int num_layers = 6;        // Transformer blocks
        int max_seq_len = 512;     // Context window length

        // 2. Training Hyperparameters (T4 VRAM Optimized)
        int train_batch_size = 16;         // Per-device batch size
        int gradient_accumulation_steps = 4; // Effective batch size: 64 (16 * 4)
        int epochs = 5;                    
        int save_every_n_steps = 250;

        // 3. Optimizer Hyperparameters
        float lr = 0.0005f;        // 5e-4 learning rate
        float beta1 = 0.9f;
        float beta2 = 0.98f;       // Adjusted beta2 for stability
        float weight_decay = 0.1f;
        */
    };
}

/*

=================================================================
                  MODEL ARCHITECTURE SUMMARY                   
=================================================================
Vocab Size:         10000
Context Window:     512 tokens
Embedding Dim:      256
Attention Heads:    4
FeedForward Dim:    1024
Transformer Layers: 4
-----------------------------------------------------------------
MODULE                             PARAMETERS     % OF TOTAL
-----------------------------------------------------------------
Embeddings (Token + Positional)    2691072        32.46%
Multi-Head Attention               1052672        12.70%
Feed Forward Network               2102272        25.36%
Layer Normalization                4608           0.06%
LM Head (Untied) / Other           2438928        29.42%
-----------------------------------------------------------------
TOTAL PARAMETERS                   8,289,552        100.00%
-----------------------------------------------------------------
PHYSICAL RAM SIZE                  31.62 MB (FP32)
=================================================================



=================================================================
                  MODEL ARCHITECTURE SUMMARY                   
=================================================================
Vocab Size:         50257
Context Window:     512 tokens
Embedding Dim:      512
Attention Heads:    8
FeedForward Dim:    2048
Transformer Layers: 6
-----------------------------------------------------------------
MODULE                             PARAMETERS     % OF TOTAL
-----------------------------------------------------------------
Embeddings (Token + Positional)    25993728       33.88%
Multi-Head Attention               6303744        8.22%
Feed Forward Network               12598272       16.42%
Layer Normalization                13312          0.02%
LM Head (Untied) / Other           31824465       41.47%
-----------------------------------------------------------------
TOTAL PARAMETERS                   76,733,521       100.00%
-----------------------------------------------------------------
PHYSICAL RAM SIZE                  292.72 MB (FP32)
=================================================================


*/