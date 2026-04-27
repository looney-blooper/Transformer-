#pragma once
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <fstream>
#include <stdexcept>

// Macro for error checking CUDA calls
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

class Tensor {
public:
    std::vector<int> shape;                                     // store the shapes of matrix. eg {2,3,4} -> mat of shape 2*3*4
    int size;          // Total number of elements
    float* h_data;     // Host (CPU) pointer
    float* d_data;     // Device (GPU) pointer
    float* d_grad;     // Device (GPU) gradients for backprop

    // NEW: The INT8 Memory Pointers & Scale
    int8_t* h_data_int8 = nullptr;
    int8_t* d_data_int8 = nullptr;
    float quant_scale;
    bool is_quantized;


    // Constructor: Allocates memory on CPU and GPU
    Tensor(std::vector<int> s, bool requires_grad = true) {     
        shape = s;
        size = 1;
        for (int dim : shape) {
            size *= dim;
        }

        // Allocate Host Memory
        h_data = (float*)malloc(size * sizeof(float));
        
        // Allocate Device Memory
        CUDA_CHECK(cudaMalloc((void**)&d_data, size * sizeof(float)));
        
        // Allocate Gradient Memory if needed
        if (requires_grad) {
            CUDA_CHECK(cudaMalloc((void**)&d_grad, size * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_grad, 0, size * sizeof(float))); // Initialize gradients to 0
        } else {
            d_grad = nullptr;
        }
    }

    // Destructor: Frees memory to prevent VRAM leaks
    ~Tensor() {
        if (h_data) free(h_data);
        if (d_data) cudaFree(d_data);
        if (d_grad) cudaFree(d_grad);

        if (h_data_int8) { delete[] h_data_int8; h_data_int8 = nullptr; }
        if (d_data_int8) { cudaFree(d_data_int8); d_data_int8 = nullptr; }
    }

    // Move data from CPU to GPU
    void to_device() {
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Move data from GPU to CPU
    void to_host() {
        CUDA_CHECK(cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // Zero out gradients before the next backward pass
    void zero_grad() {
        if (d_grad) {
            CUDA_CHECK(cudaMemset(d_grad, 0, size * sizeof(float)));
        }
    }

    void save(std::ofstream& out) {
        // 1. Pull the absolute latest weights from VRAM to CPU RAM
        this->to_host(); 
        
        // 2. Write the size (as a safety check)
        out.write(reinterpret_cast<const char*>(&size), sizeof(int));
        
        // 3. Dump the raw float array directly to binary
        out.write(reinterpret_cast<const char*>(h_data), size * sizeof(float));
    }

    void load(std::ifstream& in) {
        // 1. Read the size check
        int file_size = 0;
        in.read(reinterpret_cast<char*>(&file_size), sizeof(int));
        
        if (file_size != this->size) {
            throw std::runtime_error("Architecture mismatch! The saved weights do not match this matrix.");
        }
        
        // 2. Read the raw floats directly into CPU RAM
        in.read(reinterpret_cast<char*>(h_data), size * sizeof(float));
        
        // 3. Blast the loaded weights into VRAM
        this->to_device();
    }

    void quantize_to_int8() {
        if (is_quantized) return;
        this->to_host(); // Ensure CPU has the latest FP32 weights

        // 1. Find the absolute maximum value in the tensor
        float max_val = 1e-8f; // Avoid division by zero
        for (int i = 0; i < size; i++) {
            if (std::abs(h_data[i]) > max_val) {
                max_val = std::abs(h_data[i]);
            }
        }

        // 2. Calculate the scaling factor
        quant_scale = 127.0f / max_val;

        // 3. Allocate INT8 RAM and Compress
        h_data_int8 = new int8_t[size];
        for (int i = 0; i < size; i++) {
            float scaled = h_data[i] * quant_scale;
            // Clip and round
            scaled = std::max(-127.0f, std::min(127.0f, scaled));
            h_data_int8[i] = static_cast<int8_t>(std::round(scaled));
        }

        is_quantized = true;

        // 4. Free the heavy FP32 memory
        delete[] h_data; h_data = nullptr;
        if (d_data) { cudaFree(d_data); d_data = nullptr; }
    }

    void save_int8(std::ofstream& out) {
        if (!is_quantized) quantize_to_int8();
        
        out.write(reinterpret_cast<const char*>(&size), sizeof(int));
        out.write(reinterpret_cast<const char*>(&quant_scale), sizeof(float)); // Save the key to un-compress it
        out.write(reinterpret_cast<const char*>(h_data_int8), size * sizeof(int8_t));
    }

    void load_int8(std::ifstream& in) {
        int file_size = 0;
        in.read(reinterpret_cast<char*>(&file_size), sizeof(int));
        
        // SAFETY CHECK: This is what will catch your 36KB truncated file!
        if (file_size != this->size) {
            throw std::runtime_error("CRITICAL: INT8 Architecture mismatch. File is corrupted or truncated.");
        }

        in.read(reinterpret_cast<char*>(&quant_scale), sizeof(float));
        
        // 1. Read the tiny INT8 bytes into CPU RAM
        if (!h_data_int8) h_data_int8 = new int8_t[size];
        in.read(reinterpret_cast<char*>(h_data_int8), size * sizeof(int8_t));
        
        // 2. INFLATION: Allocate FP32 memory and decompress
        if (!h_data) h_data = new float[size];
        for (int i = 0; i < size; i++) {
            // Reverse the scaling math
            h_data[i] = static_cast<float>(h_data_int8[i]) * quant_scale;
        }
        
        // 3. Blast the inflated, mathematically restored weights into VRAM
        this->to_device();
        is_quantized = true;
    }
};