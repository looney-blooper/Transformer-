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
};