#include <iostream>
#include <string>
#include "core/ops.cuh"

// Update the forward declarations to accept the arguments
void run_train(int argc, char** argv);
void run_infer(int argc, char** argv);
void run_compress(int argc, char** argv);
void run_get_model_summary(int argc, char** argv);

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./gpt_engine [train|infer|compress] [additional_args]" << std::endl;
        return 1;
    }

    ops::init_cublas();

    std::string mode = argv[1];

    // Pass the arguments into the specific modules
    if (mode == "train") {
        run_train(argc, argv);
    } else if (mode == "infer") {
        run_infer(argc, argv);
    } else if (mode == "compress") {
        run_compress(argc, argv);
    } else if (mode == "get_summary"){
        run_get_model_summary(argc, argv);
    } else {
        std::cerr << "CRITICAL ERROR: Unknown mode '" << mode << "'." << std::endl;
        std::cerr << "Available modes: train, infer, compress" << std::endl;
        ops::destroy_cublas();
        return 1;
    }
    
    ops::destroy_cublas();
    return 0;
}