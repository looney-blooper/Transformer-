#include <string>
#include <iomanip>
#include <filesystem>
#include "checkpoint.cuh"

namespace fs = std::filesystem;

namespace CheckpointManagerNS {
    checkpointManager::checkpointManager(const std::string &dir) {
        save_dir = dir;
        telemetry_file = save_dir + "/training_telemetry.csv";

        //checks if the dir exists
        if(!fs::exists(save_dir)) fs::create_directories(save_dir);

        // Initialize telemetry header if new
        if (!fs::exists(telemetry_file)) {
            std::ofstream out(telemetry_file);
            out << "Epoch,Step,Loss,LearningRate\n";
            out.close();
        }
    }

    void checkpointManager::log_telemetry(int epoch, int step, float loss, float lr) {
        std::ofstream out(telemetry_file, std::ios::app); // Append mode
        out << epoch << "," << step << "," << std::fixed << std::setprecision(6) << loss << "," << lr << "\n";
        out.close();
    }

    void checkpointManager::save_checkpoint(model::GPT& model, core::AdamW& optimizer, int epoch, int step, float loss, float lr) {
        std::string base_name = save_dir + "/ckpt_e" + std::to_string(epoch) + "_s" + std::to_string(step);
        
        model.save_pretrained(base_name + "_weights.bin");
        
        // --> ADD THIS: Dump the momentum and variance matrices
        optimizer.save_state(base_name + "_optim.bin"); 

        std::ofstream state_out(save_dir + "/latest.state");
        state_out << epoch << "\n" << step << "\n" << loss << "\n" << lr << "\n" << base_name << "\n";
        state_out.close();

        log_telemetry(epoch, step, loss, lr);
        std::cout << "[CHECKPOINT] State secured. Epoch " << epoch << " | Step " << step << std::endl;
    }

    bool checkpointManager::load_latest(model::GPT& model, core::AdamW& optimizer, int& out_epoch, int& out_step) {
        std::string state_file = save_dir + "/latest.state";
        if (!fs::exists(state_file)) {
            std::cout << "[CHECKPOINT] No existing state found. Starting fresh." << std::endl;
            return false;
        }

        std::ifstream state_in(state_file);
        float loss, lr;
        std::string base_name;

        // 1. Parse all metadata in one pass
        state_in >> out_epoch >> out_step >> loss >> lr >> base_name;
        state_in.close(); // Close the stream safely

        std::cout << "[CHECKPOINT] Found previous state! Resuming from Epoch " << out_epoch << ", Step " << out_step << std::endl;

        // 2. Load the physical weights using the base name
        std::cout << "--> Loading model weights..." << std::endl;
        model.load_pretrained(base_name + "_weights.bin");

        // 3. Inflate the optimizer memory to prevent amnesia
        std::cout << "--> Loading optimizer momentum and variance..." << std::endl;
        optimizer.load_state(base_name + "_optim.bin");

        std::cout << "[CHECKPOINT] Checkpoint fully recovered. Ready to resume matrix math." << std::endl;
        return true;
    }
}