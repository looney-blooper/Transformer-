#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include "optimizer.cuh"
#include "../model/gpt.cuh"

namespace CheckpointManagerNS {
    class checkpointManager {
    private:
        std::string save_dir;
        std::string telemetry_file;

    public:
        checkpointManager(const std::string& dir);

        //
        void save_checkpoint(model::GPT& model, core::AdamW& optimizer, int epoch, int step, float loss, float lr);
        //
        bool load_latest(model::GPT& model, core::AdamW& optimizer, int& out_epoch, int& out_step);        
        //
        void log_telemetry(int epoch, int step, float loss, float lr);
    };
}