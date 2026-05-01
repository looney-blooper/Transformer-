import os
import subprocess
import urllib.request
from huggingface_hub import HfApi

import pandas as pd
import matplotlib.pyplot as plt

# Configuration
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
HF_REPO_ID = "mithun017/cpp-transformer-weights" 
HF_TOKEN = os.getenv("HF_TOKEN") # Pull securely from environment variables

COMPILE_CMD = ["nvcc", "-O3",
               "src/main.cu",
               "src/core/ops.cu","src/core/optimizer.cu","src/core/checkpoint.cu"
               "src/model/gpt.cu","src/model/transformer.cu", 
               "src/layers/attention.cu","src/layers/loss.cu","src/layers/modules.cu", 
               "src/data/tokenizer.cpp", "src/data/dataloader.cpp",
               "src/modes/train.cu","src/modes/infer.cu","src/modes/compress.cu", "src/modes/get_model_summary.cu",
               "-lcublas", "-o", "gpt_engine"]

def run_cmd(cmd):
    print(f"\n>>> Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    # 1. Download Dataset
    if not os.path.exists("input.txt"):
        print("\n[ DOWNLOADING DATASET ]")
        urllib.request.urlretrieve(DATA_URL, "input.txt")
    
    # 2. Compile Engine
    print("\n[ COMPILING C++ ENGINE ]")
    # Adjust path if main.cu is inside src/
    run_cmd(COMPILE_CMD)

    # 3. Train
    print("\n[ IGNITING TRAINING LOOP ]")
    run_cmd(["./gpt_engine", "train"])

    # NEW STEP 4: Generate the Loss Graph
    print("\n[ GENERATING TELEMETRY GRAPH ]")
    if os.path.exists("training_history.csv"):
        df = pd.read_csv("training_history.csv")
        plt.figure(figsize=(10, 6))
        plt.plot(df['Epoch'], df['Loss'], marker='', color='b', linewidth=2)
        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        ylabel = 'Cross-Entropy Loss'
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.savefig("loss_curve.png")
        print("Generated loss_curve.png")

    # STEP 5: Upload to Hugging Face
    if HF_TOKEN:
        print("\n[ PUSHING ARTIFACTS TO HUGGING FACE ]")
        api = HfApi()
        
        # Upload the artifacts we just created
        files_to_upload = ["gpt2_weights.bin", "vocab.bin", "training_history.csv", "loss_curve.png"]
        
        for file in files_to_upload:
            if os.path.exists(file):
                api.upload_file(
                    path_or_fileobj=file,
                    path_in_repo=file,
                    repo_id=HF_REPO_ID,
                    repo_type="model",
                    token=HF_TOKEN
                )
        print("Success! Brain, Vocab, and Telemetry secured in the cloud.")

if __name__ == "__main__":
    main()