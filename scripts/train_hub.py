import os
import subprocess
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import HfApi, snapshot_download

# ==========================================
# CONFIGURATION
# ==========================================
# For Phase 1, we are moving to TinyStories. 
# Note: Full TinyStories is massive. If you already have your 660MB input.txt locally, 
# this script will detect it and skip the download.
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" 
HF_REPO_ID = "mithun017/cpp-transformer-weights" 
HF_TOKEN = os.getenv("HF_TOKEN")

# Using the wildcard approach prevents the "missing comma" concatenation bug
COMPILE_CMD = "nvcc -O3 main.cu src/core/*.cu src/model/*.cu src/layers/*.cu src/modes/*.cu src/data/*.cpp -lcublas -o gpt_engine"

def run_cmd(cmd, use_shell=False):
    print(f"\n>>> Executing: {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    subprocess.run(cmd, check=True, shell=use_shell)

def main():
    # 1. Download Dataset (Skips if you already have the massive input.txt)
    if not os.path.exists("input.txt"):
        print("\n[ DOWNLOADING DATASET ]")
        print(f"Fetching from: {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, "input.txt")
    else:
        print("\n[ DATASET DETECTED: Skipping Download ]")
    
    # 2. Compile Engine
    print("\n[ COMPILING C++ ENGINE ]")
    run_cmd(COMPILE_CMD, use_shell=True)

    # 3. Train
    print("\n[ IGNITING TRAINING LOOP ]")
    run_cmd(["./gpt_engine", "train"], use_shell=False)

    # 4. Generate the Loss Graph
    print("\n[ GENERATING TELEMETRY GRAPH ]")
    if os.path.exists("training_history.csv"):
        try:
            df = pd.read_csv("training_history.csv")
            plt.figure(figsize=(10, 6))
            
            # Dynamically check if we have a 'Step' column (preferred) or fallback to 'Epoch'
            x_axis = 'Step' if 'Step' in df.columns else 'Epoch'
            
            plt.plot(df[x_axis], df['Loss'], marker='', color='b', linewidth=2)
            plt.title('Training Loss Curve')
            plt.xlabel(x_axis)
            plt.ylabel('Cross-Entropy Loss')
            plt.grid(True)
            plt.savefig("loss_curve.png")
            print("Generated loss_curve.png")
        except Exception as e:
            print(f"Warning: Could not generate graph. Error: {e}")
    else:
        print("Warning: training_history.csv not found. Skipping graph generation.")

    # 5. Upload to Hugging Face
    if HF_TOKEN:
        print("\n[ SYNCING CLOUD CHECKPOINTS ]")
        try:
            # This pulls the checkpoints folder and vocab file into your local directory
            snapshot_download(
                repo_id=HF_REPO_ID,
                allow_patterns=["checkpoints/*", "vocab.bin"],
                local_dir=".",
                token=HF_TOKEN
            )
            print("Successfully restored brain state from Hugging Face.")
        except Exception as e:
            print("No existing checkpoints found in the cloud (or error syncing). Starting fresh.")

            
        print("\n[ PUSHING ARTIFACTS TO HUGGING FACE ]")
        try:
            api = HfApi()
            
            # A. Upload the root files
            root_files = ["vocab.bin", "training_history.csv", "loss_curve.png"]
            for file in root_files:
                if os.path.exists(file):
                    api.upload_file(
                        path_or_fileobj=file,
                        path_in_repo=file,
                        repo_id=HF_REPO_ID,
                        repo_type="model",
                        token=HF_TOKEN
                    )
            
            # B. Upload the entire checkpoints directory
            if os.path.exists("checkpoints") and os.path.isdir("checkpoints"):
                print("Uploading checkpoints directory...")
                api.upload_folder(
                    folder_path="checkpoints",
                    path_in_repo="checkpoints",
                    repo_id=HF_REPO_ID,
                    repo_type="model",
                    token=HF_TOKEN
                )
                
            print("Success! Brain, Vocab, and Telemetry secured in the cloud.")
        except Exception as e:
            print(f"Failed to upload to Hugging Face. Ensure your HF_TOKEN is valid. Error: {e}")
    else:
        print("\n[ SKIPPING CLOUD SYNC ] No HF_TOKEN environment variable detected.")

if __name__ == "__main__":
    main()