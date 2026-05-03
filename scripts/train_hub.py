import os
import time
import threading
import subprocess
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import HfApi, snapshot_download

# ==========================================
# CONFIGURATION
# ==========================================
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" 
HF_REPO_ID = "mithun017/cpp-transformer-weights" 
HF_TOKEN = os.getenv("HF_TOKEN")
SYNC_INTERVAL_SECONDS = 300  # Wakes up every 5 minutes to backup to HF

COMPILE_CMD = ["nvcc", "-O3",
               "src/main.cu",
               "src/core/ops.cu","src/core/optimizer.cu","src/core/checkpoint.cu",
               "src/model/gpt.cu","src/model/transformer.cu", 
               "src/layers/attention.cu","src/layers/loss.cu","src/layers/modules.cu", 
               "src/data/tokenizer.cpp", "src/data/dataloader.cpp",
               "src/modes/train.cu","src/modes/infer.cu","src/modes/compress.cu", "src/modes/get_model_summary.cu",
               "-lcublas", "-o", "gpt_engine"]

training_is_running = False

def run_cmd(cmd):
    print(f"\n>>> Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def background_cloud_sync():
    """Runs in parallel with C++ to continuously save artifacts to Hugging Face."""
    api = HfApi()
    
    while training_is_running:
        time.sleep(SYNC_INTERVAL_SECONDS)
        
        # Don't try to sync if training finished while we were sleeping
        if not training_is_running:
            break 
            
        print("\n[ BACKGROUND SYNC ] Securing latest checkpoints & vocab to Cloud...")
        try:
            # 1. Sync Vocab and Telemetry
            root_files = ["vocab.bin", "training_history.csv", "loss_curve.png"]
            for file in root_files:
                if os.path.exists(file):
                    api.upload_file(
                        path_or_fileobj=file, path_in_repo=file,
                        repo_id=HF_REPO_ID, repo_type="model", token=HF_TOKEN
                    )
                    print(f"Downloaded file : {file}")
            
            # 2. Sync Checkpoints Folder
            if os.path.exists("checkpoints") and os.path.isdir("checkpoints"):
                api.upload_folder(
                    folder_path="checkpoints", path_in_repo="checkpoints",
                    repo_id=HF_REPO_ID, repo_type="model", token=HF_TOKEN
                )
        except Exception as e:
            print(f"[ BACKGROUND SYNC FAILED ] Will retry next cycle. Error: {e}")

def main():
    global training_is_running
    
    # 1. Download Dataset
    if not os.path.exists("input.txt"):
        print("\n[ DOWNLOADING DATASET ]")
        print(f"Fetching from: {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, "input.txt")
    else:
        print("\n[ DATASET DETECTED: Skipping Download ]")

    # 1.5 PULL EXISTING CHECKPOINTS (Must happen BEFORE training!)
    if HF_TOKEN:
        print("\n[ PULLING CLOUD CHECKPOINTS ]")
        try:
            snapshot_download(
                repo_id=HF_REPO_ID,
                allow_patterns=["checkpoints/*", "vocab.bin", "training_history.csv"],
                local_dir=".",
                token=HF_TOKEN
            )
            print("Successfully restored brain state from Hugging Face.")
        except Exception as e:
            print("No existing checkpoints found in the cloud (or error syncing). Starting fresh.")
    
    # 2. Compile Engine
    print("\n[ COMPILING C++ ENGINE ]")
    run_cmd(COMPILE_CMD)

    # 3. Ignite Training & Background Sync Thread
    print("\n[ IGNITING TRAINING LOOP & TELEMETRY THREAD ]")
    
    if HF_TOKEN:
        training_is_running = True
        sync_thread = threading.Thread(target=background_cloud_sync)
        sync_thread.daemon = True # Ensures thread dies if the main script is hard-killed
        sync_thread.start()
    
    try:
        # Blocks until C++ finishes or Colab crashes
        run_cmd(["./gpt_engine", "train"])
    except subprocess.CalledProcessError:
        print("\n[ CRITICAL ] C++ Engine crashed or was interrupted!")
    finally:
        # Stop the background thread whether we succeeded or crashed
        training_is_running = False 
        if HF_TOKEN:
            sync_thread.join(timeout=5)

    # 4. Generate the Loss Graph
    print("\n[ GENERATING TELEMETRY GRAPH ]")
    if os.path.exists("training_history.csv"):
        try:
            df = pd.read_csv("training_history.csv")
            plt.figure(figsize=(10, 6))
            
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

    # 5. Final Push
    if HF_TOKEN:
        print("\n[ PERFORMING FINAL CLOUD SYNC ]")
        try:
            api = HfApi()
            
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
            
            if os.path.exists("checkpoints") and os.path.isdir("checkpoints"):
                print("Uploading checkpoints directory...")
                api.upload_folder(
                    folder_path="checkpoints",
                    path_in_repo="checkpoints",
                    repo_id=HF_REPO_ID,
                    repo_type="model",
                    token=HF_TOKEN
                )
                
            print("Success! Final Brain, Vocab, and Telemetry secured in the cloud.")
        except Exception as e:
            print(f"Failed to upload to Hugging Face. Ensure your HF_TOKEN is valid. Error: {e}")
    else:
        print("\n[ SKIPPING CLOUD SYNC ] No HF_TOKEN environment variable detected.")

if __name__ == "__main__":
    main()