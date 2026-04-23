import os
import subprocess
import urllib.request
from huggingface_hub import HfApi

# Configuration
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
HF_REPO_ID = "mithun017/cpp-transformer-weights" 
HF_TOKEN = os.getenv("HF_TOKEN") # Pull securely from environment variables

COMPILE_CMD = ["nvcc", "-O3",
               "src/main.cu",
               "src/core/ops.cu","src/core/optimizer.cu",
               "src/model/gpt.cu","src/model/transformer.cu", 
               "src/layers/attention.cu","src/layers/loss.cu","src/layers/modules.cu", 
               "src/data/tokenizer.cpp", "src/data/dataloader.cpp",
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

    # 4. Upload to Hugging Face
    if HF_TOKEN:
        print("\n[ PUSHING ARTIFACT TO HUGGING FACE ]")
        api = HfApi()
        api.upload_file(
            path_or_fileobj="gpt2_weights.bin",
            path_in_repo="gpt2_weights.bin",
            repo_id=HF_REPO_ID,
            repo_type="model",
            token=HF_TOKEN
        )
        print("Success! Brain secured in the cloud.")
    else:
        print("\n[ WARNING: HF_TOKEN not found in environment. Skipping upload. ]")

if __name__ == "__main__":
    main()