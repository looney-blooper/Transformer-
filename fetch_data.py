import os
from datasets import load_dataset

# Chinchilla Optimal Target for 8.2M parameters: ~660 MB
TARGET_SIZE_MB = 660
TARGET_SIZE_BYTES = TARGET_SIZE_MB * 1024 * 1024
OUTPUT_FILE = "input.txt"

print(f"[DATA PIPELINE] Streaming TinyStories from HuggingFace...")
print(f"[DATA PIPELINE] Target extraction size: {TARGET_SIZE_MB} MB")

# We stream the dataset so we don't blow up Colab's RAM
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

current_size = 0
story_count = 0

with open(OUTPUT_FILE, "w",encoding="utf-8") as f:
    for item in dataset:
    # Clean the text and append a structural EOS (End of Sequence) token
        text = item["text"].strip() + "\n<|endoftext|>\n" 
        
        f.write(text)
        current_size += len(text.encode("utf-8"))
        story_count += 1
        
        # Print progress every 50 MB
        if current_size % (50 * 1024 * 1024) < 2000: 
            print(f" -> Downloaded: {current_size / (1024*1024):.2f} MB...")

        # Stop exactly when we hit the math-optimal limit
        if current_size >= TARGET_SIZE_BYTES:
            break

print(f"\n[SUCCESS] Extracted {story_count} synthetic stories.")
print(f"[SUCCESS] Dataset saved to {OUTPUT_FILE} ({current_size / (1024*1024):.2f} MB)")