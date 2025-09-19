# scripts/benchmark_base_tinyllama.py
import os
import json
import subprocess
import torch
from datetime import datetime

# 1. Auto-detect device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Ensure output folder exists
output_dir = "benchmarks_outputs"

# 3. Generate timestamped output file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(output_dir, f"tinyllama_base_{timestamp}.json")

# 4. Build command
command = [
    "lm_eval",
    "--model", "hf",
    "--model_args", "pretrained=TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "--tasks", "hellaswag",
    "--batch_size", "1",
    "--device", device,
    "--output_path", output_file
]

print(f"[INFO] Running evaluation... Results will be saved to {output_file}")

# 5. Run lm_eval
subprocess.run(command, check=True)

print(f"[INFO] Evaluation complete. Results saved in {output_file}")

