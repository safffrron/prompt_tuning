import subprocess

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
TASKS = "mmlu,arc_easy,hellaswag"  
DEVICE = "cuda"  
BATCH_SIZE = "auto"
MAX_BATCH_SIZE = "64"

cmd = [
    "accelerate", "launch", "-m", "lm_eval",
    "--model", "hf",
    "--model_args", f"pretrained={MODEL_NAME},trust_remote_code=True",
    "--tasks", TASKS,
    "--device", DEVICE,
    "--batch_size", BATCH_SIZE,
    "--max_batch_size", MAX_BATCH_SIZE,
    "--output_path", "./benchmarks/base_qwen_4b.json",
]

subprocess.run(cmd, check=True)