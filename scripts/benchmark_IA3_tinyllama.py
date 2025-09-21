import subprocess

# Your model paths on Kaggle
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PEFT_MODEL_PATH = "safffrron/prompt_tuning_IA3"  # Replace with actual dataset name
TASKS = "mmlu,arc_easy,hellaswag"  
DEVICE = "cuda"  
BATCH_SIZE = "auto"
MAX_BATCH_SIZE = "64"

cmd = [
    "accelerate", "launch", "-m", "lm_eval",
    "--model", "hf",
    "--model_args", f"pretrained={BASE_MODEL_NAME},peft={PEFT_MODEL_PATH},trust_remote_code=True",
    "--tasks", TASKS,
    "--device", DEVICE,
    "--batch_size", BATCH_SIZE,
    "--max_batch_size", MAX_BATCH_SIZE,
    "--output_path", "./benchmarks/p_tunedv2_results.json",
]

subprocess.run(cmd, check=True)