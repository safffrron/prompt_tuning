# --- CONFIGURATION ---
MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR="./benchmarks"
TASKS="mmlu,arc_easy,hellaswag"  # comma-separated list of tasks
DEVICE="cuda"                   # change to "cpu" if GPU not available
BATCH_SIZE="auto"
MAX_BATCH_SIZE=64
APPLY_CHAT_TEMPLATE="True"      # optional, recommended for chat models

# --- CREATE OUTPUT DIRECTORY ---
mkdir -p "$OUTPUT_DIR"

# --- RUN EVALUATION ---
accelerate launch -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL_NAME,trust_remote_code=True" \
    --tasks "$TASKS" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --max_batch_size "$MAX_BATCH_SIZE" \
    --output_path "$OUTPUT_DIR/base_tinyllama.json"
