from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print(f"Loading model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # Quick test prompt
    prompt = "<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\nWhat is the capital of India?<|end|>\n<|assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)

    print("\n=== MODEL OUTPUT ===")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
