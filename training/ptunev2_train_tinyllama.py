import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, PromptEncoderConfig, TaskType
from datasets import Dataset, concatenate_datasets
import json
import gc
import random
import os

# DISABLE WANDB COMPLETELY
os.environ["WANDB_DISABLED"] = "true"

# =============================================================================
# CONFIGURATION - All tunable parameters 
# =============================================================================
CONFIG = {
    # Data parameters
    "data_path": "/kaggle/input/amazon-converted/amazon_train_corpus.txt",
    "max_examples": 15000,   
    "max_length": 128,    
    
    # Model parameters  
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "num_virtual_tokens": 20,  
    
    # Training parameters
    "learning_rate": 1e-3,  
    "num_epochs": 3,        
    "batch_size": 2,        
    "eval_batch_size": 1,
    "warmup_steps": 100,     
    
    # Memory optimization
    "chunk_size": 500,      
    "train_split": 0.95,     
    
    # Output parameters
    "output_dir": "./p_tunedv2_tinyllama",
    "hub_model_id": "safffrron/prompt_tuning",
    "push_to_hub": False    
}

def load_data_efficiently(file_path, max_examples):
    """Load data with memory optimization for large files"""
    print(f"Loading up to {max_examples} examples from {file_path}")
    texts = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # For large files, read line by line to avoid loading all into memory
        lines = []
        for line_num, line in enumerate(f):
            if line.strip():
                lines.append(line.strip())
            if len(lines) >= max_examples * 2:  # Load 2x then sample
                break
    
    # Random sample if we have more than needed
    if len(lines) > max_examples:
        random.shuffle(lines)
        texts = lines[:max_examples]
        print(f"Randomly sampled {len(texts)} from {len(lines)} available examples")
    else:
        texts = lines
        print(f"Loaded {len(texts)} examples")
    
    return texts


def tokenize_in_chunks(texts, tokenizer, max_length, chunk_size):
    """Tokenize data in chunks to manage memory"""
    print(f"Tokenizing {len(texts)} examples in chunks of {chunk_size}")
    
    def tokenize_function(examples):
        # Tokenize with proper padding and truncation
        tokenized = tokenizer(
            examples['text'], 
            truncation=True, 
            padding='max_length',  # Pad to max_length consistently
            max_length=max_length,
            return_tensors=None  # Return lists, not tensors
        )
        # Labels are the same as input_ids for causal LM
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized
    
    tokenized_datasets = []
    
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}/{(len(texts)-1)//chunk_size + 1}")
        
        dataset_chunk = Dataset.from_dict({'text': chunk})
        tokenized_chunk = dataset_chunk.map(
            tokenize_function, 
            batched=True, 
            remove_columns=['text'],
            num_proc=1
        )
        tokenized_datasets.append(tokenized_chunk)
        
        # Memory cleanup
        del dataset_chunk, chunk
        gc.collect()
    
    # Combine all chunks
    final_dataset = concatenate_datasets(tokenized_datasets)
    del tokenized_datasets
    gc.collect()
    
    print(f"Tokenization complete. Final dataset size: {len(final_dataset)}")
    return final_dataset


def train_model():
    """Main training function"""
    print("Starting training with configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"], 
        torch_dtype=torch.float32,  # Use FP32 instead of FP16
        device_map="cuda", 
        low_cpu_mem_usage=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded successfully")
    
    # Configure PEFT
    peft_config = PromptEncoderConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=CONFIG["num_virtual_tokens"],
        encoder_hidden_size=CONFIG.get("encoder_hidden_size", 512),
        encoder_num_layers=2,   # Depth of prompt encoder MLP
        encoder_dropout=0.1,    # Optional
    )
    
    # Apply PEFT
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Memory cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load and process data
    print("\nLoading and processing data...")
    texts = load_data_efficiently(CONFIG["data_path"], CONFIG["max_examples"])
    dataset = tokenize_in_chunks(texts, tokenizer, CONFIG["max_length"], CONFIG["chunk_size"])
    
    # Clear texts from memory
    del texts
    gc.collect()
    
    # Split dataset
    train_size = int(CONFIG["train_split"] * len(dataset))
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, len(dataset)))
    
    print(f"Train examples: {len(train_dataset)}, Eval examples: {len(eval_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["eval_batch_size"],
        learning_rate=CONFIG["learning_rate"],
        warmup_steps=CONFIG["warmup_steps"],
        logging_steps=50,
        eval_strategy="epoch",  
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=CONFIG["push_to_hub"],
        hub_model_id=CONFIG["hub_model_id"],
        fp16=True,
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=True,
    )
    
    # Data collator and trainer
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=None,  # Don't pad to multiples
        return_tensors="pt"       # Return PyTorch tensors
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train and save
    print("\nStarting training...")
    
    try:
        trainer.train()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed: {e}")
        raise e
    
    print("Saving model...")
    trainer.save_model()
    
    if CONFIG["push_to_hub"]:
        print("Pushing to Hub...")
        trainer.push_to_hub()
    
    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    print("Training completed successfully!")
    return model, tokenizer


# Main execution
if __name__ == "__main__":
    # Train model
    model, tokenizer = train_model()

 
# =============================================================================
# UNCOMMENT FOR TESTING THE MODEL 
# =============================================================================

# def generate_text(model_path, prompt, max_new_tokens=100):
#     """Generate text using trained model"""
#     from peft import PeftModel
    
#     base_model = AutoModelForCausalLM.from_pretrained(CONFIG["model_name"])
#     model = PeftModel.from_pretrained(base_model, model_path)
#     tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
    
#     inputs = tokenizer(prompt, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=True,
#             temperature=0.7,
#             pad_token_id=tokenizer.eos_token_id
#         )
    
#     response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
#     return response


# # Test generation
# print("\nTesting generation...")
# response = generate_text(
#     CONFIG["output_dir"], 
#     "This product is amazing because", 
#     max_new_tokens=50
# )
# print(f"Generated text: {response}")