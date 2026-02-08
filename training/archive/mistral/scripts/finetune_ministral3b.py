#!/usr/bin/env python3
"""
Finetune Ministral 3B for Josi
Uses same training data as 8B model
"""

import torch
from transformers import (
    Mistral3ForConditionalGeneration,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# Config
MODEL_NAME = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
OUTPUT_DIR = "./models/mivalta-josi-ministral3b"
TRAIN_FILE = "./data/gold_combined.jsonl"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = Mistral3ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

# LoRA config
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

print("Applying LoRA...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("Loading dataset...")
dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")

def format_example(example):
    # Extract from messages format
    messages = example['messages']
    user_msg = ""
    assistant_msg = ""
    for msg in messages:
        if msg['role'] == 'user':
            user_msg = msg['content']
        elif msg['role'] == 'assistant':
            assistant_msg = msg['content']
    text = f"[INST] {user_msg} [/INST] {assistant_msg}"
    return {"text": text}

dataset = dataset.map(format_example)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
        padding="max_length",
    )

dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# Training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_steps=50,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("Starting training...")
trainer.train()

print("Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Done!")
