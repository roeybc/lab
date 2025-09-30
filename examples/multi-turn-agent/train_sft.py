"""
Supervised Fine-Tuning (SFT) for multi-turn agent on Qwen-7B.
Uses HuggingFace Transformers and TRL for training.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


@dataclass
class ScriptArguments:
    """Arguments for the SFT training script."""

    dataset_path: str = field(
        default="data/multi_turn_agent_dataset.json",
        metadata={"help": "Path to the dataset JSON file"}
    )
    model_name: str = field(
        default="Qwen/Qwen2.5-7B-Instruct",
        metadata={"help": "Model name or path"}
    )
    output_dir: str = field(
        default="models/qwen-7b-agent-sft",
        metadata={"help": "Output directory for the fine-tuned model"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=2,
        metadata={"help": "Training batch size per device"}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Gradient accumulation steps"}
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "Learning rate"}
    )
    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "Use flash attention 2"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Use LoRA for efficient fine-tuning"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha"}
    )


def load_and_prepare_dataset(dataset_path: str):
    """Load the dataset and convert to HF Dataset format."""

    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples")

    # Convert to format suitable for chat template
    formatted_data = []
    for example in data:
        # The messages already include user, assistant, and tool messages
        formatted_data.append({
            "messages": example["messages"],
            "topic": example["topic"]
        })

    return Dataset.from_list(formatted_data)


def format_chat_for_training(example, tokenizer):
    """Format messages using the chat template."""

    # Apply chat template
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text": text}


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Load tokenizer
    print(f"Loading tokenizer from {script_args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_and_prepare_dataset(script_args.dataset_path)

    # Format dataset
    print("Formatting dataset...")
    dataset = dataset.map(
        lambda x: format_chat_for_training(x, tokenizer),
        remove_columns=dataset.column_names
    )

    # Split into train/eval
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")

    # Load model
    print(f"Loading model {script_args.model_name}...")

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }

    if script_args.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        **model_kwargs
    )

    # Setup LoRA if enabled
    if script_args.use_lora:
        from peft import LoraConfig, get_peft_model

        print("Setting up LoRA...")
        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        num_train_epochs=script_args.num_train_epochs,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=True,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=script_args.max_seq_length,
        packing=False,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving model to {script_args.output_dir}...")
    trainer.save_model(script_args.output_dir)
    tokenizer.save_pretrained(script_args.output_dir)

    print("Training complete!")


if __name__ == "__main__":
    main()