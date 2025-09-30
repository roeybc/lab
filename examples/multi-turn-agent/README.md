# Multi-Turn Agent Training Pipeline

Complete pipeline for training a multi-turn agent with search and report generation capabilities using SFT and GRPO on Qwen-7B.

## Overview

This example demonstrates:
1. **Dataset Generation**: Synthetic multi-turn conversations with tool calls (search + report generation)
2. **SFT Training**: Supervised fine-tuning on Qwen-7B using HuggingFace
3. **GRPO Training**: Group Relative Policy Optimization with validators and OpenAI judge

## Project Structure

```
multi-turn-agent/
generate_dataset.py    # Generate synthetic dataset with OpenAI
train_sft.py           # SFT training script
train_grpo.py          # GRPO training with validators
data/                  # Generated datasets
models/                # Trained model checkpoints
```

## Setup

Install dependencies with uv:

```bash
uv sync
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### 1. Generate Dataset

Generate 50 synthetic multi-turn conversations:

```bash
uv run python generate_dataset.py
```

This creates `data/multi_turn_agent_dataset.json` with conversations where the agent:
- Receives a research task
- Performs multiple searches
- Synthesizes information
- Generates a final report

### 2. Train with SFT

Supervised fine-tuning on Qwen-7B:

```bash
uv run python train_sft.py \
    --dataset_path data/multi_turn_agent_dataset.json \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --output_dir models/qwen-7b-agent-sft \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --use_lora True
```

**Key parameters:**
- `--use_lora`: Enable LoRA for efficient training (recommended)
- `--lora_r`: LoRA rank (default: 16)
- `--max_seq_length`: Maximum sequence length (default: 2048)

### 3. Train with GRPO

Reinforcement learning with validators and OpenAI judge:

```bash
uv run python train_grpo.py \
    --dataset_path data/multi_turn_agent_dataset.json \
    --model_name models/qwen-7b-agent-sft \
    --output_dir models/qwen-7b-agent-grpo \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-6 \
    --num_generations 4
```

**GRPO Validators:**
- `has_search`: Checks if agent uses search tool
- `multiple_searches`: Rewards thorough research (2+ searches)
- `has_report`: Checks if agent generates final report
- `correct_order`: Validates workflow (search � report)
- `llm_judge`: OpenAI evaluates quality (0-1 score)

**Reward Composition:**
- Tool usage correctness: 40%
- LLM quality judgment: 60%

## Tools Available to Agent

### search(query: str)
Search for information on a topic. Returns relevant search results.

### generate_report(title: str, content: str)
Generate a comprehensive report based on gathered information.

## Example Interaction

```
User: Research the environmental impact of electric vehicles vs hydrogen fuel cell vehicles

Agent: Let me search for information on both technologies.
[Uses: search("environmental impact electric vehicles")]
[Uses: search("environmental impact hydrogen fuel cell vehicles")]
[Uses: search("comparison EV vs hydrogen environmental footprint")]

Based on my research, I'll now generate a comprehensive report.
[Uses: generate_report(
    title="Environmental Impact: Electric Vehicles vs Hydrogen Fuel Cells",
    content="..."
)]
```

## Hardware Requirements

- **SFT**: 1x GPU with 24GB+ VRAM (or use LoRA with smaller GPU)
- **GRPO**: 1x GPU with 24GB+ VRAM
- **CPU Training**: Possible but very slow

For smaller GPUs, adjust:
- `--per_device_train_batch_size 1`
- `--gradient_accumulation_steps 16`
- `--use_lora True`

## Notes

- Dataset generation requires OpenAI API access
- GRPO judging uses OpenAI API (gpt-4o-mini)
- Models are saved with tokenizer for easy inference
- Training uses bfloat16 precision by default
- Gradient checkpointing enabled to reduce memory usage

## Extending

To add more tools:
1. Update `TOOLS` list in `generate_dataset.py`
2. Add tool execution logic in `generate_agent_conversation()`
3. Add validation logic in `validate_tool_usage()` in `train_grpo.py`
4. Update LLM judge prompt to evaluate new tools