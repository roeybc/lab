"""
GRPO (Group Relative Policy Optimization) training for multi-turn agent.
Uses validators to check correctness of tool usage and report generation.
OpenAI API is used as a judge for quality assessment.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import torch
from datasets import Dataset
from openai import OpenAI
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import GRPOConfig, GRPOTrainer

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@dataclass
class ScriptArguments:
    """Arguments for GRPO training script."""

    dataset_path: str = field(
        default="data/multi_turn_agent_dataset.json",
        metadata={"help": "Path to the dataset JSON file"}
    )
    model_name: str = field(
        default="models/qwen-7b-agent-sft",
        metadata={"help": "Path to the SFT model"}
    )
    output_dir: str = field(
        default="models/qwen-7b-agent-grpo",
        metadata={"help": "Output directory for the GRPO model"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Training batch size per device"}
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "Gradient accumulation steps"}
    )
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "Learning rate"}
    )
    num_generations: int = field(
        default=4,
        metadata={"help": "Number of generations per prompt for GRPO"}
    )


# Define the tools for validation
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for information on a given topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": "Generate a comprehensive report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["title", "content"]
            }
        }
    }
]


def extract_tool_calls(text: str) -> List[Dict]:
    """Extract tool calls from generated text."""

    tool_calls = []

    # Look for function call patterns in the text
    # This is a simplified parser - in practice, you'd use the tokenizer's tool parsing
    search_pattern = r'search\(["\'](.+?)["\']\)'
    report_pattern = r'generate_report\(["\'](.+?)["\']\s*,\s*["\'](.+?)["\']\)'

    for match in re.finditer(search_pattern, text, re.DOTALL):
        tool_calls.append({
            "function": "search",
            "arguments": {"query": match.group(1)}
        })

    for match in re.finditer(report_pattern, text, re.DOTALL):
        tool_calls.append({
            "function": "generate_report",
            "arguments": {
                "title": match.group(1),
                "content": match.group(2)
            }
        })

    return tool_calls


def validate_tool_usage(generated_text: str, topic: str) -> Dict[str, float]:
    """Validate that the agent uses tools correctly."""

    scores = {}

    # Extract tool calls
    tool_calls = extract_tool_calls(generated_text)

    # Check if search was used
    search_calls = [tc for tc in tool_calls if tc["function"] == "search"]
    scores["has_search"] = 1.0 if len(search_calls) > 0 else 0.0

    # Prefer multiple searches (more thorough research)
    scores["multiple_searches"] = 1.0 if len(search_calls) >= 2 else 0.5 if len(search_calls) == 1 else 0.0

    # Check if report generation was used
    report_calls = [tc for tc in tool_calls if tc["function"] == "generate_report"]
    scores["has_report"] = 1.0 if len(report_calls) > 0 else 0.0

    # Report should be generated after searches (good workflow)
    if search_calls and report_calls:
        # Simple heuristic: check if search appears before report in text
        first_search_pos = generated_text.find("search(")
        first_report_pos = generated_text.find("generate_report(")

        if first_report_pos > first_search_pos:
            scores["correct_order"] = 1.0
        else:
            scores["correct_order"] = 0.0
    else:
        scores["correct_order"] = 0.0

    return scores


def judge_quality_with_llm(generated_text: str, topic: str) -> float:
    """Use OpenAI API to judge the quality of the agent's response."""

    prompt = f"""You are evaluating an AI agent's performance on a research task.

Topic: {topic}

Agent's response:
{generated_text}

Evaluate the agent's response on the following criteria:
1. Did it search for relevant information?
2. Did it synthesize the information well?
3. Did it generate a comprehensive report?
4. Was the workflow logical (search -> analyze -> report)?
5. Overall quality and coherence

Provide a score from 0 to 1 (where 1 is perfect).
Respond with ONLY a number between 0 and 1, no explanation."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=10
        )

        score_text = response.choices[0].message.content.strip()
        score = float(score_text)
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]

    except Exception as e:
        print(f"Error in LLM judging: {e}")
        return 0.5  # Default score on error


def compute_reward(generated_text: str, topic: str) -> float:
    """Compute the reward for a generated response."""

    # Validate tool usage
    tool_scores = validate_tool_usage(generated_text, topic)

    # Get LLM judge score
    llm_score = judge_quality_with_llm(generated_text, topic)

    # Combine scores
    # Tool usage correctness: 40%
    # LLM quality judgment: 60%
    tool_score_avg = sum(tool_scores.values()) / len(tool_scores)
    final_reward = 0.4 * tool_score_avg + 0.6 * llm_score

    print(f"Tool scores: {tool_scores}")
    print(f"LLM score: {llm_score:.3f}")
    print(f"Final reward: {final_reward:.3f}")

    return final_reward


def load_and_prepare_dataset(dataset_path: str):
    """Load the dataset for GRPO training."""

    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples")

    # For GRPO, we need prompts that the model will complete
    # We'll use just the initial user message as the prompt
    formatted_data = []
    for example in data:
        # Find the first user message
        user_message = None
        for msg in example["messages"]:
            if msg["role"] == "user":
                user_message = msg["content"]
                break

        if user_message:
            formatted_data.append({
                "prompt": user_message,
                "topic": example["topic"],
            })

    return Dataset.from_list(formatted_data)


class RewardFunction:
    """Reward function that wraps the compute_reward logic."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, prompts: List[str], generations: List[str], topics: List[str]) -> List[float]:
        """Compute rewards for a batch of generations."""

        rewards = []
        for prompt, generation, topic in zip(prompts, generations, topics):
            reward = compute_reward(generation, topic)
            rewards.append(reward)

        return rewards


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

    # Split into train/eval
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")

    # Load model
    print(f"Loading model from {script_args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Create reward function
    reward_fn = RewardFunction(tokenizer)

    # GRPO configuration
    grpo_config = GRPOConfig(
        output_dir=script_args.output_dir,
        num_train_epochs=script_args.num_train_epochs,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        logging_steps=5,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        max_length=script_args.max_seq_length,
        num_generations=script_args.num_generations,
        temperature=0.7,
        top_p=0.9,
        remove_unused_columns=False,
    )

    # Custom function to prepare prompts
    def prepare_prompt(example):
        messages = [
            {"role": "system", "content": "You are a helpful research agent with access to search and report generation tools."},
            {"role": "user", "content": example["prompt"]}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return {"query": prompt, "topic": example["topic"]}

    # Prepare datasets
    train_dataset = train_dataset.map(prepare_prompt)
    eval_dataset = eval_dataset.map(prepare_prompt)

    # Create GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_functor=lambda prompts, completions: reward_fn(
            prompts,
            completions,
            [train_dataset[i]["topic"] for i in range(len(prompts))]
        ),
    )

    # Train
    print("Starting GRPO training...")
    trainer.train()

    # Save final model
    print(f"Saving model to {script_args.output_dir}...")
    trainer.save_model(script_args.output_dir)
    tokenizer.save_pretrained(script_args.output_dir)

    print("GRPO training complete!")


if __name__ == "__main__":
    main()