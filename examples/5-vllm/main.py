# import torch
from vllm import LLM, SamplingParams
from dotenv import load_dotenv
# from transformers import AutoTokenizer

def main():
    llm = LLM(
    model=MODEL_NAME,
    max_model_len=2048,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.95,
    gpu_memory_utilization_threshold=0.95,
    gpu_memory_utilization_threshold_type="absolute",
    )

    llm.generate(
        "What is the capital of France?",
        SamplingParams(temperature=0.5, top_p=0.95),
    )
    print("Hello from 5-vllm!")



if __name__ == "__main__":
    main()
