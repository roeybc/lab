# Agentic Search Benchmark

This project benchmarks regular RAG (Retrieval-Augmented Generation) against agentic search using a QA dataset.

## Features

- **Regular RAG**: Traditional single-shot retrieval and generation
- **Agentic Search**: Iterative search with confidence-based convergence
- **LLM-as-a-Judge**: Binary correctness evaluation using GPT-4o-mini
- **Dataset**: HotpotQA multi-hop reasoning questions

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Set your OpenAI API key:
```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```

3. Run the benchmark:
```bash
uv run python main.py
```

## How It Works

### Regular RAG
1. Retrieves top-k documents for the query
2. Generates answer using retrieved context
3. Returns answer immediately

### Agentic Search
1. Generates initial answer with confidence score
2. If confidence < threshold, analyzes gaps in knowledge
3. Performs additional targeted searches
4. Iterates until confidence threshold met or max iterations reached
5. Returns final answer

### Evaluation
- Uses GPT-4o-mini as judge to compare generated answers with ground truth
- Binary evaluation: correct or incorrect
- Focuses on factual accuracy and completeness

## Results

The benchmark outputs:
- Accuracy comparison between RAG, advanced RAG and Agentic search
- Average iterations for agentic search
- Convergence rate statistics
- Detailed results in JSON and CSV format

## Architecture

- `src/data_processor.py`: Document processing and vector store creation
- `src/rag_search.py`: Traditional RAG implementation
- `src/agentic_search.py`: Iterative agentic search
- `src/benchmark.py`: benchmark framework
- `main.py`: Main execution script

## Dataset Attribution

This benchmark uses the **HotpotQA** dataset:
- **Source**: [HotpotQA dataset on Hugging Face](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
- **License**: CC BY-SA 4.0 (Creative Commons Attribution-ShareAlike 4.0 International)
- **Citation**: Yang, Zhilin, et al. "HotpotQA: A dataset for diverse, explainable multi-hop question answering." *arXiv preprint arXiv:1809.09600* (2018).
- **Description**: 113k Wikipedia-based question-answer pairs requiring multi-hop reasoning

The dataset is used for benchmarking purposes only and is not redistributed with this code.