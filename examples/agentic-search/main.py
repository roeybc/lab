"""Main script to run the agentic search benchmark."""

import asyncio
import os
from dotenv import load_dotenv
from src.benchmark import QABenchmark


async def main():
    # Load environment variables
    load_dotenv()

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY environment variable")
        print("You can create a .env file with: OPENAI_API_KEY=your_key_here")
        return

    print("🚀 Starting Agentic Search Benchmark")
    print("This will compare regular RAG vs agentic search on a QA dataset")

    # Initialize benchmark with 1000 samples
    benchmark = QABenchmark(sample_size=50)  # Start small for testing

    # Run benchmark with safer concurrency to avoid API rate limits
    results = await benchmark.run_benchmark(max_concurrent=30)

    # Print summary
    benchmark.print_summary_report(results)

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Save results
    benchmark.save_results(results, "results/benchmark_results.json")
    benchmark.save_results_csv(results, "results/benchmark_results.csv")

    print("\n✅ Benchmark completed!")


if __name__ == "__main__":
    asyncio.run(main())
