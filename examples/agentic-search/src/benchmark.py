"""Async benchmarking framework using dataset and vector stores."""

import asyncio
import json
import os
from typing import List, Dict, Any, Optional
import pandas as pd
from datasets import load_dataset
from tqdm.asyncio import tqdm

import instructor
import openai
from pydantic import BaseModel, Field
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from .rag_search import RAGSearcher
from .agentic_search import AgenticSearcher
from .advanced_rag import AdvancedRAGSearcher


class AnswerEvaluation(BaseModel):
    """Structured evaluation result."""
    is_correct: bool = Field(description="Whether the generated answer is factually correct")


class QABenchmark:
    """Benchmark using a QA dataset with LangChain vector stores."""

    def __init__(
        self,
        vector_store_path: str = "qa_vector_store",
        openai_api_key: Optional[str] = None,
        sample_size: int = 1000
    ):
        self.vector_store_path = vector_store_path
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.sample_size = sample_size

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=self.openai_api_key
        )

        # Initialize text splitter with token-based splitting
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=256,
            chunk_overlap=32
        )

        # Initialize judge client with instructor
        self.judge_client = instructor.from_openai(openai.AsyncOpenAI(api_key=self.openai_api_key))
        self._load_dataset()
        self._setup_vector_store()
        self._setup_searchers()


    def _load_dataset(self):
        """Load dataset."""
        print("Loading dataset...")
        dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")

        # Take validation split and sample
        validation_data = dataset["validation"]
        validation_data = validation_data.select(range(min(self.sample_size, len(validation_data))))

        self.questions = []
        self.contexts = []
        self.ground_truth_answers = []

        for item in validation_data:
            self.questions.append(item["question"])
            self.ground_truth_answers.append(item["answer"])

            # Combine all context documents - HotpotQA has parallel title/sentences arrays
            context_text = ""
            titles = item["context"]["title"]
            sentences_lists = item["context"]["sentences"]

            for title, sentences in zip(titles, sentences_lists):
                context_text += f"Title: {title}\n"
                context_text += " ".join(sentences) + "\n\n"

            self.contexts.append(context_text.strip())

    def _setup_vector_store(self):
        """Create LangChain vector store from contexts."""
        print("Setting up vector store from contexts...")

        # Check if vector store already exists
        if os.path.exists(self.vector_store_path):
            print("Loading existing vector store...")
            self.vector_store = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return

        print("Creating new vector store...")

        # Create documents from contexts
        documents = []
        for i, context in enumerate(self.contexts):
            # Split context by titles/paragraphs
            parts = context.split("Title: ")
            for j, part in enumerate(parts[1:]):  # Skip first empty part
                if part.strip():
                    lines = part.split("\n", 1)
                    title = lines[0] if lines else f"Document_{i}_{j}"
                    content = lines[1] if len(lines) > 1 else part
                    print(f"title: {title}")
                    print(f"question_id: {i}")
                    print(f"Content: {content}")

                    doc = Document(
                        page_content=content.strip(),
                        metadata={
                            "source": f"qa_{i}",
                            "title": title.strip(),
                            "question_id": i
                        }
                    )
                    documents.append(doc)

        # Split documents into chunks using LangChain
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")

        # Create FAISS vector store
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)

        # Save vector store
        self.vector_store.save_local(self.vector_store_path)
        print("Vector store created and saved")

    def _setup_searchers(self):
        """Initialize all searchers with LangChain vector store."""
        self.rag_searcher = RAGSearcher(
            self.vector_store,
            openai_api_key=self.openai_api_key,
            top_k=5
        )

        self.agentic_searcher = AgenticSearcher(
            self.vector_store,
            openai_api_key=self.openai_api_key,
            top_k=5,
            confidence_threshold=80,
            max_iterations=3
        )

        self.advanced_rag_searcher = AdvancedRAGSearcher(
            self.vector_store,
            openai_api_key=self.openai_api_key,
            top_k=8,
            final_k=5
        )

    async def evaluate_answer(self, question: str, ground_truth: str, generated_answer: str) -> bool:
        """Evaluate if generated answer is correct using structured output."""
        response = await self.judge_client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=AnswerEvaluation,
            messages=[
                {"role": "system", "content": "You are a precise fact-checker evaluating answer correctness. Focus on factual accuracy and completeness, not phrasing."},
                {"role": "user", "content": f"""Question: {question}
Ground Truth: {ground_truth}
Generated Answer: {generated_answer}

Evaluation Criteria:
1. Does the generated answer contain the same key facts as the ground truth?
2. Does it directly answer the question asked?
3. Is the information factually accurate?

Ignore minor differences in:
- Wording or phrasing
- Article usage (a/an/the)
- Capitalization or formatting
- Order of information (if all facts are present)

Consider INCORRECT if:
- Key facts are missing or wrong
- The answer is vague when specificity is required
- The answer doesn't address the actual question
- Contains factual errors or contradictions

Determine if the generated answer is factually correct and complete."""}
            ],
            temperature=0
        )
        return response.is_correct

    async def process_single_question(self, semaphore: asyncio.Semaphore, i: int, question: str, ground_truth: str) -> Dict[str, Any]:
        """Process a single question with concurrency limiting."""
        async with semaphore:
            print(f"\nQuestion {i+1}: {question}")

            # Get all three search results in parallel
            rag_task = asyncio.create_task(self.rag_searcher.search(question))
            agentic_task = asyncio.create_task(self.agentic_searcher.search(question))
            advanced_rag_task = asyncio.create_task(self.advanced_rag_searcher.search(question))

            rag_result, agentic_result, advanced_rag_result = await asyncio.gather(
                rag_task, agentic_task, advanced_rag_task
            )

            rag_answer = rag_result["answer"]
            agentic_answer = agentic_result["answer"]
            advanced_rag_answer = advanced_rag_result["answer"]

            # Evaluate all answers in parallel
            rag_eval_task = asyncio.create_task(self.evaluate_answer(question, ground_truth, rag_answer))
            agentic_eval_task = asyncio.create_task(self.evaluate_answer(question, ground_truth, agentic_answer))
            advanced_rag_eval_task = asyncio.create_task(self.evaluate_answer(question, ground_truth, advanced_rag_answer))

            rag_correct, agentic_correct, advanced_rag_correct = await asyncio.gather(
                rag_eval_task, agentic_eval_task, advanced_rag_eval_task
            )

            result = {
                "question_id": i,
                "question": question,
                "ground_truth": ground_truth,
                "rag": rag_correct,
                "agentic": agentic_correct,
                "advanced_rag": advanced_rag_correct
            }

            # Print quick summary
            print(f"RAG: {'✓' if rag_correct else '✗'}")
            print(f"Agentic: {'✓' if agentic_correct else '✗'}")
            print(f"Advanced RAG: {'✓' if advanced_rag_correct else '✗'}")

            return result

    async def run_benchmark(self, max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """Run complete benchmark with limited concurrency."""
        print(f"Running async benchmark on {len(self.questions)} questions (max {max_concurrent} concurrent)...")

        # Create semaphore to limit concurrent tasks
        semaphore = asyncio.Semaphore(max_concurrent)

        # Create tasks for all questions
        tasks = [
            self.process_single_question(semaphore, i, question, ground_truth)
            for i, (question, ground_truth) in enumerate(zip(self.questions, self.ground_truth_answers))
        ]

        # Run with progress bar
        results = []
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing questions"):
            result = await task
            results.append(result)

        # Sort results by question_id to maintain order
        results.sort(key=lambda x: x["question_id"])

        return results

    def generate_summary_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not results:
            return {}

        rag_correct = [r["rag"] for r in results]
        agentic_correct = [r["agentic"] for r in results]
        advanced_rag_correct = [r["advanced_rag"] for r in results]

        return {
            "total_questions": len(results),
            "rag_accuracy": sum(rag_correct) / len(rag_correct),
            "agentic_accuracy": sum(agentic_correct) / len(agentic_correct),
            "advanced_rag_accuracy": sum(advanced_rag_correct) / len(advanced_rag_correct)
        }

    def print_summary_report(self, results: List[Dict[str, Any]]):
        """Print formatted summary report."""
        summary = self.generate_summary_report(results)

        print(f"\n{'='*60}")
        print("QA BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Total Questions: {summary['total_questions']}")

        rag_correct = sum([r["rag"] for r in results])
        agentic_correct = sum([r["agentic"] for r in results])
        advanced_rag_correct = sum([r["advanced_rag"] for r in results])

        print(f"\nRAG: {summary['rag_accuracy']:.1%} ({rag_correct}/{summary['total_questions']})")
        print(f"Agentic: {summary['agentic_accuracy']:.1%} ({agentic_correct}/{summary['total_questions']})")
        print(f"Advanced RAG: {summary['advanced_rag_accuracy']:.1%} ({advanced_rag_correct}/{summary['total_questions']})")

        # Determine best performer
        best_accuracy = max(summary['rag_accuracy'], summary['agentic_accuracy'], summary['advanced_rag_accuracy'])
        if best_accuracy == summary['advanced_rag_accuracy']:
            print("\n🏆 Advanced RAG performs best!")
        elif best_accuracy == summary['agentic_accuracy']:
            print("\n🎉 Agentic search performs best!")
        else:
            print("\n📈 Simple RAG performs best!")

    def save_results(self, results: List[Dict[str, Any]], filepath: str):
        """Save benchmark results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {filepath}")

    def save_results_csv(self, results: List[Dict[str, Any]], filepath: str):
        """Save benchmark results to CSV file."""
        rows = []
        for r in results:
            rows.append({
                "question_id": r["question_id"],
                "question": r["question"],
                "ground_truth": r["ground_truth"],
                "rag_correct": r["rag"],
                "agentic_correct": r["agentic"],
                "advanced_rag_correct": r["advanced_rag"]
            })

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")