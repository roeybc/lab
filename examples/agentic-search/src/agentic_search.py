"""Agentic search implementation with iterative refinement and convergence."""

import asyncio
import os
from typing import List, Dict, Any, Optional, Tuple

import instructor
import openai
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
import json


from typing import Optional

class AgentResponse(BaseModel):
    """Agent's response - either search or answer."""
    action_type: str = Field(description="Either 'search' or 'answer'")

    # Search fields (only used when action_type = 'search')
    query: Optional[str] = Field(default=None, description="Search query (only for search action)")

    # Answer fields (only used when action_type = 'answer')
    answer: Optional[str] = Field(default=None, description="Final answer (only for answer action)")
    confidence: Optional[int] = Field(default=None, description="Confidence 0-100 (only for answer action)")

# Agentic system prompt
AGENTIC_SYSTEM_PROMPT = """You are an expert researcher that answers complex questions requiring multi-hop reasoning.

You can either search for documents or provide a final answer.

Process:
1. Analyze the question and any context you have
2. If you need more information, set action_type='search' and provide a query
3. If you have sufficient information, set action_type='answer' with your answer and confidence

Focus on relevant information, ignore distractors, and be strategic about what you search for."""




class AgenticSearcher:
    """Agentic search with iterative refinement and convergence."""

    def __init__(
        self,
        vector_store: FAISS,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        top_k: int = 5,
        confidence_threshold: float = 75,
        max_iterations: int = 3
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.model_name = model_name

        # Initialize OpenAI clients
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.instructor_client = instructor.from_openai(openai.AsyncOpenAI(api_key=api_key))
        self.openai_client = openai.AsyncOpenAI(api_key=api_key)


    async def retrieve_documents(self, query: str) -> List[Tuple[Any, float]]:
        """Retrieve relevant documents for a query."""
        # Use LangChain's similarity search with scores (run in thread since it's sync)
        docs_with_scores = await asyncio.to_thread(
            self.vector_store.similarity_search_with_score, query, k=self.top_k
        )

        # Return documents with relevance scores (higher = more relevant)
        return [(doc, score) for doc, score in docs_with_scores]



    async def execute_search(self, query: str) -> str:
        """Execute a search and return formatted results."""
        retrieved_docs = await self.retrieve_documents(query)

        if not retrieved_docs:
            return "No relevant documents found for this query."

        # Format search results
        results = []
        for i, (doc, score) in enumerate(retrieved_docs):
            title = doc.metadata.get('title', f'Document {i+1}')
            results.append(f"**Source {i+1}: {title}** (Relevance: {score:.3f})\n{doc.page_content.strip()}")

        return "\n\n---\n\n".join(results)

    async def search(self, question: str) -> Dict[str, Any]:
        """Perform truly agentic search with instructor."""
        context = ""
        search_calls = []
        max_iterations = 5

        for iteration in range(max_iterations):
            # Prepare the prompt with current context
            user_prompt = f"Question: {question}"
            if context:
                user_prompt += f"\n\nContext from previous searches:\n{context}"
            user_prompt += "\n\nWhat would you like to do next?"

            # Get agent's decision using instructor
            response = await self.instructor_client.chat.completions.create(
                model=self.model_name,
                response_model=AgentResponse,
                messages=[
                    {"role": "system", "content": AGENTIC_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )

            if response.action_type == "search":
                # Execute search
                search_results = await self.execute_search(response.query)

                search_calls.append({
                    "query": response.query,
                    "results_preview": search_results[:200] + "..." if len(search_results) > 200 else search_results
                })

                # Add to context
                context += f"\n\nSearch '{response.query}':\n{search_results}"

            elif response.action_type == "answer":
                # Agent provided answer - check confidence threshold
                if response.confidence >= self.confidence_threshold:
                    # Confidence is high enough, return the answer
                    return {
                        "answer": response.answer,
                        "confidence": response.confidence,
                        "search_type": "agentic",
                        "query": question,
                        "search_calls": search_calls,
                        "total_searches": len(search_calls),
                        "converged": True
                    }
                else:
                    # Confidence too low, force another search
                    context += f"\n\nPrevious answer attempt: '{response.answer}' (confidence: {response.confidence}) - confidence too low, need more information."

        # If we hit max iterations, return the last attempt
        return {
            "answer": "Max iterations reached - insufficient confidence",
            "confidence": 0,
            "search_type": "agentic",
            "query": question,
            "search_calls": search_calls,
            "total_searches": len(search_calls),
            "converged": False
        }