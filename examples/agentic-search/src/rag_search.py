"""Regular RAG search implementation."""

import asyncio
import os
from typing import List, Dict, Any, Optional, Tuple

import openai
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document


# Prompt constants
SYSTEM_PROMPT = "You are a helpful assistant that answers questions based on the provided context. Focus on relevant information and ignore distractors. Use step-by-step reasoning to connect information from multiple sources when needed."

USER_PROMPT = """Context:
{context}

Question: {question}

Instructions:
- Carefully read through all the provided context
- Focus on information that directly relates to the question - ignore irrelevant details
- Look for information that relates to the question
- If the question requires connecting information from multiple sources, do so step by step
- Provide a clear, factual answer based only on the relevant context
- If the context doesn't contain enough information to answer the question, say so clearly

Answer:"""


class RAGSearcher:
    """Traditional RAG search implementation."""

    def __init__(
        self,
        vector_store: FAISS,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        top_k: int = 5
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self.model_name = model_name

        # Initialize OpenAI client
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def retrieve_documents(self, query: str) -> List[Tuple[Any, float]]:
        """Retrieve relevant documents for a query."""
        # Use LangChain's similarity search with scores (run in thread since it's sync)
        docs_with_scores = await asyncio.to_thread(
            self.vector_store.similarity_search_with_score, query, k=self.top_k
        )

        # Return documents with relevance scores (higher = more relevant)
        return [(doc, score) for doc, score in docs_with_scores]

    async def generate_answer(self, query: str, retrieved_docs: List[Tuple[Document, float]]) -> Dict[str, Any]:
        """Generate answer using retrieved documents."""
        # Prepare context from retrieved documents
        context_parts = []
        for i, (doc, score) in enumerate(retrieved_docs):
            context_parts.append(f"[Document {i+1}] {doc.page_content}")

        context = "\n\n".join(context_parts)

        # Generate answer using OpenAI client
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(context=context, question=query)}
            ],
            temperature=0
        )

        return {
            "answer": response.choices[0].message.content,
            "retrieved_docs": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": score
                }
                for doc, score in retrieved_docs
            ],
            "context_used": context
        }

    async def search(self, query: str) -> Dict[str, Any]:
        """Perform complete RAG search."""
        # Retrieve relevant documents
        retrieved_docs = await self.retrieve_documents(query)

        if not retrieved_docs:
            return {
                "answer": "No relevant documents found for your query.",
                "retrieved_docs": [],
                "context_used": ""
            }

        # Generate answer
        result = await self.generate_answer(query, retrieved_docs)

        # Add metadata
        result.update({
            "search_type": "rag",
            "query": query,
            "num_retrieved": len(retrieved_docs)
        })

        return result