"""Advanced RAG implementation with HyDE, hybrid search, and reranking."""

import asyncio
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import instructor
import openai
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document


# Prompt constants
HYDE_SYSTEM_PROMPT = "Generate a hypothetical document that would perfectly answer the given question. The document should be detailed, factual, and comprehensive."

HYDE_USER_PROMPT = "Question: {question}\n\nWrite a hypothetical document that would contain the answer:"

ANSWER_SYSTEM_PROMPT = "You are an expert analyst that excels at multi-hop reasoning and finding connections across documents. Focus on relevant information, ignore distractors, and provide precise answers with confidence assessment."

ANSWER_USER_PROMPT = """Context (ranked by relevance):
{context}

Question: {question}

Analysis Strategy:
1. **Scan for Relevance**: Identify which documents contain information directly related to the question
2. **Extract Key Facts**: Pull out the specific facts, dates, names, or relationships needed
3. **Connect Information**: If the answer requires combining facts from multiple documents, trace the logical connections
4. **Synthesize Answer**: Provide a clear, precise answer based on the evidence
5. **Assess Confidence**: Rate your confidence based on evidence quality and completeness

Provide your answer and confidence score (0-100)."""


class HydeDocument(BaseModel):
    """Hypothetical document for HyDE technique."""
    document: str = Field(description="A comprehensive hypothetical document that would answer the question")


class AnswerWithConfidence(BaseModel):
    """Structured answer with confidence score."""
    answer: str = Field(description="The answer to the question based on the provided context")
    confidence: int = Field(description="Confidence score from 0-100", ge=0, le=100)


class AdvancedRAGSearcher:
    """Advanced RAG with HyDE, hybrid search (cosine + BM25), and reranking."""

    def __init__(
        self,
        vector_store: FAISS,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        top_k: int = 20,
        final_k: int = 5,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-2-v2"
    ):
        self.vector_store = vector_store
        self.top_k = top_k  # Initial retrieval count
        self.final_k = final_k  # Final results after reranking

        # Initialize OpenAI client with instructor
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.client = instructor.from_openai(openai.AsyncOpenAI(api_key=api_key))
        self.model_name = model_name

        # Initialize reranker
        self.reranker = CrossEncoder(reranker_model)

        # Extract documents and create BM25 index
        self._setup_bm25_index()

    def _setup_bm25_index(self):
        """Setup BM25 index from vector store documents."""
        # Get all documents from vector store
        # Note: This is a simplified approach. In production, you'd want to persist this.
        self.documents = []
        self.doc_texts = []

        # Extract documents from FAISS vector store
        # This is a workaround since FAISS doesn't directly expose documents
        # In practice, you'd maintain a separate document store
        if hasattr(self.vector_store, 'docstore'):
            for doc_id in self.vector_store.index_to_docstore_id.values():
                doc = self.vector_store.docstore.search(doc_id)
                if doc:
                    self.documents.append(doc)
                    self.doc_texts.append(doc.page_content)

        # Create BM25 index
        tokenized_docs = [doc.split() for doc in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized_docs)

    def _setup_prompts(self):
        """Setup prompt templates."""

        self.hyde_prompt = ChatPromptTemplate.from_template("""
Generate a hypothetical document that would perfectly answer the following question.
The document should be detailed, factual, and comprehensive.

Question: {question}

Write a hypothetical document that would contain the answer:
""")

        self.rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer the question based solely on the provided context
- If the context doesn't contain enough information to answer the question, say so
- Be concise but comprehensive
- Cite specific parts of the context when relevant

Answer:
""")

    async def generate_hyde_document(self, question: str) -> HydeDocument:
        """Generate hypothetical document using HyDE technique."""
        response = await self.client.chat.completions.create(
            model=self.model_name,
            response_model=HydeDocument,
            messages=[
                {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                {"role": "user", "content": HYDE_USER_PROMPT.format(question=question)}
            ],
        )
        return response

    async def hybrid_retrieve(self, query: str, hyde_doc: str) -> List[Tuple[Document, float]]:
        """Simplified hybrid retrieval with score boosting for duplicate docs."""

        # 1. Get all retrieval results
        semantic_results = await asyncio.to_thread(
            self.vector_store.similarity_search_with_score, query, k=self.top_k
        )
        hyde_results = await asyncio.to_thread(
            self.vector_store.similarity_search_with_score, hyde_doc, k=self.top_k
        )

        # BM25 results
        bm25_scores = self.bm25.get_scores(query.split())
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:self.top_k]
        bm25_results = [
            (self.documents[idx], min(bm25_scores[idx] / 10.0, 1.0))
            for idx in top_bm25_indices if idx < len(self.documents)
        ]

        # 2. Combine with score boosting for duplicates
        doc_scores = {}
        for doc, score in semantic_results + hyde_results + bm25_results:
            doc_key = doc.page_content[:100]
            if doc_key in doc_scores:
                # Boost score if seen before (max 1.0)
                doc_scores[doc_key] = (doc, min(doc_scores[doc_key][1] + score * 0.3, 1.0))
            else:
                doc_scores[doc_key] = (doc, score)

        # 3. Return top results
        results = list(doc_scores.values())
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:self.top_k]

    async def rerank_documents(self, query: str, candidates: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Rerank documents using cross-encoder."""
        if not candidates:
            return candidates

        # Prepare pairs for reranking
        pairs = [(query, doc.page_content) for doc, _ in candidates]

        # Get reranker scores
        rerank_scores = await asyncio.to_thread(self.reranker.predict, pairs)

        # Combine with original documents
        reranked_results = [
            (doc, float(score))
            for (doc, _), score in zip(candidates, rerank_scores)
        ]

        # Sort by reranker score
        reranked_results.sort(key=lambda x: x[1], reverse=True)

        return reranked_results[:self.final_k]

    async def generate_answer_with_confidence(self, query: str, retrieved_docs: List[Tuple[Document, float]]) -> AnswerWithConfidence:
        """Generate answer with confidence score using retrieved documents."""
        # Prepare context from retrieved documents with clear relevance ranking
        context_parts = []
        for i, (doc, score) in enumerate(retrieved_docs):
            title = doc.metadata.get('title', f'Document {i+1}')
            context_parts.append(f"**Document {i+1}: {title}** (Relevance: {score:.3f})\n{doc.page_content.strip()}")

        context = "\n\n---\n\n".join(context_parts)

        response = await self.client.chat.completions.create(
            model=self.model_name,
            response_model=AnswerWithConfidence,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": ANSWER_USER_PROMPT.format(context=context, question=query)}
            ],
        )

        return response

    async def search(self, query: str) -> Dict[str, Any]:
        """Perform simplified advanced RAG search."""

        # Step 1: Direct semantic retrieval (no HyDE complexity)
        retrieved_docs = await asyncio.to_thread(
            self.vector_store.similarity_search_with_score, query, k=self.top_k
        )

        if not retrieved_docs:
            return {
                "answer": "No relevant documents found for your query.",
                "confidence": 0,
                "search_type": "advanced_rag"
            }

        # Step 2: Simple reranking using cross-encoder (the main "advanced" feature)
        reranked_docs = await self.rerank_documents(query, retrieved_docs)

        # Step 3: Generate final answer with confidence
        answer_response = await self.generate_answer_with_confidence(query, reranked_docs)

        return {
            "answer": answer_response.answer,
            "confidence": answer_response.confidence,
            "search_type": "advanced_rag"
        }