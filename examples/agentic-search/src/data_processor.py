"""Data processing module for document ingestion and chunking."""

import os
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
import tiktoken
import faiss
import numpy as np
import pickle


class DocumentProcessor:
    """Handles document loading, chunking, and embedding generation."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-large"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

    def _get_directory_hash(self, directory: str) -> str:
        """Generate hash of directory contents for caching."""
        hash_md5 = hashlib.md5()
        directory_path = Path(directory)

        for file_path in sorted(directory_path.rglob("*.txt")):
            try:
                with open(file_path, 'rb') as f:
                    hash_md5.update(f.read())
                hash_md5.update(str(file_path).encode())
            except Exception:
                continue

        return hash_md5.hexdigest()

    def _should_rebuild_vector_store(self, directory: str, vector_store_path: str) -> bool:
        """Check if vector store needs to be rebuilt."""
        if not os.path.exists(vector_store_path):
            return True

        # Check if hash file exists
        hash_file = vector_store_path + ".hash"
        if not os.path.exists(hash_file):
            return True

        # Compare current hash with stored hash
        current_hash = self._get_directory_hash(directory)
        try:
            with open(hash_file, 'r') as f:
                stored_hash = f.read().strip()
            return current_hash != stored_hash
        except Exception:
            return True

    def _save_directory_hash(self, directory: str, vector_store_path: str):
        """Save directory hash for future comparison."""
        current_hash = self._get_directory_hash(directory)
        hash_file = vector_store_path + ".hash"
        with open(hash_file, 'w') as f:
            f.write(current_hash)

    def load_documents_from_directory(self, directory: str) -> List[Document]:
        """Load text documents from a directory."""
        documents = []
        directory_path = Path(directory)

        for file_path in directory_path.rglob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    doc = Document(
                        page_content=content,
                        metadata={"source": str(file_path), "filename": file_path.name}
                    )
                    documents.append(doc)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        chunks = []
        for doc in documents:
            doc_chunks = self.text_splitter.split_documents([doc])
            for i, chunk in enumerate(doc_chunks):
                chunk.metadata.update({
                    "chunk_id": f"{doc.metadata.get('filename', 'unknown')}_{i}",
                    "chunk_index": i,
                    "total_chunks": len(doc_chunks)
                })
            chunks.extend(doc_chunks)
        return chunks

    def create_embeddings(self, chunks: List[Document]) -> np.ndarray:
        """Generate embeddings for document chunks."""
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embeddings.embed_documents(texts)
        return np.array(embeddings)

    def create_vector_store(self, chunks: List[Document], embeddings: np.ndarray) -> Dict[str, Any]:
        """Create a FAISS vector store."""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))

        return {
            "index": index,
            "chunks": chunks,
            "embeddings": embeddings,
            "metadata": {
                "embedding_model": self.embedding_model,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            }
        }

    def save_vector_store(self, vector_store: Dict[str, Any], filepath: str):
        """Save vector store to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(vector_store, f)

    def load_vector_store(self, filepath: str) -> Dict[str, Any]:
        """Load vector store from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def process_documents(self, directory: str, vector_store_path: str = "vector_store.pkl") -> Dict[str, Any]:
        """Complete document processing pipeline with intelligent caching."""

        # Check if we need to rebuild the vector store
        if not self._should_rebuild_vector_store(directory, vector_store_path):
            print(f"Loading existing vector store from {vector_store_path}")
            return self.load_vector_store(vector_store_path)

        print("Building new vector store...")
        print("Loading documents...")
        documents = self.load_documents_from_directory(directory)
        print(f"Loaded {len(documents)} documents")

        print("Chunking documents...")
        chunks = self.chunk_documents(documents)
        print(f"Created {len(chunks)} chunks")

        print(f"Generating embeddings using {self.embedding_model}...")
        embeddings = self.create_embeddings(chunks)
        print(f"Generated embeddings with shape {embeddings.shape}")

        print("Creating vector store...")
        vector_store = self.create_vector_store(chunks, embeddings)

        print(f"Saving vector store to {vector_store_path}")
        self.save_vector_store(vector_store, vector_store_path)
        self._save_directory_hash(directory, vector_store_path)

        return vector_store