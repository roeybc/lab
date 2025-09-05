from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field, PrivateAttr
import faiss
from typing import List, Tuple
from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
client = AsyncOpenAI()

class MemoryStore(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"
    
    def model_post_init(self, __context) -> None:
        # Manual assignment without PrivateAttr
        self._texts = []
        self._emb_dim = 384
        self._index = faiss.IndexFlatL2(self._emb_dim)
        self._embedder = SentenceTransformer(self.model_name)

    async def save(self, text: str):
        output = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"give me a short summary of the following text"},
                    {"role":"user","content":text}]
        )
    
        summary = output.choices[0].message.content

        emb = self._embedder.encode([summary])
        self._index.add(emb)
        self._texts.append(summary)

    def search(self, query: str, k: int = 3) -> List[str]:
        if not self._texts: return []
        emb = self._embedder.encode([query])
        D, I = self._index.search(emb, min(k, len(self._texts)))
        return [self._texts[i] for i in I[0]]

class ShortTermMemory(BaseModel):
    window: int = 6
    messages: List[Tuple[str, str]] = Field(default_factory=list)
    def append(self, role: str, content: str):
        self.messages.append((role, content))
    def last_window(self):
        return self.messages[-self.window:]