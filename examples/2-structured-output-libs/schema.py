from typing import List, Optional, Literal
from datetime import date
from pydantic import BaseModel, Field, conlist, field_validator

Priority = Literal["low", "medium", "high", "urgent"]

class Task(BaseModel):
    description: str = Field(..., description="Short imperative description of the task")
    priority: Priority
    owner: Optional[str] = Field(None, description="Handle or name of the person responsible")
    tags: conlist(str, max_length=8) = Field(default_factory=list)
    deadline: Optional[date] = None
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, v):
        if not v:
            return []
        out, seen = [], set()
        for t in v:
            if not isinstance(t, str):
                continue
            t = t.strip().lower()
            if t and t not in seen:
                seen.add(t); out.append(t)
        return out
