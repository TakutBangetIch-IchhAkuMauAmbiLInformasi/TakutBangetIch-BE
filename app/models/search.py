from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class SearchQuery(BaseModel):
    query: str
    semantic_weight: Optional[float] = Field(default=0.7, ge=0.0, le=1.0, description="Weight for semantic search component")
    text_weight: Optional[float] = Field(default=0.3, ge=0.0, le=1.0, description="Weight for text search component")
    author: Optional[str] = None
    category: Optional[str] = None
    year: Optional[str] = None
    limit: Optional[int] = Field(default=10, ge=1, le=100)
    offset: Optional[int] = Field(default=0, ge=0)

class SearchResult(BaseModel):
    id: str
    title: str
    content: str
    score: float
    metadata: Optional[Dict[str, Any]] = None
    highlights: Optional[Dict[str, List[str]]] = None

class SearchResponse(BaseModel):
    summary: str
    results: List[SearchResult]
    total: int
    query: str 

class SummarizeResult(BaseModel):
    id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    response_metadata: Optional[Dict[str, Any]] = None
    usage_metadata: Optional[Dict[str, Any]] = None