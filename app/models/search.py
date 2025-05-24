from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class SearchQuery(BaseModel):
    query: str
    return_passage: Optional[bool] = Field(default=False, description="Whether to return the full passage content")
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
    passage: Optional[str] = None  # Full content of the PDF document
    metadata: Optional[Dict[str, Any]] = None
    highlights: Optional[Dict[str, List[str]]] = None

class SearchResponse(BaseModel):
    summary: Optional[str] = None
    results: List[SearchResult]  # Each result includes title, abstract, passage, and metadata
    total: int
    query: str

class SummarizeResult(BaseModel):
    id: str
    content: str
    passage: Optional[str] = None  # Full content of the PDF document
    metadata: Optional[Dict[str, Any]] = None
    response_metadata: Optional[Dict[str, Any]] = None
    usage_metadata: Optional[Dict[str, Any]] = None
    
class QuerySummary(BaseModel):
    query: str
    
class QuerySummaryResponse(BaseModel):
    summary: str

class ChatResponse(BaseModel):
    message: str
    response: str
    status: str = "success"

class ChatRequest(BaseModel):
    message: str