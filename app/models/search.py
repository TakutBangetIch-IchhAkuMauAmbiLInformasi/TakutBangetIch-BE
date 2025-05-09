from pydantic import BaseModel
from typing import List, Optional

class SearchQuery(BaseModel):
    query: str
    limit: Optional[int] = 10
    offset: Optional[int] = 0

class SearchResult(BaseModel):
    id: str
    title: str
    content: str
    score: float
    metadata: Optional[dict] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: int
    query: str 