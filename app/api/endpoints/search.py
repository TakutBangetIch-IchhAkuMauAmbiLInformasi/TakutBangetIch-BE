from fastapi import APIRouter, Depends
from app.models.search import SearchQuery, SearchResponse, SearchResult
from app.services.elasticsearch_service import ElasticsearchService
from typing import List

router = APIRouter()

async def get_elasticsearch_service():
    service = ElasticsearchService()
    try:
        yield service
    finally:
        await service.close()

@router.post("/search", response_model=SearchResponse)
async def search(
    query: SearchQuery,
    es_service: ElasticsearchService = Depends(get_elasticsearch_service)
):
    """
    Perform semantic search on the indexed documents
    """
    response = await es_service.search(
        query=query.query,
        limit=query.limit,
        offset=query.offset
    )
    
    hits = response["hits"]["hits"]
    results = [
        SearchResult(
            id=hit["_id"],
            title=hit["_source"]["title"],
            content=hit["_source"]["content"],
            score=hit["_score"],
            metadata=hit["_source"].get("metadata")
        )
        for hit in hits
    ]
    
    return SearchResponse(
        results=results,
        total=response["hits"]["total"]["value"],
        query=query.query
    ) 