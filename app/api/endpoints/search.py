from fastapi import APIRouter, Depends, HTTPException
from app.models.search import SearchQuery, SearchResponse, SearchResult
from app.services.elasticsearch_service import ElasticsearchService
from typing import List, Dict, Optional

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
    Perform hybrid search combining semantic and text-based search
    """
    try:
        response = await es_service.multi_search(
            query=query.query,
            semantic_weight=query.semantic_weight,
            text_weight=query.text_weight,
            author=query.author,
            category=query.category,
            year=query.year,
            limit=query.limit,
            offset=query.offset
        )
        
        hits = response["hits"]["hits"]
        results = [
            SearchResult(
                id=hit["_id"],
                title=hit["_source"]["title"],
                content=hit["_source"]["abstract"],
                score=hit["_score"],
                metadata={
                    "authors": hit["_source"].get("authors", "Unknown"),
                    "categories": hit["_source"].get("categories", ""),
                    "doi": hit["_source"].get("doi", ""),
                    "year": hit["_source"].get("year", ""),
                    "submitter": hit["_source"].get("submitter", "")
                },
                highlights=hit.get("highlight")
            )
            for hit in hits
        ]
        
        return SearchResponse(
            results=results,
            total=response["hits"]["total"]["value"],
            query=query.query
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/autocomplete")
async def autocomplete(
    prefix: str,
    limit: int = 5,
    es_service: ElasticsearchService = Depends(get_elasticsearch_service)
):
    """
    Get title suggestions for autocomplete
    """
    try:
        response = await es_service.autocomplete_title(prefix, limit)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/categories")
async def get_categories(
    es_service: ElasticsearchService = Depends(get_elasticsearch_service)
):
    """
    Get all available categories
    """
    try:
        response = await es_service.get_categories()
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/author/{author}")
async def search_by_author(
    author: str,
    limit: int = 10,
    offset: int = 0,
    es_service: ElasticsearchService = Depends(get_elasticsearch_service)
):
    """
    Search for papers by a specific author
    """
    try:
        response = await es_service.search_by_author(author, limit, offset)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/doi/{doi}")
async def search_by_doi(
    doi: str,
    es_service: ElasticsearchService = Depends(get_elasticsearch_service)
):
    """
    Search for a paper by its DOI
    """
    try:
        response = await es_service.search_by_doi(doi)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/paper/{paper_id}", response_model=SearchResult)
async def get_paper_by_id(
    paper_id: str,
    es_service: ElasticsearchService = Depends(get_elasticsearch_service)
):
    """
    Get a single paper by its ID
    """
    try:
        # Create a simple filter query to find the paper by ID
        response = await es_service.search_by_id(paper_id)
        
        if not response or not response["hits"]["hits"]:
            raise HTTPException(status_code=404, detail=f"Paper with ID {paper_id} not found")
            
        hit = response["hits"]["hits"][0]
        result = SearchResult(
            id=hit["_id"],
            title=hit["_source"]["title"],
            content=hit["_source"]["abstract"],
            score=1.0,  # Default score since we're fetching by ID
            metadata={
                "authors": hit["_source"].get("authors", "Unknown"),
                "categories": hit["_source"].get("categories", ""),
                "doi": hit["_source"].get("doi", ""),
                "year": hit["_source"].get("year", ""),
                "submitter": hit["_source"].get("submitter", "")
            }
        )
        return result
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))