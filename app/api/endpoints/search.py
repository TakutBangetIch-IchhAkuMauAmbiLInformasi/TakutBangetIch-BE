from fastapi import APIRouter, Depends, FastAPI, HTTPException
from app.models.search import SearchQuery, SearchResponse, SearchResult, SummarizeResult
from app.services.elasticsearch_service import ElasticsearchService
from app.services.langchain_service import LangchainService
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

langchain_service: LangchainService | None = None  # global reference

@asynccontextmanager
async def lifespan(app: FastAPI):
    global langchain_service
    langchain_service = LangchainService()
    print("LangchainService initialized")

    yield  

    print("LangchainService closed")

router = APIRouter(lifespan=lifespan)

async def get_elasticsearch_service():
    service = ElasticsearchService()
    try:
        yield service
    finally:
        await service.close()

async def get_langchain_service() -> LangchainService:
    return langchain_service

        
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
                    "authors": hit["_source"]["authors"],
                    "categories": hit["_source"]["categories"],
                    "doi": hit["_source"]["doi"],
                    "year": hit["_source"]["year"],
                    "submitter": hit["_source"]["submitter"]
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
        print(response["hits"]["hits"])
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
    


@router.get("/summary",response_model=SummarizeResult)
async def search_by_doi(
    doi: str,
    es_service: ElasticsearchService = Depends(get_elasticsearch_service),
    lc_service: LangchainService = Depends(get_langchain_service)
):
    """
    Summarize its paper using Langchain Hugginface
    """
    try:
        context={}
        response = await es_service.search_by_doi(doi)
        doc = response["hits"]["hits"].pop()["_source"]
        context["abstract"] = doc["abstract"]
        context["title"] = doc["title"]
        return lc_service.summarize(context)
         
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
    
