import asyncio
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from app.models.search import SearchQuery, SearchResponse, SearchResult, SummarizeResult, QuerySummary, QuerySummaryResponse
from app.services.elasticsearch_service import ElasticsearchService
from app.services.deepseek_service import DeepSeekService
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

deepseek_service: DeepSeekService | None = None  # global reference

@asynccontextmanager
async def lifespan(app: FastAPI):
    global deepseek_service
    deepseek_service = DeepSeekService()
    print("DeepSeekService initialized")

    yield  

    print("DeepSeekService closed")

router = APIRouter(lifespan=lifespan)

async def get_elasticsearch_service():
    service = ElasticsearchService()
    try:
        yield service
    finally:
        await service.close()

async def get_deepseek_service() -> DeepSeekService:
    return deepseek_service

@router.post("/search", response_model=SearchResponse)
async def search(
    query: SearchQuery,
    es_service: ElasticsearchService = Depends(get_elasticsearch_service)
):
    """
    Perform hybrid search combining semantic and text-based search
    Returns results quickly without summary generation
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
                passage=hit["_source"]["passage"],  # Include passage as a top-level field
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
        
        # Return results immediately without summary
        return SearchResponse(
            results=results,
            total=response["hits"]["total"]["value"],
            query=query.query
        )
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

@router.get("/paper/{id}", response_model=SearchResult)
async def get_paper_by_id(
    id: str,
    es_service: ElasticsearchService = Depends(get_elasticsearch_service)
):
    """
    Get a specific paper by its ID
    """
    try:
        response = await es_service.search_by_id(id)
        
        # Check if we found the paper
        if response["hits"]["total"]["value"] == 0:
            raise HTTPException(status_code=404, detail=f"Paper with ID {id} not found")
        
        # Extract the first hit
        hit = response["hits"]["hits"][0]
        
        # Format the response as a SearchResult
        result = SearchResult(
            id=hit["_id"],
            title=hit["_source"]["title"],
            content=hit["_source"]["abstract"],
            score=hit["_score"],
            passage=hit["_source"]["passage"],
            metadata={
                "authors": hit["_source"]["authors"],
                "categories": hit["_source"]["categories"],
                "doi": hit["_source"]["doi"],
                "year": hit["_source"]["year"],
                "submitter": hit["_source"]["submitter"]
            }
        )
        
        return result
    except HTTPException:
        # Re-raise HTTPExceptions (like 404s)
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary",response_model=SummarizeResult)
async def summary_top_k(
    doi: str,
    es_service: ElasticsearchService = Depends(get_elasticsearch_service),
    deepseek_service: DeepSeekService = Depends(get_deepseek_service)
):
    """
    Summarize its paper using DeepSeek API
    """
    try:
        context={}
        response = await es_service.search_by_doi(doi)
        doc = response["hits"]["hits"].pop()["_source"]
        context["abstract"] = doc["abstract"]
        context["title"] = doc["title"]
        context["passage"] = doc["passage"]  # Include the full document passage
        return await deepseek_service.summarize(context)
         
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/summarize-query", response_model=QuerySummaryResponse)
async def generate_summary(
    query_data: QuerySummary,
    es_service: ElasticsearchService = Depends(get_elasticsearch_service),
    deepseek_service: DeepSeekService = Depends(get_deepseek_service)
):
    """
    Generate a summary of search results for a given query
    This is a separate endpoint to allow asynchronous summary generation
    """
    try:
        # Create search query to get top k results
        search_query = SearchQuery(
            query=query_data.query,
            semantic_weight=0.7,
            text_weight=0.3,
            limit=3,  # Use top 3 results for summary
            offset=0
        )
        
        # Get search results
        response = await es_service.multi_search(
            query=search_query.query,
            semantic_weight=search_query.semantic_weight,
            text_weight=search_query.text_weight,
            limit=search_query.limit,
            offset=search_query.offset
        )
        
        hits = response["hits"]["hits"]
        results = [
            SearchResult(
                id=hit["_id"],
                title=hit["_source"]["title"],
                content=hit["_source"]["abstract"],
                score=hit["_score"],
                passage=hit["_source"]["passage"],  # Include passage as a top-level field 
                metadata={
                    "authors": hit["_source"]["authors"],
                    "categories": hit["_source"]["categories"],
                    "doi": hit["_source"]["doi"],
                    "year": hit["_source"]["year"],
                    "submitter": hit["_source"]["submitter"]
                }
            )
            for hit in hits
        ]
        
        # Generate summary
        summary = await deepseek_service.summarize(query_data.query, results)
        return {"summary": summary}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
