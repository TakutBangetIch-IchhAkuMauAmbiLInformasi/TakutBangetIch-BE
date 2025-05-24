# filepath: app/api/endpoints/paper.py
from fastapi import APIRouter, Depends, HTTPException
from app.models.search import SearchResult
from app.services.elasticsearch_service import ElasticsearchService
from app.services.deepseek_service import DeepSeekService
from typing import Dict
from datetime import datetime

router = APIRouter()

async def get_elasticsearch_service():
    service = ElasticsearchService()
    try:
        yield service
    finally:
        await service.close()

async def get_deepseek_service():
    return DeepSeekService()


@router.get("/{paper_id}", response_model=SearchResult)
async def get_paper_by_id(
    paper_id: str,
    es_service: ElasticsearchService = Depends(get_elasticsearch_service)
):
    """
    Get a paper by its ID
    """
    try:
        response = await es_service.search_by_id(paper_id)
        hits = response["hits"]["hits"]
        
        if not hits:
            raise HTTPException(status_code=404, detail=f"Paper with ID {paper_id} not found")
            
        hit = hits[0]
        source = hit["_source"]
        
        # Map Elasticsearch document to SearchResult model
        result = SearchResult(
            id=paper_id,
            title=source.get("title", ""),
            content=source.get("abstract", ""),  # Using abstract as the content field
            score=hit.get("_score", 0.0),
            passage=source.get("passage", None),
            metadata={
                "authors": source.get("authors", ""),
                "categories": source.get("categories", ""),
                "year": source.get("year", ""),
                "doi": source.get("doi", ""),
                "submitter": source.get("submitter", "")
            },
            highlights=None
        )
          # Make sure we're returning a paper result that matches the expected format
        paper_result = {
            "id": result.id,
            "title": result.title,
            "content": result.content,
            "score": result.score,
            "passage": result.passage,
            "metadata": result.metadata,
            "highlights": result.highlights
        }
        
        return SearchResult(**paper_result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error fetching paper: {str(e)}"
        )


@router.post("/{paper_id}/insights")
async def generate_paper_insights(
    paper_id: str,
    es_service: ElasticsearchService = Depends(get_elasticsearch_service),
    deepseek_service: DeepSeekService = Depends(get_deepseek_service)
):
    """
    Generate AI insights for a specific paper
    """
    try:
        # First, fetch the paper data
        response = await es_service.search_by_id(paper_id)
        hits = response["hits"]["hits"]
        
        if not hits:
            raise HTTPException(status_code=404, detail=f"Paper with ID {paper_id} not found")
            
        hit = hits[0]
        source = hit["_source"]
        
        # Create SearchResult object for the paper
        paper = SearchResult(
            id=paper_id,
            title=source.get("title", ""),
            content=source.get("abstract", ""),
            score=hit.get("_score", 0.0),
            passage=source.get("passage", None),
            metadata={
                "authors": source.get("authors", ""),
                "categories": source.get("categories", ""),
                "year": source.get("year", ""),
                "doi": source.get("doi", ""),
                "submitter": source.get("submitter", "")
            },
            highlights=None
        )
          # Generate insights using DeepSeek
        insights = await deepseek_service.generate_insights(paper)
        
        return {
            "paper_id": paper_id,
            "insights": insights,
            "generated_at": datetime.utcnow().isoformat() + "Z"
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating insights: {str(e)}"
        )
