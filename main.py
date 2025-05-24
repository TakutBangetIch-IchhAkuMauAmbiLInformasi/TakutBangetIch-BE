from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import search, paper
from app.core.config import settings
from app.services.elasticsearch_service import ElasticsearchService
from dotenv import load_dotenv

load_dotenv(".env", override=True)

app = FastAPI(
    title="TakutBangetIch Search Engine API",
    description="A semantic search engine API",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use configured origins instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    search.router,
    prefix=settings.API_V1_STR,
    tags=["search"]
)

app.include_router(
    paper.router,
    prefix=f"{settings.API_V1_STR}/paper",
    tags=["paper"]
)

@app.get("/")
async def root():
    return {"message": "Welcome to TakutBangetIch Search Engine API"}

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    return {"status": "healthy", "service": "TakutBangetIch Search Engine API"}

# Optional debug endpoint (only for development)
@app.get("/debug/elasticsearch")
async def debug_elasticsearch():
    """Debug endpoint to check Elasticsearch connection"""
    try:
        es = ElasticsearchService()
        stats = await es.debug_index_stats()
        await es.close()
        return {
            "status": "connected", 
            "document_count": stats.get('document_count', {}).get('count', 0)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable (for Railway) or default to 8001
    port = int(os.getenv("PORT", 8001))
    
    # Start the server without running debug code automatically
    uvicorn.run(app, host="0.0.0.0", port=port)