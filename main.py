from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import search
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
    allow_origins=["*"],  # In production, replace with specific origins
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

@app.get("/")
async def root():
    return {"message": "Welcome to TakutBangetIch Search Engine API"}

# ...existing code...

if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    async def debug_and_run():
        es = ElasticsearchService()
        stats = await es.debug_index_stats()
        print(stats['document_count']['count'])
        
        await es.es.close()  # Don't forget to close the connection
    
    asyncio.run(debug_and_run())
    uvicorn.run(app, host="0.0.0.0", port=8001)