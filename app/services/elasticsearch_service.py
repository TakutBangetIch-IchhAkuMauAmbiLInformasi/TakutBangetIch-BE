from elasticsearch import AsyncElasticsearch
from sentence_transformers import SentenceTransformer
from app.core.config import settings
import numpy as np

class ElasticsearchService:
    def __init__(self):
        self.es = AsyncElasticsearch(
            hosts=[f"http://{settings.ELASTICSEARCH_HOST}:{settings.ELASTICSEARCH_PORT}"],
            basic_auth=(settings.ELASTICSEARCH_USERNAME, settings.ELASTICSEARCH_PASSWORD) if settings.ELASTICSEARCH_USERNAME else None
        )
        self.model = SentenceTransformer(settings.MODEL_NAME)
        self.index_name = "takutbangetich"

    async def create_index(self):
        """Create the Elasticsearch index with proper mappings"""
        if not await self.es.indices.exists(index=self.index_name):
            await self.es.indices.create(
                index=self.index_name,
                body={
                    "mappings": {
                        "properties": {
                            "title": {"type": "text"},
                            "content": {"type": "text"},
                            "embedding": {
                                "type": "dense_vector",
                                "dims": 384  # Dimension for all-MiniLM-L6-v2
                            },
                            "metadata": {"type": "object"}
                        }
                    }
                }
            )

    async def index_document(self, doc_id: str, title: str, content: str, metadata: dict = None):
        """Index a document with its embedding"""
        # Generate embedding for the content
        embedding = self.model.encode(content)
        
        document = {
            "title": title,
            "content": content,
            "embedding": embedding.tolist(),
            "metadata": metadata or {}
        }
        
        await self.es.index(
            index=self.index_name,
            id=doc_id,
            body=document
        )

    async def search(self, query: str, limit: int = 10, offset: int = 0):
        """Search documents using semantic search"""
        # Generate query embedding
        query_embedding = self.model.encode(query)
        
        # Perform semantic search
        response = await self.es.search(
            index=self.index_name,
            body={
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": query_embedding.tolist()}
                        }
                    }
                },
                "size": limit,
                "from": offset
            }
        )
        
        return response

    async def close(self):
        """Close the Elasticsearch connection"""
        await self.es.close() 