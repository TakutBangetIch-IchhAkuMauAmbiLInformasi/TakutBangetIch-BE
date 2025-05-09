from elasticsearch import AsyncElasticsearch
from transformers import AutoTokenizer, AutoModel  # changed from BertTokenizer, BertModel
import torch
from app.core.config import settings
import numpy as np

class ElasticsearchService:
    def __init__(self):
        # Configure Elasticsearch client with URL and API key
        if settings.ELASTICSEARCH_API_KEY is not None:
            self.es = AsyncElasticsearch(
                settings.ELASTICSEARCH_URL,
                api_key=settings.ELASTICSEARCH_API_KEY
            )

            if self.es.ping():
                print("Elasticsearch connection successful")
        else:
            self.es = AsyncElasticsearch(
                settings.ELASTICSEARCH_URL
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
        self.model = AutoModel.from_pretrained(settings.MODEL_NAME)
        self.model.eval()
        self.index_name = settings.ELASTICSEARCH_INDEX_NAME

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embeddings for the given text using mean pooling"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state  # shape: [batch, seq_len, hidden_size]
            attention_mask = inputs['attention_mask'].unsqueeze(-1)  # shape: [batch, seq_len, 1]
            embeddings = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            embedding = embeddings[0].cpu().numpy()
        
        return embedding

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
                                "index": True,
                                "type": "dense_vector",
                                "similarity": "cosine",
                                "dims": 384 # Adjust based on your model's output size
                            },
                            "metadata": {"type": "object"}
                        }
                    }
                }
            )

    async def index_document(self, doc_id: str, title: str, content: str, metadata: dict = None):
        """Index a document with embedding"""
        embedding = self.get_embedding(content)
        
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
        """Hybrid BM25 + embedding re-ranking"""
        # generate query embedding
        query_embedding = self.get_embedding(query)
        
        # Hybrid: BM25 base + embedding-based re-ranking
        response = await self.es.search(
            index=self.index_name,
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^2", "content", "authors", "categories"],
                    }
                },
                "size": limit,
                "from": offset,
                "rescore": {
                    "window_size": 100,
                    "query": {
                        "rescore_query": {
                            "script_score": {
                                "query": { "match_all": {} },
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                    "params": {
                                        "query_vector": query_embedding.tolist()
                                    }
                                }
                            }
                        },
                        "query_weight": 1.0,
                        "rescore_query_weight": 2.0
                    }
                }
            }
        )
        
        return response
    
    async def index_exists(self) -> bool:
        """Check if the Elasticsearch index exists"""
        return await self.es.indices.exists(index=self.index_name)
    
    async def delete_index(self):
        """Delete the Elasticsearch index"""
        if await self.index_exists():
            await self.es.indices.delete(index=self.index_name)

    async def close(self):
        """Close the Elasticsearch connection"""
        await self.es.close()

if __name__ == "__main__":
    import asyncio
    # Index must exist (run index_data.py first)
    es_service = ElasticsearchService()
    response = asyncio.run(es_service.search("Embedded systems based on ARM processors", limit=1))
    print(response)
    asyncio.run(es_service.close())