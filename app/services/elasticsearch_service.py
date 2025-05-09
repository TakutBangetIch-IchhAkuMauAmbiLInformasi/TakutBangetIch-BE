from elasticsearch import AsyncElasticsearch
from transformers import BertTokenizer, BertModel
import torch
from app.core.config import settings
import numpy as np

class ElasticsearchService:
    def __init__(self):
        # Configure Elasticsearch client with URL and API key
        self.es = AsyncElasticsearch(
            settings.ELASTICSEARCH_URL,
            api_key=settings.ELASTICSEARCH_API_KEY
        )
        # Initialize BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()  # Set to evaluation mode
        self.index_name = "takutbangetich"

    def get_bert_embedding(self, text: str) -> np.ndarray:
        """Generate BERT embeddings for the given text"""
        # Tokenize and prepare input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding as sentence representation
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        
        return embeddings[0]  # Return first (and only) embedding

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
                                "dims": 768  # BERT base dimension
                            },
                            "metadata": {"type": "object"}
                        }
                    }
                }
            )

    async def index_document(self, doc_id: str, title: str, content: str, metadata: dict = None):
        """Index a document with its BERT embedding"""
        # Generate BERT embedding for the content
        embedding = self.get_bert_embedding(content)
        
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
        """Search documents using BERT semantic search"""
        # Generate query embedding using BERT
        query_embedding = self.get_bert_embedding(query)
        
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