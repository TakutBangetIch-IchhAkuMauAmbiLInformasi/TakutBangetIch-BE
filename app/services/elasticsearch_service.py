from elasticsearch import AsyncElasticsearch
from transformers import BertTokenizer, BertModel
import torch
from app.core.config import settings
import numpy as np
from typing import List, Dict, Optional, Union
import json

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
        self.index_name = "indexpesol"

    def get_bert_embedding(self, text: str) -> np.ndarray:
        """Generate BERT embeddings for the given text"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return embeddings[0]

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return float(dot_product / (norm1 * norm2))

    async def create_index(self):
        """Create the Elasticsearch index with proper mappings for arXiv papers"""
        if not await self.es.indices.exists(index=self.index_name):
            await self.es.indices.create(
                index=self.index_name,
                body={
                    "settings": {
                        "analysis": {
                            "analyzer": {
                                "author_analyzer": {
                                    "type": "custom",
                                    "tokenizer": "standard",
                                    "filter": ["lowercase", "trim"]
                                }
                            }
                        }
                    },
                    "mappings": {
                        "properties": {
                            "id": {"type": "keyword"},
                            "title": {
                                "type": "text",
                                "analyzer": "english",
                                "fields": {
                                    "keyword": {"type": "keyword"},
                                    "completion": {
                                        "type": "completion"
                                    }
                                }
                            },
                            "abstract": {
                                "type": "text",
                                "analyzer": "english"
                            },
                            "authors": {
                                "type": "text",
                                "analyzer": "author_analyzer",
                                "fields": {
                                    "keyword": {"type": "keyword"}
                                }
                            },
                            "categories": {
                                "type": "keyword"
                            },
                            "doi": {
                                "type": "keyword"
                            },
                            "submitter": {
                                "type": "keyword"
                            },
                            "year": {
                                "type": "keyword",
                                "fields": {
                                    "numeric": {
                                        "type": "integer"
                                    }
                                }
                            },
                            "embedding": {
                                "type": "dense_vector",
                                "dims": 768,
                                "index": True,
                                "similarity": "cosine"
                            }
                        }
                    }
                }
            )

    async def index_document(self, doc: Dict):
        """Index an arXiv paper document with its BERT embedding"""
        # Generate BERT embedding for combined title and abstract
        combined_text = f"{doc['title']} {doc['abstract']}"
        embedding = self.get_bert_embedding(combined_text)
        
        # Prepare document for indexing
        document = {
            "id": doc["id"],
            "title": doc["title"],
            "abstract": doc["abstract"],
            "authors": doc["authors"],
            "categories": doc["categories"],
            "doi": doc["doi"],
            "submitter": doc["submitter"],
            "year": doc["year"],
            "embedding": embedding.tolist()
        }
        
        await self.es.index(
            index=self.index_name,
            id=doc["id"],
            body=document
        )

    async def bulk_index(self, documents: List[Dict]):
        """Bulk index multiple documents"""
        operations = []
        for doc in documents:
            # Add operation type and metadata
            operations.append({"index": {"_index": self.index_name, "_id": doc["id"]}})
            # Add the document body as a separate action
            operations.append(doc)
        
        await self.es.bulk(body=operations)

    async def search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        limit: int = 10,
        offset: int = 0,
        semantic_weight: float = 0.7,
        text_weight: float = 0.3
    ):
        """
        Two-stage search:
        1. BM25 for initial retrieval
        2. Semantic re-ranking of top candidates
        """
        # 1. Initial BM25 retrieval
        initial_query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^3", "abstract^2"],
                                "type": "best_fields",
                                "operator": "or"
                            }
                        }
                    ],
                    "filter": self._build_filters(filters)
                }
            },
            "highlight": {
                "fields": {
                    "title": {},
                    "abstract": {}
                }
            },
            "size": 100  # Get more candidates for re-ranking
        }

        # Get initial results
        initial_response = await self.es.search(
            index=self.index_name,
            body=initial_query
        )

        hits = initial_response["hits"]["hits"]
        if not hits:
            return initial_response

        # 2. Re-rank top candidates with semantic search
        query_embedding = self.get_bert_embedding(query)
        
        # Re-rank only top candidates
        for hit in hits:
            doc_embedding = hit["_source"]["embedding"]
            semantic_score = self.cosine_similarity(query_embedding, doc_embedding)
            
            # Normalize BM25 score (varies widely)
            bm25_score = hit["_score"] / 10  # Simple normalization
            
            # Combine scores with weights
            hit["_score"] = semantic_weight * semantic_score + text_weight * bm25_score

        # Sort by combined score
        hits.sort(key=lambda x: x["_score"], reverse=True)

        # Apply pagination
        paginated_hits = hits[offset:offset + limit]

        # Return re-ranked results
        return {
            "hits": {
                "total": {"value": len(hits)},
                "hits": paginated_hits
            }
        }

    def _build_filters(self, filters: Optional[Dict]) -> List[Dict]:
        """Build Elasticsearch filters"""
        if not filters:
            return []
            
        filter_clauses = []
        if "author" in filters and filters["author"]:
            filter_clauses.append({"term": {"authors.keyword": filters["author"]}})
        if "category" in filters and filters["category"]:
            filter_clauses.append({"term": {"categories": filters["category"]}})
        if "year" in filters and filters["year"]:
            filter_clauses.append({"term": {"year": filters["year"]}})
            
        return filter_clauses

    # Multi-search method compatible with current API
    async def multi_search(
        self,
        query: str,
        semantic_weight: float = 0.7,
        text_weight: float = 0.3,
        author: Optional[str] = None,
        category: Optional[str] = None,
        year: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ):
        """Wrapper for search method to maintain API compatibility"""
        filters = {
            "author": author,
            "category": category,
            "year": year
        }
        
        return await self.search(
            query=query,
            filters=filters,
            limit=limit,
            offset=offset,
            semantic_weight=semantic_weight,
            text_weight=text_weight
        )

    async def autocomplete_title(self, prefix: str, limit: int = 5):
        """Get title suggestions for autocomplete"""
        response = await self.es.search(
            index=self.index_name,
            body={
                "suggest": {
                    "title_suggest": {
                        "prefix": prefix,
                        "completion": {
                            "field": "title.completion",
                            "size": limit
                        }
                    }
                }
            }
        )
        return response

    async def search_by_doi(self, doi: str):
        """Search for a paper by its DOI"""
        response = await self.es.search(
            index=self.index_name,
            body={
                "query": {
                    "term": {
                        "doi": doi
                    }
                }
            }
        )
        return response

    async def get_categories(self):
        """Get all available categories"""
        response = await self.es.search(
            index=self.index_name,
            body={
                "size": 0,
                "aggs": {
                    "categories": {
                        "terms": {
                            "field": "categories",
                            "size": 1000
                        }
                    }
                }
            }
        )
        return response

    async def search_by_author(self, author: str, limit: int = 10, offset: int = 0):
        """Search for papers by a specific author"""
        response = await self.es.search(
            index=self.index_name,
            body={
                "query": {
                    "match": {
                        "authors": author
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