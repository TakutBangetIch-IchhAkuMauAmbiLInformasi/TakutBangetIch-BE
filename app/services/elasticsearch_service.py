from elasticsearch import AsyncElasticsearch
from transformers import AutoTokenizer, AutoModel
import torch
from app.core.config import settings
import numpy as np
from typing import List, Dict, Optional

class ElasticsearchService:
    def __init__(self):
        # Configure Elasticsearch client with URL and API key
        if settings.ELASTICSEARCH_API_KEY is not None:
            self.es = AsyncElasticsearch(
                settings.ELASTICSEARCH_URL,
                api_key=settings.ELASTICSEARCH_API_KEY
            )
        else:
            self.es = AsyncElasticsearch(
                settings.ELASTICSEARCH_URL
            )

        # Initialize BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(settings.EMBEDDINGS_MODEL)
        self.model = AutoModel.from_pretrained(settings.EMBEDDINGS_MODEL).to(settings.DEVICE).eval()
        self.index_name = settings.ELASTICSEARCH_INDEX_NAME
        print(f"DEBUG: ELASTICSEARCH_INDEX_NAME = {self.index_name}")

    def get_bert_embedding(self, text: str) -> np.ndarray:
        """Generate BERT embeddings for the given text"""
        if not text or not text.strip():
            # Return zero vector for empty text
            print("No text detected")
            return torch.zeros(settings.EMBEDDINGS_DIM)
        
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=settings.MAX_LENGTH)
        
        # Check if tokenization resulted in valid tokens
        if tokens["input_ids"].shape[1] == 0:
            print("Tokenization resulted in empty input_ids.")
            return torch.zeros(settings.EMBEDDINGS_DIM)
        
        with torch.no_grad():
            outputs = self.model(**tokens).last_hidden_state
            mask = tokens['attention_mask'].unsqueeze(-1)
            embeddings = (outputs * mask).sum(dim=1) / mask.sum(dim=1)
            embeddings = embeddings.squeeze().numpy()

            return embeddings
    
    def get_enhanced_embedding(self, doc: Dict) -> np.ndarray:
        """
        Generate enhanced BERT embeddings that include semantic metadata sentences
        """
        # Create semantic sentences from metadata
        metadata_sentences = []
        
        # Author sentence
        if doc.get('authors'):
            metadata_sentences.append(f"this paper was written by {doc['authors']}.")
        
        # Categories sentence
        if doc.get('categories'):
            metadata_sentences.append(f"belongs to the categories {doc['categories']}.")
        
        # Year sentence
        if doc.get('year'):
            metadata_sentences.append(f"published in {doc['year']}.")
        
        # Submitter sentence
        if doc.get('submitter'):
            metadata_sentences.append(f"submitted by {doc['submitter']}.")

        if doc.get('passage'):
            metadata_sentences.append(f"passage: {doc['passage']}.")
        
        # Combine title, abstract, and metadata sentences
        combined_text = f"{doc['title']} {doc['abstract']} " + " ".join(metadata_sentences)
        
        # # Truncate if needed to fit BERT's max token length
        # if len(combined_text) > 5000:  # Arbitrary limit to avoid tokenizer issues
        #     combined_text = combined_text[:5000]
        splitted = combined_text.split()
        if len(splitted) > settings.MAX_LENGTH:
            bound = int(0.8 * len(splitted)) - 1
            end = -(int(0.2 * len(splitted)))
            combined_text = " ".join(splitted[:bound])
            combined_text += "<truncated>" + " ".join(splitted[end:])
        
        # Generate BERT embedding
        return self.get_bert_embedding(combined_text)

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
                                    "filter": ["lowercase", "trim", "asciifolding"]
                                },
                                "category_analyzer": {
                                    "type": "custom",
                                    "tokenizer": "keyword",
                                    "filter": ["lowercase"]
                                }
                            },
                            "normalizer": {
                                "lowercase_normalizer": {
                                    "type": "custom",
                                    "filter": ["lowercase", "asciifolding"]
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
                                    "keyword": {"type": "keyword", "normalizer": "lowercase_normalizer"},
                                    "completion": {"type": "completion"},
                                    "exact": {"type": "text", "analyzer": "standard"}
                                }
                            },
                            "abstract": {
                                "type": "text",
                                "analyzer": "english",
                                "fields": {
                                    "standard": {"type": "text", "analyzer": "standard"}
                                }
                            },
                            "passage": {
                                "type": "text",
                                "analyzer": "english",
                                "fields": {
                                    "standard": {"type": "text", "analyzer": "standard"}
                                }
                            }, 
                            "authors": {
                                "type": "text",
                                "analyzer": "author_analyzer",
                                "fields": {
                                    "keyword": {"type": "keyword", "normalizer": "lowercase_normalizer"}
                                }
                            },
                            "categories": {
                                "type": "keyword",
                                "fields": {
                                    "analyzed": {"type": "text", "analyzer": "category_analyzer"}
                                }
                            },
                            "doi": {
                                "type": "keyword"
                            },
                            "submitter": {
                                "type": "keyword",
                                "fields": {
                                    "text": {"type": "text", "analyzer": "standard"}
                                }
                            },
                            "year": {
                                "type": "keyword",
                                "fields": {
                                    "numeric": {"type": "integer"}
                                }
                            },
                            "embedding": {
                                "type": "dense_vector",
                                "dims": settings.EMBEDDINGS_DIM,
                                "index": True,
                                "similarity": "cosine"
                            },
                            "all_content": {
                                "type": "text",
                                "analyzer": "english"
                            }
                        }
                    }
                }
            )

    async def index_document(self, doc: Dict):
        """Index an arXiv paper document with semantically enhanced embeddings"""
        # Generate enhanced embedding with metadata sentences
        if settings.DOC_LENGTH_LIMIT > 0: 
            doc['passage'] = doc['passage'][:min(settings.DOC_LENGTH_LIMIT, len(doc['passage']))]
        embedding = self.get_enhanced_embedding(doc)
        
        # Create a combined field for general search
        all_content = f"title {doc['title']} authored by {doc['authors']} has category (categories) {doc['categories']} released in or has year {doc['year']} has abstract {doc['abstract']} with content {doc['passage']}"
        
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
            "passage": doc["passage"],
            "embedding": embedding.tolist(),
            "all_content": all_content
        }
        
        await self.es.index(
            index=self.index_name,
            id=doc["id"],
            body=document
        )

    async def bulk_index(self, documents: List[Dict], precompute_embeddings: bool = False):
        """Bulk index documents with semantically enhanced embeddings"""
        operations = []
        for doc in documents:
            # Generate enhanced embedding with metadata sentences
            if not precompute_embeddings:
                embedding = self.get_enhanced_embedding(doc)
            else:
                embedding = doc.get("embedding")
            
            # Create a combined field for general search
            all_content = f"{doc['title']} {doc['authors']} {doc['categories']} {doc['year']} {doc['abstract']} {doc['passage']}"
            
            # Prepare document for indexing
            indexed_doc = {
                "id": doc["id"],
                "title": doc["title"],
                "abstract": doc["abstract"],
                "authors": doc["authors"],
                "categories": doc["categories"],
                "doi": doc["doi"],
                "submitter": doc["submitter"],
                "year": doc["year"],
                "passage": doc["passage"],
                "embedding": embedding.tolist() if not isinstance(embedding, list) else embedding,
                "all_content": all_content
            }
            
            # Add operation type and metadata
            operations.append({"index": {"_index": self.index_name, "_id": doc["id"]}})
            # Add the document body as a separate action
            operations.append(indexed_doc)
        
        await self.es.bulk(body=operations, refresh="wait_for")

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
        Two-stage search with enhanced metadata handling:
        1. BM25 for initial retrieval with expanded field set
        2. Semantic re-ranking of top candidates
        """
        # 1. Initial BM25 retrieval
        initial_query = {
            "query": {
                "bool": {
                    "should": [
                        # Search in title with high boost
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^5", "title.exact^3"],
                                "type": "best_fields",
                                "operator": "or",
                                "boost": 2.0
                            }
                        },
                        # Search in abstract
                        {
                            "match": {
                                "abstract": {
                                    "query": query,
                                    "boost": 1.0
                                }
                            }
                        },
                        # Search in all content (includes metadata)
                        {
                            "match": {
                                "all_content": {
                                    "query": query,
                                    "boost": 0.5
                                }
                            }
                        },
                        # Author name matches
                        {
                            "match": {
                                "authors": {
                                    "query": query,
                                    "boost": 1.5
                                }
                            }
                        },
                        # Category matches
                        {
                            "match": {
                                "categories.analyzed": {
                                    "query": query,
                                    "boost": 1.0
                                }
                            }
                        }
                    ],
                    "filter": self._build_filters(filters),
                    "minimum_should_match": 0
                }
            },
            "highlight": {
                "fields": {
                    "title": {},
                    "abstract": {},
                    "authors": {},
                    "categories.analyzed": {}
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
            print("No hits found in initial search.")
            return initial_response

        # 2. Re-rank top candidates with semantic search
        query_embedding = self.get_bert_embedding(query)
        
        # Re-rank only top candidates
        for hit in hits:
            doc_embedding = hit["_source"]["embedding"]
            semantic_score = self.cosine_similarity(query_embedding, doc_embedding)
            
            # Normalize BM25 score (varies widely)
            bm25_score = hit["_score"] / initial_response["hits"]["max_score"]
            
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
        """Build Elasticsearch filters with improved metadata handling"""
        if not filters:
            return []
            
        filter_clauses = []
        
        # Author filter with better matching
        if "author" in filters and filters["author"]:
            # Use match query for partial author name matching
            filter_clauses.append({
                "match_phrase": {
                    "authors": {
                        "query": filters["author"],
                        "slop": 2  # Allow slight variations in word order
                    }
                }
            })
        
        # Category filter with exact matching
        if "category" in filters and filters["category"]:
            # Categories can be comma-separated, so we need to handle multiple values
            if "," in filters["category"]:
                categories = [c.strip() for c in filters["category"].split(",")]
                filter_clauses.append({
                    "terms": {
                        "categories": categories
                    }
                })
            else:
                filter_clauses.append({
                    "term": {
                        "categories": filters["category"]
                    }
                })
        
        # Year filter with range support
        if "year" in filters and filters["year"]:
            # Support year ranges like "2018-2021"
            if "-" in filters["year"]:
                start_year, end_year = filters["year"].split("-")
                filter_clauses.append({
                    "range": {
                        "year.numeric": {
                            "gte": int(start_year),
                            "lte": int(end_year)
                        }
                    }
                })
            else:
                filter_clauses.append({
                    "term": {
                        "year": filters["year"]
                    }
                })
        
        # DOI filter
        if "doi" in filters and filters["doi"]:
            filter_clauses.append({
                "term": {
                    "doi": filters["doi"]
                }
            })
        
        # Submitter filter
        if "submitter" in filters and filters["submitter"]:
            filter_clauses.append({
                "match": {
                    "submitter": filters["submitter"]
                }
            })
            
        return filter_clauses

    async def metadata_search(
        self,
        metadata_filters: Dict,
        limit: int = 10,
        offset: int = 0
    ):
        """
        Search documents using only metadata filters without a text query
        Useful for browsing papers by category, author, year, etc.
        """
        # Build filters from metadata
        filter_clauses = self._build_filters(metadata_filters)
        
        if not filter_clauses:
            raise ValueError("At least one metadata filter must be provided")
        
        # Build the metadata search query
        metadata_query = {
            "query": {
                "bool": {
                    "filter": filter_clauses
                }
            },
            "size": limit,
            "from": offset,
            "sort": [
                {"year.numeric": {"order": "desc"}},  # Sort by year descending
                {"_score": {"order": "desc"}}         # Then by relevance
            ]
        }
        
        # Execute the search
        response = await self.es.search(
            index=self.index_name,
            body=metadata_query
        )
        
        return response

    # Multi-search method compatible with current API
    async def multi_search(
        self,
        query: str,
        semantic_weight: float = 0.7,
        text_weight: float = 0.3,
        author: Optional[str] = None,
        category: Optional[str] = None,
        year: Optional[str] = None,
        doi: Optional[str] = None,
        submitter: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ):
        """Enhanced wrapper for search method with better metadata support"""
        # If query is empty but metadata filters are provided, use metadata search
        if not query or query.strip() == "":
            filters = {
                "author": author,
                "category": category,
                "year": year,
                "doi": doi,
                "submitter": submitter
            }
            # Remove None values
            filters = {k: v for k, v in filters.items() if v is not None}
            
            if filters:
                return await self.metadata_search(
                    metadata_filters=filters,
                    limit=limit,
                    offset=offset
                )
            else:
                # If no query and no filters, return latest papers
                return await self.es.search(
                    index=self.index_name,
                    body={
                        "query": {"match_all": {}},
                        "sort": [{"year.numeric": {"order": "desc"}}],
                        "size": limit,
                        "from": offset
                    }
                )
        
        # If we have a query, use hybrid search
        filters = {
            "author": author,
            "category": category,
            "year": year,
            "doi": doi,
            "submitter": submitter
        }
        # Remove None values
        filters = {k: v for k, v in filters.items() if v is not None}
        
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
                    "match_phrase": {
                        "authors": {
                            "query": author,
                            "slop": 2
                        }
                    }
                },
                "size": limit,
                "from": offset
            }
        )
        return response
    
    # Add this to your elasticsearch_service.py for debugging
    async def debug_index_stats(self):
        """Debug method to check index statistics"""
        stats = await self.es.indices.stats(index=self.index_name)
        count = await self.es.count(index=self.index_name)
        return {
            "index_stats": stats,
            "document_count": count
        }

    async def close(self):
        """Close the Elasticsearch connection"""
        await self.es.close() 


if __name__ == "__main__":
    es_service = ElasticsearchService()