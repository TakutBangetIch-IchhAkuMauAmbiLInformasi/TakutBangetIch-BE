import asyncio
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModel
import aiohttp
import numpy as np
import requests
import json
from app.core.config import settings
from app.models.search import SearchResult
from elasticsearch import AsyncElasticsearch
import torch

class ChatBotService:
    def __init__(self):
        self.api_key = settings.DEEPSEEK_API_KEY
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.model_name = "deepseek-chat" 
        if settings.ELASTICSEARCH_API_KEY is not None:
            self.es = AsyncElasticsearch(
                settings.ELASTICSEARCH_URL,
                api_key=settings.ELASTICSEARCH_API_KEY
            )
        else:
            self.es = AsyncElasticsearch(
                settings.ELASTICSEARCH_URL
            )

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
        
    async def chat(
        self, 
        message: str, 
        conversation_history: List[Dict[str, str]] = None,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        ) -> str:
        """
        Send a chat message to DeepSeek API
        
        Args:
            message: User's message
            conversation_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
            system_prompt: Optional system prompt to set behavior
            temperature: Randomness of response (0.0-1.0)
            max_tokens: Maximum tokens in response
            
        Returns:
            Assistant's response
        """
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        if conversation_history:
            messages.extend(conversation_history)
            
        messages.append({"role": "user", "content": message})
        print(messages)
        # Prepare request payload
        payload_identifier = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1,
            "stream": False
        }
        try:
            # Make API request
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload_identifier,
                timeout=100
            )
            
            response.raise_for_status() # YES/NO for query DB
            
            # Parse response

            result = response.json() 
            
            if "choices" in result and len(result["choices"]) > 0:
                if result["choices"][0]["message"]["content"].strip() == "YES":
                    messages.append({"role": "system", "content": f"Extract the academic concept from {message}. [Explain me about Knowledge Graph >> Knowledge Graph] [Do you know what is Djikstra >> Djikstra]"})

                    payload_ext_concept = {
                        "model": self.model_name,
                        "messages": messages[1:],
                        "temperature": temperature,
                        "max_tokens": 20,
                        "stream": False
                    }
                    response = requests.post(
                        self.base_url,
                        headers=self.headers,
                        json=payload_ext_concept,
                        timeout=100
                        )
                    
                    response.raise_for_status() # YES/NO for query DB
                    response = await self.search(
                                    query= result["choices"][0]["message"]["content"].strip(),
                                )
                    
                    hits =  response["hits"]["hits"]

                    
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
                    summary = await self.summarize(message, results)
                    messages.append({"role": "system", "content": f"Summary information: {summary}"})
                    messages.append({"role": "system", "content": f"Answer {message} based on the summary Information"}) 
                    messages.append({"role": "system", "content": f"Always add Citation from the given summary information [Citation example from summary information: **Citations** [0] 2503.04110 [1] 2501.06293 [2] 2501.17037]!"}) 

                payload = {
                        "model": self.model_name,
                        "messages": messages[1:],
                        "temperature": temperature,
                        "max_tokens": 512,
                        "stream": False
                    }
                response = requests.post(
                        self.base_url,
                        headers=self.headers,
                        json=payload,
                        timeout=100
                        )
                response.raise_for_status()  # GENERATE CHAT 
                result = response.json()

                return result["choices"][0]["message"]["content"].strip()
            else:
                return "Sorry, I couldn't generate a response."
                
        except requests.exceptions.RequestException as e:
            print(f"DeepSeek API request error: {e}")
            return f"Error: Unable to connect to DeepSeek API - {str(e)}"
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return "Error: Invalid response from DeepSeek API"
        except Exception as e:
            print(f"Unexpected error: {e}")
            return f"Error: {str(e)}"
    
    
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

    async def close(self):
        """Close  connection"""
        await self.es.close() 

    async def summarize(
        self, query: str, context: List[SearchResult], top_k: int = 3
    ) -> str:
        
        input_texts = []
        paper_ids = []
        titles = []
        for i, paper in enumerate(context[:top_k]):
            # Get the paper ID directly from the paper object, not from metadata
            paper_id = paper.id if hasattr(paper, 'id') and paper.id else paper.metadata.get('id', f'unknown-{i}')
            paper_info = (
                f"Paper {i}  >>"
                f"Title: {paper.title}\n"
                f"Abstract: {paper.content}\n"
                f"ID: {paper_id}\n"
            )
            input_texts.append(paper_info)
            paper_ids.append(paper_id)
            titles.append(paper.title)

        papers_text = "\n\n".join(input_texts)
        titles_text = ", ".join([f"'{title}'" for title in titles])

        # Generate a clean, well-referenced bullet-point summary
        system_prompt = "You are an expert academic researcher that summarizes scientific papers accurately and concisely."
        user_prompt = (
            f"{papers_text}\n\n"
            f"Based on your query '{query}', these are the most relevant papers:\n\n"
            f"1. Begin with a brief introduction about '{query}'.\n"
            f"2. For EACH paper, create bullet points that explain:\n"
            f"   - The main methodology or approach used\n"
            f"   - Key findings and results\n"  
            f"   - Specific contributions to the field\n"
            f"3. Every bullet point MUST reference the source paper by number [0], [1], etc.\n"
            f"4. Paraphrase rather than directly quote - explain concisely what each paper says about '{query}'.\n"
            f"5. Focus only on information relevant to '{query}'.\n\n"
            f"Summary of '{query}' based on papers: {titles_text}:"
        )

        print(f"Sending request to DeepSeek API with query: {query}")
        
        # Create the API request payload
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        try:
            # Create a timeout for the client session
            timeout = aiohttp.ClientTimeout(total=90)  # 90 seconds timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Add retry logic for API calls
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        async with session.post(self.base_url, headers=self.headers, json=payload) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                raise Exception(f"DeepSeek API returned status code {response.status}: {error_text}")
                            
                            response_data = await response.json()
                            summary_part = response_data["choices"][0]["message"]["content"].strip()
                            break  # Success, exit the retry loop
                    
                    except (aiohttp.ClientConnectorError, aiohttp.ClientResponseError, aiohttp.ServerTimeoutError) as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            raise Exception(f"Failed to connect to DeepSeek API after {max_retries} attempts: {str(e)}")
                        print(f"Connection error, retrying ({retry_count}/{max_retries}): {str(e)}")
                        await asyncio.sleep(2 * retry_count)  # Exponential backoff
            
            # Format the final output with citations
            formatted_output = f"**Summary on '{query}'**\n{summary_part}\n\n**Citations**\n"
            for i, paper_id in enumerate(paper_ids):
                formatted_output += f"[{i}] {paper_id}\n"
            
            return formatted_output.strip()
            
        except Exception as e:
            print(f"Error calling DeepSeek API: {str(e)}")
            return f"Error generating summary: {str(e)}"


