import asyncio
import json
import sys
import os
from pathlib import Path
import logging
from tqdm import tqdm
import torch

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.elasticsearch_service import ElasticsearchService
from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def index_arxiv_data():
    try:
        # Initialize Elasticsearch service
        logger.info("Initializing Elasticsearch service and loading BERT model...")
        es_service = ElasticsearchService()
        
        # Drop existing index if it exists
        if await es_service.es.indices.exists(index=es_service.index_name):
            logger.info(f"Dropping existing index: {es_service.index_name}")
            await es_service.es.indices.delete(index=es_service.index_name)
        
        # Create index with proper mappings
        await es_service.create_index()
        logger.info("Index created successfully with proper mappings")
        
        # Load the JSON data
        logger.info("Loading arXiv data from JSON file...")
        json_file_path = r"C:\Users\Asus\Documents\Semester 6\TBI\TugasKelompok\TakutBangetIch-BE\app\data\arxiv1k.jsonl"
        
        # Process documents in batches
        batch_size = 100
        total_docs = 0
        processed_docs = 0
        
        # First count total documents
        with open(json_file_path, 'r', encoding='utf-8') as f:
            for _ in f:
                total_docs += 1
        
        logger.info(f"Found {total_docs} documents to process")
        
        # Process documents in batches
        with open(json_file_path, 'r', encoding='utf-8') as f:
            batch = []
            
            for line in tqdm(f, total=total_docs):
                try:
                    # Parse JSON document
                    doc = json.loads(line.strip())
                    
                    # Generate BERT embedding for combined title and abstract
                    combined_text = f"{doc['title']} {doc['abstract']}"
                    embedding = es_service.get_bert_embedding(combined_text)
                    
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
                        "embedding": embedding.tolist()
                    }
                    
                    batch.append(indexed_doc)
                    
                    if len(batch) >= batch_size:
                        # Bulk index the batch
                        await es_service.bulk_index(batch)
                        
                        processed_docs += len(batch)
                        logger.info(f"Indexed {processed_docs}/{total_docs} documents")
                        
                        # Clear batch
                        batch = []
                        
                        # Clear CUDA cache if using GPU
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON: {str(e)}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing document: {str(e)}")
                    continue
            
            # Process remaining documents
            if batch:
                await es_service.bulk_index(batch)
                processed_docs += len(batch)
                logger.info(f"Indexed {processed_docs}/{total_docs} documents")
        
        # Refresh index to make documents searchable
        await es_service.es.indices.refresh(index=es_service.index_name)
        logger.info("Index refreshed - all documents are now searchable")
        
        logger.info("Indexing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}")
        raise
    finally:
        await es_service.close()

if __name__ == "__main__":
    asyncio.run(index_arxiv_data()) 