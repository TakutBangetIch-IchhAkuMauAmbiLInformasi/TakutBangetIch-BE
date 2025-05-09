import asyncio
import json
import sys
import os
from pathlib import Path
import logging
from tqdm import tqdm
import torch
import re

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
        logger.info("Index created successfully with enhanced metadata mappings")
        
        # Load the JSON data
        logger.info("Loading arXiv data from JSON file...")
        json_file_path = r"C:\Users\Asus\Documents\Semester 6\TBI\TugasKelompok\TakutBangetIch-BE\app\data\arxiv1k.jsonl"
        
        # Process documents in batches
        batch_size = 50  # Reduced batch size for enhanced embeddings
        total_docs = 0
        processed_docs = 0
        
        # First count total documents
        with open(json_file_path, 'r', encoding='utf-8') as f:
            for _ in f:
                total_docs += 1
        
        logger.info(f"Found {total_docs} documents to process with enhanced embeddings")
        
        # Process documents in batches
        with open(json_file_path, 'r', encoding='utf-8') as f:
            batch = []
            
            for line in tqdm(f, total=total_docs):
                try:
                    # Parse JSON document
                    doc = json.loads(line.strip())
                    
                    # Clean up data
                    doc = clean_document(doc)
                    
                    batch.append(doc)
                    
                    if len(batch) >= batch_size:
                        # Bulk index the batch with enhanced embeddings
                        logger.info(f"Generating enhanced embeddings for batch of {len(batch)} documents...")
                        await es_service.bulk_index(batch)
                        
                        processed_docs += len(batch)
                        logger.info(f"Indexed {processed_docs}/{total_docs} documents with enhanced embeddings")
                        
                        # Clear batch
                        batch = []
                        
                        # Clear CUDA cache if using GPU
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            logger.info("Cleared GPU cache")
                            
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON: {str(e)}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing document: {str(e)}")
                    continue
            
            # Process remaining documents
            if batch:
                logger.info(f"Generating enhanced embeddings for final batch of {len(batch)} documents...")
                await es_service.bulk_index(batch)
                processed_docs += len(batch)
                logger.info(f"Indexed {processed_docs}/{total_docs} documents with enhanced embeddings")
        
        # Refresh index to make documents searchable
        await es_service.es.indices.refresh(index=es_service.index_name)
        logger.info("Index refreshed - all documents are now searchable")
        
        logger.info("Indexing with enhanced embeddings completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}")
        raise
    finally:
        await es_service.close()

def clean_document(doc):
    """Clean and normalize document fields"""
    # Ensure all fields exist
    for field in ['id', 'title', 'abstract', 'authors', 'categories', 'doi', 'submitter', 'year']:
        if field not in doc:
            doc[field] = ""
    
    # Convert year to string if it's numeric
    if isinstance(doc['year'], int):
        doc['year'] = str(doc['year'])
    
    # Normalize categories (remove extra spaces, lowercase)
    if doc['categories']:
        categories = doc['categories'].split(',')
        doc['categories'] = ','.join([c.strip() for c in categories])
    
    # Ensure authors is a string
    if isinstance(doc['authors'], list):
        doc['authors'] = ', '.join(doc['authors'])
    
    return doc

if __name__ == "__main__":
    asyncio.run(index_arxiv_data()) 