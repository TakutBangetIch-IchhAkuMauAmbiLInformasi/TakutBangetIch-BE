import asyncio
import json
import sys
import os
from pathlib import Path
import logging
import re
import torch
from tqdm import tqdm
import numpy as np

from dotenv import load_dotenv

load_dotenv(".env", override=True)

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.elasticsearch_service import ElasticsearchService

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
        json_file_path = r"scripts\arxiv-csOnly-circa-2025.jsonl"
        passage_dir = r"scripts\output_pymu"
        
        data = np.load(r"scripts\arxiv_embeddings.npz")

        # Access the arrays
        ids = data["ids"]           # shape: (N,) — string IDs
        embeddings = data["embeddings"]  # shape: (N, 768) — float32 vectors
        
        # Process documents in batches
        batch_size = 22
        total_docs = 0
        processed_docs = 0
        
        # First count total documents
        with open(json_file_path, 'r', encoding='utf-8') as f:
            for _ in f:
                total_docs += 1

        if total_docs % batch_size != 0:
            logger.warning(f"Total documents {total_docs} is not a multiple of batch size {batch_size}.")
        
        logger.info(f"Found {total_docs} documents to process with enhanced embeddings")
        
        # Process documents in batches
        with open(json_file_path, 'r', encoding='utf-8') as f:
            batch = []
            
            for line in tqdm(f, total=total_docs, desc="Processing documents", unit="doc"):
                try:
                    # Parse JSON document
                    doc = json.loads(line.strip())
                    doc_id = str(doc.get('id', None))
                    if not doc_id:
                        logger.warning(f"Document ID: {doc_id} is missing, skipping document")
                        continue
                    
                    passage_file = os.path.join(passage_dir, f"{doc_id}.txt")
                    if os.path.exists(passage_file):
                        with open(passage_file, 'r', encoding='utf-8') as passage_f:
                            doc['passage'] = passage_f.read()

                    if not doc_id in ids:
                        logger.warning(f"Document ID: {doc_id} not found in embeddings, skipping document")
                        continue
                    else:
                        # Get the corresponding embedding
                        embedding = embeddings[ids == doc_id]
                        if len(embedding) == 0:
                            logger.warning(f"Embedding for Document ID: {doc_id} not found, skipping document")
                            continue
                        doc['embedding'] = embedding.tolist()[0]
                    
                    # Clean up data
                    doc = clean_document(doc)
                    
                    batch.append(doc)
                    
                    if len(batch) >= batch_size:
                        # Bulk index the batch with enhanced embeddings
                        logger.info(f"Generating enhanced embeddings for batch of {len(batch)} documents...")
                        await es_service.bulk_index(batch, precompute_embeddings=True)
                        
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
            
            # Process remaining documents
            if batch:
                logger.info(f"Generating enhanced embeddings for final batch of {len(batch)} documents...")
                await es_service.bulk_index(batch, precompute_embeddings=True)
                processed_docs += len(batch)
                logger.info(f"Indexed {processed_docs}/{total_docs} documents with enhanced embeddings")
        
        # Refresh index to make documents searchable
        await es_service.es.indices.refresh(index=es_service.index_name)
        logger.info("Index refreshed - all documents are now searchable")
        
        logger.info("Indexing with enhanced embeddings completed successfully!")

        count = await es_service.es.count(index=es_service.index_name)
        print("Final document count:", count["count"])

        await es_service.close()
        
    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}")
        raise

def clean_text(text, rem_delimiter=False):
    """Clean and normalize text"""
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # remove "--- Page xxx ---" delimiter
    if rem_delimiter:
        text = re.sub(r'--- Page \d+ ---', '', text)
    
    # Normalize whitespace
    text = text.strip().lower()
    
    return text

def clean_document(doc):
    """Clean and normalize document fields"""
    # Ensure all fields exist
    for field in ['id', 'title', 'abstract', 'authors', 'categories', 'doi', 'submitter', 'year', 'passage']:
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

    if doc['title']:
        doc['title'] = clean_text(doc['title'])

    if doc['abstract']:
        doc['abstract'] = clean_text(doc['abstract'])
    
    if doc['passage']:
        doc['passage'] = clean_text(doc['passage'], rem_delimiter=True)
    
    return doc

if __name__ == "__main__":
    asyncio.run(index_arxiv_data()) 