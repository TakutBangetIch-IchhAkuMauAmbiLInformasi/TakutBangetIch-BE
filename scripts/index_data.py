import asyncio
import pandas as pd
import sys
import os
from pathlib import Path
import json

sys.path.append(str(Path(__file__).parent.parent))

from app.services.elasticsearch_service import ElasticsearchService
import logging
from tqdm import tqdm
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def index_arxiv_data():
    try:
        # Initialize Elasticsearch service
        logger.info("Initializing Elasticsearch service and loading embedder model...")
        es_service = ElasticsearchService()
        await es_service.delete_index()
        
        # Create index if it doesn't exist
        await es_service.create_index()
        logger.info("Index created successfully with embedding")

        # Stream and index documents in batches to avoid high memory usage
        logger.info("Streaming arXiv data from JSONL for memory efficiency...")

        with open('scripts/arxiv1k.jsonl', 'r') as f:
            for line in tqdm(f, desc="Reading and indexing", unit="docs"):
                data = json.loads(line)
                # Create metadata dictionary
                metadata = {
                    "authors": data['authors'],
                    "categories": data['categories'],
                    "doi": data['doi'],
                    "submitter": data['submitter'],
                    "year": data['year']
                }
                
                combined_text = f"{data['title']} {data['abstract']}"
                
                await es_service.index_document(
                    doc_id=data['id'],
                    title=data['title'],
                    content=combined_text,
                    metadata=metadata
                )
                # Clear CUDA cache if using GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                logger.info(f"Indexed document ID: {data['id']}")
                   
        logger.info("Indexing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}")
        raise

    finally:
        asyncio.run(es_service.close())

if __name__ == "__main__":
    asyncio.run(index_arxiv_data())
    logger.info("Indexing successful.")