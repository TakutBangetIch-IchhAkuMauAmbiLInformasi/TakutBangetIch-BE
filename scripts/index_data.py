import asyncio
import pandas as pd
import sys
import os
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.elasticsearch_service import ElasticsearchService
from app.core.config import settings
import logging
from tqdm import tqdm
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def index_arxiv_data():
    try:
        # Initialize Elasticsearch service
        logger.info("Initializing Elasticsearch service and loading BERT model...")
        es_service = ElasticsearchService()
        
        # Create index if it doesn't exist
        await es_service.create_index()
        logger.info("Index created successfully with BERT embedding dimensions (768)")
        
        # Load the data and convert to DataFrame
        logger.info("Loading arXiv data...")
        data_list = pd.read_pickle('app/data/arxiv_10k.pkl')
        df = pd.DataFrame(data_list)
        
        # Index documents in batches
        batch_size = 100
        total_docs = len(df)
        
        logger.info(f"Starting to index {total_docs} documents with BERT embeddings...")
        
        for i in tqdm(range(0, total_docs, batch_size)):
            batch = df.iloc[i:i + batch_size]
            
            for _, row in batch.iterrows():
                # Create metadata dictionary
                metadata = {
                    "authors": row['authors'],
                    "categories": row['categories'],
                    "doi": row['doi'],
                    "submitter": row['submitter'],
                    "year": row['year']
                }
                
                # Combine title and abstract for better semantic representation
                combined_text = f"{row['title']} {row['abstract']}"
                
                # Index the document with BERT embedding
                await es_service.index_document(
                    doc_id=row['id'],
                    title=row['title'],
                    content=combined_text,  # Using combined text for better semantic search
                    metadata=metadata
                )
            
            logger.info(f"Indexed {min(i + batch_size, total_docs)}/{total_docs} documents")
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info("Indexing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}")
        raise
    finally:
        await es_service.close()

if __name__ == "__main__":
    asyncio.run(index_arxiv_data()) 