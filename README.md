# TakutBangetIch-BE

A semantic search engine for academic papers using FastAPI, Elasticsearch, and BERT embeddings. This API powers hybrid search capabilities combining text-based and semantic search with enhanced metadata handling.

## Features

- **Hybrid Search**: Combines traditional BM25 text search with BERT-based semantic search
- **Enhanced Embeddings**: Generates embeddings that understand both content and metadata
- **Metadata Filters**: Search by author, category, year, DOI, and more
- **Autocomplete**: Provides title suggestions as you type
- **Rich Highlighting**: Shows exactly why a document matched your query

## Prerequisites

- Python 3.8+
- Elasticsearch 7.10+ with API key authentication
- ArXiv dataset in JSONL format

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/TakutBangetIch-BE.git
   cd TakutBangetIch-BE
   ```

2. **Create and activate virtual environment**:
   ```bash
   # Windows
   python -m venv env
   env\Scripts\activate

   # Linux/macOS
   python -m venv env
   source env/bin/activategi
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Elasticsearch**:
   Create a `.env` file in the project root:
   ```
   ELASTICSEARCH_URL=your_elasticsearch_url
   ELASTICSEARCH_API_KEY=your_api_key
   ```

5. **Prepare dataset**:
   Place your ArXiv dataset in JSONL format at:
   ```
   app/data/arxiv1k.jsonl
   ```

## Running the Application

1. **Index the data**:
   ```bash
   python scripts/index_data.py
   ```
   This will:
   - Create the Elasticsearch index with proper mappings
   - Generate enhanced BERT embeddings for each document
   - Index documents in batches with progress tracking

2. **Start the API server**:
   ```bash
   python main.py
   ```
   The API will be available at http://localhost:8001

## API Usage

### 1. Basic Search
```http
POST /api/v1/search
Content-Type: application/json

{
  "query": "quantum computing"
}
```

### 2. Advanced Search with Filters
```http
POST /api/v1/search
Content-Type: application/json

{
  "query": "quantum computing",
  "semantic_weight": 0.8,
  "text_weight": 0.2,
  "author": "Yamamoto",
  "category": "quant-ph",
  "year": "2018-2021",
  "limit": 20,
  "offset": 0
}
```

### 3. Autocomplete
```http
GET /api/v1/autocomplete?prefix=quant&limit=5
```

### 4. Category Listing
```http
GET /api/v1/categories
```

### 5. Author Search
```http
GET /api/v1/author/John%20Smith
```

### 6. DOI Search
```http
GET /api/v1/doi/10.1088%2F1361-6463%2Fab5c71
```

## Technical Implementation

The search engine uses a two-stage approach:

1. **Initial Retrieval**: Fast BM25 text search to find potential candidates
2. **Re-ranking**: Semantic re-ranking using BERT embeddings with cosine similarity

Enhanced embeddings include semantic metadata sentences like:
- "This paper was written by {authors}."
- "This research belongs to the categories {categories}."
- "This was published in {year}."

This approach enables the model to understand not just the content but also the metadata relationships.

## License

MIT