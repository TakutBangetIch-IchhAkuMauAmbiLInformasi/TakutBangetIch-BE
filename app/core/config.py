from pydantic_settings import BaseSettings
from typing import Optional, List
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env", override=True)

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "TakutBangetIch Search Engine"
      # CORS Settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",  # React development server
        "http://localhost:3001",  # Alternative frontend port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "https://your-frontend-domain.com",  # Replace with your actual frontend domain
        "https://your-app.vercel.app",  # If using Vercel
        "https://your-app.netlify.app",  # If using Netlify
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override ALLOWED_ORIGINS from environment variable if set
        env_origins = os.getenv("ALLOWED_ORIGINS")
        if env_origins:
            try:
                # Try to parse as JSON array
                self.ALLOWED_ORIGINS = json.loads(env_origins)
            except json.JSONDecodeError:
                # Fallback: split by comma and strip whitespace
                self.ALLOWED_ORIGINS = [origin.strip() for origin in env_origins.split(",")]
    
    # Elasticsearch Settings
    ELASTICSEARCH_URL: str = "http://localhost:9200"  # Default fallback URL
    ELASTICSEARCH_API_KEY: Optional[str] = None
    HUGGINGFACEHUB_API_TOKEN: str
    DEEPSEEK_KEY: str 
    ELASTICSEARCH_INDEX_NAME: str = "test"
    DEEPSEEK_API_KEY: str 

    EMBEDDINGS_MODEL: str = "allenai/longformer-base-4096"
    MAX_LENGTH: int = 4096
    EMBEDDINGS_DIM: int = 768
    DEVICE: str = "cpu"

    DOC_LENGTH_LIMIT: int = 0
    
    class Config:
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / ".env"
        case_sensitive = True

settings = Settings() 
print(f"DEBUG: ELASTICSEARCH_INDEX_NAME = {settings.ELASTICSEARCH_INDEX_NAME}")
print(f"DEBUG: Environment variable = {os.getenv('ELASTICSEARCH_INDEX_NAME', 'Not set')}")
print(f"DEBUG: Current working directory = {os.getcwd()}")
print(f"DEBUG: .env file exists = {os.path.exists('.env')}")