from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env", override=True)

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "TakutBangetIch Search Engine"
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