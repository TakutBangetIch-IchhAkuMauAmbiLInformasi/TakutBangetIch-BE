from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "TakutBangetIch Search Engine"
    
    # Elasticsearch Settings
    ELASTICSEARCH_URL: str = "http://localhost:9200"  # Default fallback URL
    ELASTICSEARCH_API_KEY: Optional[str] = None
    
    # Model Settings
    MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 