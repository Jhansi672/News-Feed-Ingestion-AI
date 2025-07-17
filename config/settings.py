# config/settings.py
import os
from typing import List, Dict, Any
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # API Settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # OpenAI Settings
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")
    
    # Vector Store Settings
    chroma_db_path: str = Field(default="./chroma_db", env="CHROMA_DB_PATH")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    
    # Processing Settings
    batch_size: int = Field(default=10, env="BATCH_SIZE")
    processing_interval: int = Field(default=300, env="PROCESSING_INTERVAL")  # 5 minutes
    max_articles_per_feed: int = Field(default=50, env="MAX_ARTICLES_PER_FEED")
    
    # RSS Feed Sources
    rss_feeds: List[str] = Field(default=[
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://rss.cnn.com/rss/edition.rss",
        "https://feeds.reuters.com/reuters/topNews",
        "https://feeds.npr.org/1001/rss.xml",
        "https://feeds.washingtonpost.com/rss/world",
        "https://www.theguardian.com/world/rss",
        "https://feeds.skynews.com/feeds/rss/world.xml"
    ])
    
    # Content Processing Settings
    min_content_length: int = Field(default=100, env="MIN_CONTENT_LENGTH")
    max_content_length: int = Field(default=10000, env="MAX_CONTENT_LENGTH")
    
    # NLP Model Settings
    spacy_model: str = Field(default="en_core_web_sm", env="SPACY_MODEL")
    sentiment_model: str = Field(default="cardiffnlp/twitter-roberta-base-sentiment-latest", env="SENTIMENT_MODEL")
    
    # Classification Categories
    classification_categories: List[str] = Field(default=[
        "Politics",
        "Technology",
        "Business",
        "Sports",
        "Entertainment",
        "Science",
        "Health",
        "World News",
        "Local News",
        "Opinion"
    ])
    
    # Redis Settings (for caching and task queue)
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # Logging Settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="news_service.log", env="LOG_FILE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()