# backend/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file.
# This makes them available to the entire application before any other code runs.
load_dotenv()

class Settings(BaseSettings):
    """
    Defines the application's configuration settings, loaded from the .env file.
    """
    # Google Cloud settings
    GOOGLE_CLOUD_PROJECT_ID: str
    GOOGLE_APPLICATION_CREDENTIALS: str

    # Azure settings
    AZURE_SPEECH_KEY: str
    AZURE_SPEECH_REGION: str

    # LangSmith settings (optional)
    LANGCHAIN_TRACING_V2: Optional[str] = None
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: Optional[str] = None

    PERFORMANCE_TEST_API_KEY: str

    # RAG Model and Vector Store Configurations
    EMBEDDING_MODEL: str = "vertexai"  # Options: "vertexai", "openai", "sentence-transformers"
    EMBEDDING_MODEL_NAME: str = "text-embedding-005"  # Specific model name (e.g., "text-embedding-005", "all-MiniLM-L6-v2")

    # Generative Model Configuration
    GENERATION_MODEL: str = "gemini"  # Options: "gemini", "openai", "mistral"
    GENERATION_MODEL_NAME: str = "gemini-2.5-flash"  # Options: "gemini-2.5-flash", "gpt-4", "mistralai/Mistral-7B-v0.1"

    VECTOR_STORE: str = "chroma"  # Options: "chroma", "pinecone", "faiss"
    CHROMA_DB_PATH: str = "data/chroma_db"  # Path to Chroma DB (if using Chroma)
    PINECONE_INDEX_NAME: Optional[str] = None  # Pinecone index name (if using Pinecone)





    # This tells Pydantic to read from the .env file.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Create a single, globally accessible settings object
settings = Settings()