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

    # This tells Pydantic to read from the .env file.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Create a single, globally accessible settings object
settings = Settings()