# backend/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from typing import Optional # Import Optional for optional types
from dotenv import load_dotenv # Import load_dotenv

# Load environment variables from .env file immediately
load_dotenv()

class Settings(BaseSettings):
    # Google Cloud Project ID
    GOOGLE_CLOUD_PROJECT_ID: str

    # Gemini API Key (if you're using the direct genai client, not Vertex AI with service accounts)
    # Use Optional[str] or str | None to correctly indicate it can be a string or None
    # It should NOT have a hardcoded default value here for security reasons.
    GEMINI_API_KEY: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Load settings on startup
settings = Settings()

# DO NOT set GOOGLE_APPLICATION_CREDENTIALS here.
# It should be set as an environment variable in your shell before running the app.
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

#testing branch
x=1