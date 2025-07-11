import vertexai
from config import settings

try:
    print(f"Attempting to initialize with Project ID: {settings.GOOGLE_CLOUD_PROJECT_ID}")
    vertexai.init(project=settings.GOOGLE_CLOUD_PROJECT_ID, location="us-central1")
    print("✅ Success! Your environment is authenticated correctly.")
except Exception as e:
    print(f"❌ Failure! There is an error with your environment setup: {e}")