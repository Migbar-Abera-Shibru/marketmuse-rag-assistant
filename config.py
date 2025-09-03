import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration - with better fallback handling
def get_api_key():
    """Get API key from environment or return empty string"""
    return os.getenv("GROQ_API_KEY", "").strip()

GROQ_API_KEY = get_api_key()
GROQ_MODEL = "llama-3.1-8b-instant"

# Vector Store Configuration
VECTOR_STORE_PATH = "chroma_db"
COLLECTION_NAME = "marketmuse_documents"

# Document Processing Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Supported File Types
SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.pptx', '.html', '.md']