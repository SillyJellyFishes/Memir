import os
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

# Load environment variables
load_dotenv()

# Directory where ChromaDB will persist data
PERSIST_DIR = os.getenv("CHROMADB_PERSIST_DIR", ".chromadb")

# Initialize ChromaDB client with DuckDB+Parquet for local persistence
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=PERSIST_DIR
))

# Helper to ensure persistence on shutdown or as needed

def init_db():
    """Initialize and persist ChromaDB storage."""
    client.persist()
