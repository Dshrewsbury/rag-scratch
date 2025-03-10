import os
from pathlib import Path

# Base directory settings
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = os.getenv("MODELS_DIR", str(BASE_DIR / "assets/models"))
DATA_DIR = os.getenv("DATA_DIR", str(BASE_DIR / "assets"))
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR", str(BASE_DIR / "assets/embeddings"))
CHAT_HISTORY_DIR = os.getenv("CHAT_HISTORY_DIR", str(BASE_DIR / "assets/"))

# Model paths
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", str(Path(MODELS_DIR) / "mxbai-embed-large-v1-f16.gguf"))
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", str(Path(MODELS_DIR) / "llama-2-7b-chat.Q4_K_M.gguf"))

# API settings
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Vector DB settings
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", str(Path(EMBEDDINGS_DIR) / "recursive"))
CHAT_DB_PATH = os.getenv("CHAT_DB_PATH", str(Path(CHAT_HISTORY_DIR) / "chat_history.db"))

# LLM settings 
MAX_TOKENS = 512
BATCH_SIZE = 512 