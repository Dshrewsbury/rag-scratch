from llama_cpp import Llama
from config.settings import EMBEDDING_MODEL_PATH, BATCH_SIZE, MAX_TOKENS

class EmbeddingGenerator:
    def __init__(self, model_path=None, batch_size=BATCH_SIZE, max_tokens=MAX_TOKENS):
        """
        Initialize an embedding generator using Llama
        
        Args:
            model_path: Path to the embedding model
            batch_size: Number of texts to process at once
            max_tokens: Maximum tokens per text
        """
        self.model_path = model_path or EMBEDDING_MODEL_PATH
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.model = self._load_model()
        
    def _load_model(self):
        """Load the embedding model"""
        return Llama(
            model_path=self.model_path,
            embedding=True,
            verbose=False,
            n_batch=self.batch_size,
            max_tokens=self.max_tokens
        )
    
    def generate_embeddings(self, texts):
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        if isinstance(texts, str):
            texts = [texts]
            
        try:
            result = self.model.create_embedding(texts)
            return [item['embedding'] for item in result['data']]
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return [] 