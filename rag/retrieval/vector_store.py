from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import uuid
import logging

from config.settings import VECTOR_DB_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, collection_name="recursive", url=None, in_memory=False):
        """
        Initialize the vector store client
       
        Args:
            collection_name: Name of the collection to use
            url: URL for Qdrant connection
            in_memory: Whether to use in-memory storage (for tests)
        """
        self.collection_name = collection_name
        
        logger.info(f"Initializing QdrantClient - collection: {collection_name}, in_memory: {in_memory}")
        
        if in_memory:
            # Use in-memory storage for tests
            logger.info("Using in-memory storage for Qdrant")
            self.client = QdrantClient(location=":memory:")
        else:
            # Use URL connection for production
            self.url = "http://qdrant:6333"
            logger.info(f"Using URL connection for Qdrant: {self.url}")
            self.client = QdrantClient(url=self.url)

        
        
    def ensure_collection_exists(self, vector_size=1024):
        """
        Ensure the collection exists, create it if it doesn't
        
        Args:
            vector_size: Size of the embedding vectors
        """
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
    
    def store(self, texts, embeddings):
        """
        Store texts and their embeddings in the vector store
        
        Args:
            texts: List of text strings
            embeddings: List of embedding vectors
            
        Returns:
            Operation info from Qdrant
        """
        self.ensure_collection_exists(len(embeddings[0]))
        
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={"text": text}
            )
            for text, embedding in zip(texts, embeddings)
        ]
        
        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=points
        )
        
        return operation_info
    
    def search(self, query_vector, limit=5, filter_params=None):
        """
        Search for similar vectors
        
        Args:
            query_vector: Embedding vector to search for
            limit: Maximum number of results
            filter_params: Filter for the search
            
        Returns:
            Search results from Qdrant
        """
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        ) 