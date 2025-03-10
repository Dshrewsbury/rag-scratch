import pytest
from typing import Dict, List, Any

from rag.agent.agent import Agent
from rag.retrieval.vector_store import VectorStore
from rag.processing.embedding.embedding_generator import EmbeddingGenerator


# Fake clients for unit testing
class FakeLlamaClient:
    """Mock client for Llama embeddings"""
    def __init__(self, *args, **kwargs):
        pass
        
    def create_embedding(self, texts):
        """Return fake embeddings"""
        if isinstance(texts, str):
            texts = [texts]
            
        return {
            'data': [
                {'embedding': [0.1, 0.2, 0.3] * 341} for _ in texts  # 1023 dimensions
            ],
            'model': 'fake-embedding-model',
            'usage': {'prompt_tokens': 0, 'total_tokens': 0}
        }


class FakeQdrantClient:
    """Mock client for Qdrant vector store"""
    def __init__(self, *args, **kwargs):
        self.collections = {}
        self.points = {}
    
    def get_collection(self, collection_name):
        if collection_name not in self.collections:
            raise Exception(f"Collection {collection_name} does not exist")
        return {"name": collection_name}
    
    def create_collection(self, collection_name, vectors_config):
        self.collections[collection_name] = vectors_config
        self.points[collection_name] = []
        return True
    
    def upsert(self, collection_name, points, **kwargs):
        if collection_name not in self.points:
            self.points[collection_name] = []
        self.points[collection_name].extend(points)
        return {"status": "Success"}
    
    def search(self, collection_name, query_vector, limit=5, **kwargs):
        """Return fake search results"""
        class FakeScoredPoint:
            def __init__(self, id, payload, score):
                self.id = id
                self.payload = payload
                self.score = score
        
        # Return some fake results
        return [
            FakeScoredPoint(
                id="1",
                payload={"text": "This is a test document about RAG systems."},
                score=0.95
            ),
            FakeScoredPoint(
                id="2",
                payload={"text": "Vector databases store embeddings for search."},
                score=0.85
            ),
            FakeScoredPoint(
                id="3",
                payload={"text": "LLMs can generate text based on context."},
                score=0.75
            )
        ][:limit]


class FakeCompletionAPI:
    """Mock client for LLM API"""
    def __init__(self, *args, **kwargs):
        pass
    
    def invoke(self, *args, **kwargs):
        """Return fake LLM response"""
        return {"content": "This is a fake response from the LLM model."}
    
    def stream(self, *args, **kwargs):
        """Return fake streaming response"""
        tokens = list("This is a fake response from the LLM model.")
        for token in tokens:
            yield {"choices": [{"delta": {"content": token}}]}


# Fixtures for unit testing
@pytest.fixture
def fake_llama_client():
    return FakeLlamaClient()


@pytest.fixture
def fake_qdrant_client():
    return FakeQdrantClient()


@pytest.fixture
def fake_completion_api():
    return FakeCompletionAPI()


@pytest.fixture
def embedding_generator(fake_llama_client):
    """Mock embedding generator for unit tests"""
    class MockEmbeddingGenerator(EmbeddingGenerator):
        def __init__(self):
            # Skip initialization of real model
            self.model = fake_llama_client
        
        def generate_embeddings(self, texts):
            """Return fake embeddings"""
            if isinstance(texts, str):
                return [[0.1, 0.2, 0.3] * 341]  # 1023 dimensions
                
            return [[0.1, 0.2, 0.3] * 341 for _ in texts]
    
    return MockEmbeddingGenerator()


@pytest.fixture
def vector_store(fake_qdrant_client):
    """Mock vector store for unit tests"""
    class MockVectorStore(VectorStore):
        def __init__(self):
            # Skip initialization of real client
            self.collection_name = "test_collection"
            self.db_path = None
            self.client = fake_qdrant_client
        
        def ensure_collection_exists(self, vector_size=1024):
            # Just pretend collection exists
            return True
        
        def store(self, texts, embeddings):
            # Mock storage operation
            return {"status": "success"}
        
        def search(self, query_vector, limit=5, filter_params=None):
            # Return fake search results from mock client
            return self.client.search(self.collection_name, query_vector, limit)
    
    return MockVectorStore()


@pytest.fixture
def llm_generator(embedding_generator, vector_store, memory_db, fake_completion_api):
    """Mock LLM generator for unit tests"""
    class MockAgent(Agent):
        def __init__(self):
            # Skip initialization of real components
            self.vector_store = vector_store
            self.memory_db = memory_db
            self.embedding_generator = embedding_generator
            self.generations = {}
            
        def generate_response(self, conversation_id, query):
            """Mock response generation"""
            tokens = list("This is a mock response from the LLM.")
            self.generations[conversation_id] = {
                "tokens": tokens,
                "is_complete": True,
                "final_response": "".join(tokens)
            }
            # Update memory
            self._update_memory(conversation_id, query, "".join(tokens))
            
        def _retrieve_context(self, conversation_id, query):
            """Mock context retrieval"""
            return "Mock context for RAG retrieval."
            
        def generate_tokens(self, conversation_id):
            """Generator to yield tokens"""
            if conversation_id not in self.generations:
                yield None
                return
                
            for token in self.generations[conversation_id]["tokens"]:
                yield token
    
    return MockAgent()