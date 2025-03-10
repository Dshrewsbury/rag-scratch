import pytest
import os
import random
from testcontainers.qdrant import QdrantContainer
from testcontainers.generic import ServerContainer
from testcontainers.core.image import DockerImage
from testcontainers.core.waiting_utils import wait_for_logs

from rag.retrieval.vector_store import VectorStore
from rag.agent.agent import Agent
from rag.processing.embedding.embedding_generator import EmbeddingGenerator



# Test data
@pytest.fixture(scope="module")
def sample_docs():
    return [
        "Retrieval Augmented Generation (RAG) combines retrieval with generation for better responses.",
        "Vector databases like Qdrant store document embeddings for semantic search capabilities.",
        "Embedding models convert text into numerical vectors that represent semantic meaning.",
        "LLMs can generate responses based on context provided from retrieved documents."
    ]


# # Containers
@pytest.fixture(scope="module")
def qdrant_container():
    with QdrantContainer("qdrant/qdrant:latest") as container:
        port = container.get_exposed_port(6333)
        yield f"http://localhost:{port}"


# For mocking an LLM API if you're using one. Adjust as needed for your specific setup.
@pytest.fixture(scope="module")
def server_container():
 with DockerImage(path="./", 
                    tag="app-server:test",
                    dockerfile_path="app/") as image:
        # Create container from built image
        with ServerContainer(str(image)).\
                with_exposed_ports(8000) as container:
            wait_for_logs(container, "Application startup complete")  # Adjust this log message
            port = container.get_exposed_port(8000)
            yield f"http://localhost:{port}"


# Components with real service integration
@pytest.fixture(scope="function")
def vector_store():
    """Create a vector store with in-memory storage for isolation"""
    import logging
    import uuid
    logger = logging.getLogger("tests")
    
    logger.debug("Creating test vector store with in-memory storage")
    
    # Initialize with in-memory storage
    vector_store = VectorStore(
        collection_name=f"test_collection_{uuid.uuid4()}",  # Unique collection name
        in_memory=True  # Use in-memory storage
    )
    
    vector_store.ensure_collection_exists(vector_size=1024)
    
    # Log the configuration
    logger.debug(f"Vector store created with collection: {vector_store.collection_name}")
    
    yield vector_store
    
    # No need to clean up in-memory storage, it disappears when the client is destroyed

@pytest.fixture(scope="function")
def embedding_generator():
    # Use a mock embedding generator for integration testing
    class MockEmbeddingGenerator:
        def generate_embeddings(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            # Generate stable random embeddings (same input = same embedding)
            return [
                [random.uniform(0.0, 1.0) for _ in range(1024)] 
                for _ in texts
            ]
    return MockEmbeddingGenerator()


@pytest.fixture(scope="function")
def llm_generator(vector_store, memory_db, embedding_generator):
    # Create LLM generator with injected dependencies
    generator = Agent(
        vector_store=vector_store,
        memory_db=memory_db,
        embedding_generator=embedding_generator
    )
    return generator


# Utility function to load test data
@pytest.fixture(scope="function")
def populate_vector_store(vector_store, embedding_generator, sample_docs):
    # Generate embeddings
    embeddings = embedding_generator.generate_embeddings(sample_docs)
    
    # Store in vector database
    vector_store.store(sample_docs, embeddings)
    
    return vector_store