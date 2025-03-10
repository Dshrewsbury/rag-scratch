import pytest
import time
import os
from testcontainers.compose import DockerCompose
from config.settings import VECTOR_DB_PATH, EMBEDDINGS_DIR

# @pytest.fixture(scope="session", autouse=True)
# def setup_test_environment():
#     """Setup necessary directories for tests"""
#     os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
#     #os.makedirs(VECTOR_DB_PATH, exist_ok=True)
#     # Possibly initialize an empty Qdrant collection here
#     print(f"Test environment setup complete. VECTOR_DB_PATH={VECTOR_DB_PATH}")

#     # Remove existing vector DB if it exists
#     if os.path.exists(VECTOR_DB_PATH):
#         shutil.rmtree(VECTOR_DB_PATH)
    
#     # Create a fresh directory
#     os.makedirs(VECTOR_DB_PATH, exist_ok=True)
#     yield
#     # Optional cleanup code here

@pytest.fixture(scope="module")
def compose():
    with DockerCompose(".") as compose:
        time.sleep(20)
        yield compose