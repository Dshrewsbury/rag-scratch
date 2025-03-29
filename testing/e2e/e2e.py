from fastapi.testclient import TestClient
from app.api import app as fastapi_app
from rag.agent.agent import Agent
from config.settings import LLM_MODEL_PATH

# Create a global llm_generator instance for testing
llm_generator = Agent(LLM_MODEL_PATH)

# Override the global variable in the app
import app.api
app.api.llm_generator = llm_generator

client = TestClient(fastapi_app)

def test_stream_response():
    # Create a conversation first
    response = client.post("/api/conversation")
    conversation_id = response.json()["conversation_id"]
    
    # Test streaming a response
    response = client.get(f"/api/stream/{conversation_id}/Hello%20world")
    
    # Check it's a streaming response
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    
    # Parse the response content
    content = response.content.decode()
    assert "data:" in content