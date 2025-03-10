from fastapi.testclient import TestClient
from app.api import app as fastapi_app
from core.agent.agent import Agent
from config.settings import LLM_MODEL_PATH

# Create a global llm_generator instance for testing
llm_generator = Agent(LLM_MODEL_PATH)

# Override the global variable in the app
import app.api
app.api.llm_generator = llm_generator

client = TestClient(fastapi_app)

def test_send_message(compose):
    # Create a conversation first
    response = client.post("/api/conversation")
    conversation_id = response.json()["conversation_id"]
    
    # Test sending a message
    response = client.post(
        "/api/message", 
        json={
            "conversation_id": conversation_id,
            "content": "Hello, world!"
        }
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Hello, world!"