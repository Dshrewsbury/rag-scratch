import datetime
import json
import os
import traceback
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from urllib.parse import unquote

from rag.agent.agent import Agent
from config.settings import LLM_MODEL_PATH

class Message(BaseModel):
    conversation_id: str
    content: str

class ModelConfig(BaseModel):
    model: str = "gpt-4o"
    
    class Config:
        extra = "allow"

# Read model configuration from environment or use defaults
def get_model_config() -> Dict[str, Any]:
    """
    Extract model configuration from environment variables.
    
    Returns:
        Dictionary of model configuration
    """
    model_type = os.environ.get("MODEL_TYPE", "openai").lower()
    
    config = {
        "model": os.environ.get("LLM_MODEL", "gpt-4o"),
    }
    
    # Add provider-specific configuration
    if model_type == "openai":
        if os.environ.get("OPENAI_API_KEY"):
            config["api_key"] = os.environ.get("OPENAI_API_KEY")
        if os.environ.get("OPENAI_API_BASE"):
            config["api_base"] = os.environ.get("OPENAI_API_BASE")
            
    elif model_type == "ollama":
        # Format: ollama/model_name
        config["model"] = f"ollama/{os.environ.get('LLM_MODEL', 'llama2')}"
        if os.environ.get("OLLAMA_API_BASE"):
            config["api_base"] = os.environ.get("OLLAMA_API_BASE")
            
    elif model_type == "anthropic":
        # Format: anthropic/claude-3-opus
        if not config["model"].startswith("anthropic/"):
            config["model"] = f"anthropic/{config['model']}"
        if os.environ.get("ANTHROPIC_API_KEY"):
            config["api_key"] = os.environ.get("ANTHROPIC_API_KEY")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize LLM generator on startup and cleanup on shutdown.
    """
    # Initialize LLM generator with configuration
    global agent
    model_config = get_model_config()
    agent = Agent(model_config)
    yield


app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker."""
    return {
        "status": "healthy", 
        "timestamp": datetime.datetime.now().isoformat(),
        "model": agent.model_config.get("model", "unknown")
    }

@app.post('/api/message')
async def handle_message(message: Message):
    """
    Process a message and return the complete response.
    
    Args:
        message: User message with conversation ID and content
    
    Returns:
        Complete response
    """
    try:
        response = await agent.generate_response(
            message.conversation_id,
            message.content
        )
        
        return {
            'conversation_id': message.conversation_id,
            'message': message.content,
            'response': response,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/stream/{conversation_id}/{message}')
async def stream_response(conversation_id: str, message: str):
    """
    Stream the response for a message.
    
    Args:
        conversation_id: ID of the conversation
        message: URL-encoded user message
    
    Returns:
        Server-sent events stream with tokens
    """
    # Decode the URL-encoded message
    decoded_message = unquote(message)
    
    async def event_generator():
        """Generate server-sent events for each token."""
        try:
            async for token in agent.generate_response_stream(
                conversation_id,
                decoded_message
            ):
                # Yield token as a server-sent event
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n"
                    
            # Signal completion
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            # Send error event
            error_traceback = traceback.format_exc()
            yield f"data: {json.dumps({'error': error_traceback})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

@app.post('/api/conversation')
async def create_conversation():
    """Create a new conversation and return its ID."""
    conversation_id = agent.memory_db.create_conversation()
    return {
        'conversation_id': conversation_id,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

@app.get('/api/conversations')
async def get_conversations():
    """Get list of all conversations."""
    try:
        conversations = agent.memory_db.get_conversation_history()
        return {
            'conversations': [
                {
                    'id': conv['conversation_id'],
                    'title': conv.get('title') or f"Conversation {conv['conversation_id']}",
                    'message_count': conv.get('message_count', 0),
                    'start_time': conv.get('start_time')
                }
                for conv in conversations or []
            ]
        }
    except Exception as e:
        return {'conversations': [], 'error': str(e)}

@app.delete('/api/conversation/{conversation_id}')
async def delete_conversation(conversation_id: str):
    """Delete a conversation and its messages."""
    success = agent.memory_db.delete_conversation(conversation_id)
    if success:
        return {"status": "success", "message": f"Conversation {conversation_id} deleted"}
    else:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to delete conversation {conversation_id}"
        )

@app.get('/api/model_info')
async def get_model_info():
    """Get information about the current model configuration."""
    return {
        "model": agent.model_config.get("model", "unknown"),
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info") 