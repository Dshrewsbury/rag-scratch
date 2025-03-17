import datetime
import json
import os
import traceback
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from urllib.parse import unquote

from rag.agent.agent import Agent
from app.models.models import (
    MessageRequest,
    AgentResponse,
    HealthResponse,
    ModelConfigData,
    ModelConfigResponse
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize LLM generator on startup and cleanup on shutdown.
    """
    global agent
    agent = Agent(model_config={
        "provider": os.getenv("MODEL_TYPE", "openai").lower(),
        "model": os.getenv("LLM_MODEL", "gpt-4o"),
        "api_base": os.getenv("API_BASE", "https://models.inference.ai.azure.com")
    })
    yield
    del agent  # Clean up when app shuts down


app = FastAPI(lifespan=lifespan)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=HealthResponse.now(),
        model=agent.model_config.get("model", "unknown")
    )

@app.post('/api/generate_title/{conversation_id}')
async def generate_title(conversation_id: str):
    """Generate a title for a conversation based on the first message."""
    try:
        # Get the first message from the conversation
        messages = agent.memory_db.get_messages(conversation_id)
        if not messages or len(messages) < 1:
            return {"error": "No messages found in conversation"}
            
        # Get the first user message
        first_user_message = None
        for role, message, _ in messages:
            if role == "user":
                first_user_message = message
                break
                
        if not first_user_message:
            return {"error": "No user message found"}
            
        # Generate a concise title using the LLM
        system_prompt = "Generate a concise, descriptive title (4-8 words) for a conversation that starts with this message. Return ONLY the title text with no quotes or additional explanation."
        
        title = await agent.generate_title(system_prompt, first_user_message)
        
        # Update the conversation title in the database
        success = agent.memory_db.update_conversation_title(conversation_id, title)

        # Log the result for debugging
        print(f"Title generated: '{title}', update success: {success}")
        
        return {"success": success, "title": title}
    except Exception as e:
        return {"error": str(e)}


@app.post('/api/message', response_model=AgentResponse)
async def handle_message(message: MessageRequest):
    try:
        response_text = await agent.generate_response(
            message.conversation_id,
            message.content
        )
        return AgentResponse(
            conversation_id=message.conversation_id,
            message=message.content,
            response=response_text,
            timestamp=AgentResponse.now()
        )
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


@app.post("/api/model_config", response_model=ModelConfigResponse)
async def set_model_config(config: ModelConfigData):
    agent.model_config = {
        "provider": config.provider,
        "model": config.model,
        "api_base": config.api_base
    }
    return ModelConfigResponse(
        provider=config.provider,
        model=config.model,
        api_base=config.api_base,
        timestamp=ModelConfigResponse.now()
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