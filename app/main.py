"""
RAG-powered API server that provides conversational AI capabilities.

This FastAPI application integrates with a RAG (Retrieval-Augmented Generation) agent
to provide conversational capabilities via a RESTful API. The application supports
streaming responses, conversation management, and model configuration.
"""

import datetime
import json
import os
import traceback
from contextlib import asynccontextmanager
from typing import Optional
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from rag.agent.agent import Agent

from app.models.models import (
    AgentResponse,
    HealthResponse,
    MessageRequest,
    ModelConfigData,
    ModelConfigResponse,
)

agent: Optional[Agent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the lifecycle of the RAG agent.

    Initializes the Agent on application startup and performs cleanup on shutdown.
    Uses environment variables to configure the underlying language model.

    Args:
        app: FastAPI application instance
    """
    global agent 
    agent = Agent(
        model_config={
            "provider": os.getenv("MODEL_TYPE", "openai").lower(),
            "model": os.getenv("LLM_MODEL", "gpt-4o"),
            "api_base": os.getenv("API_BASE", "https://models.inference.ai.azure.com"),
        }
    )
    yield
    del agent  # Clean up when app shuts down


app = FastAPI(lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check the health status of the API service.

    Returns:
        HealthResponse: Object containing health status, timestamp, and model information
    """
    return HealthResponse(
        status="healthy",
        timestamp=HealthResponse.now(),
        model=agent.model_config.get("model", "unknown"),
    )


@app.post("/api/generate_title/{conversation_id}")
async def generate_title(conversation_id: str):
    """
    Generate a title for a conversation based on its first user message.

    Uses the agent's LLM to create a concise, descriptive title and updates
    the conversation record in the database.

    Args:
        conversation_id: Unique identifier for the conversation

    Returns:
        dict: Contains success status and generated title, or error information
    """
    try:
        assert agent is not None
        messages = agent.memory_db.get_messages(conversation_id)
        if not messages or len(messages) < 1:
            return {"error": "No messages found in conversation"}

        # Find the first user message
        first_user_message = None
        for role, message, _ in messages:
            if role == "user":
                first_user_message = message
                break

        if not first_user_message:
            return {"error": "No user message found"}

        system_prompt = "Generate a concise, descriptive title (4-8 words) for a conversation that starts with this message. Return ONLY the title text with no quotes or additional explanation."

        title = await agent.generate_title(system_prompt, first_user_message)

        success = agent.memory_db.update_conversation_title(conversation_id, title)

        print(f"Title generated: '{title}', update success: {success}")

        return {"success": success, "title": title}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/message", response_model=AgentResponse)
async def handle_message(message: MessageRequest):
    """
    Process a user message and generate a response.

    Args:
        message: Contains conversation_id and content of the user message

    Returns:
        AgentResponse: Object containing the original message, generated response,
                      conversation ID, and timestamp

    Raises:
        HTTPException: If an error occurs during processing
    """
    try:
        assert agent is not None
        response_text = await agent.generate_response(
            message.conversation_id, message.content
        )
        return AgentResponse(
            conversation_id=message.conversation_id,
            message=message.content,
            response=response_text,
            timestamp=AgentResponse.now(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stream/{conversation_id}/{message}")
async def stream_response(conversation_id: str, message: str):
    """
    Stream the AI response for a message token by token.

    Uses server-sent events to stream each token of the response to the client
    as it's generated, enabling real-time display of the response.

    Args:
        conversation_id: ID of the conversation
        message: URL-encoded user message

    Returns:
        StreamingResponse: Server-sent events stream with tokens
    """
    decoded_message = unquote(message)

    async def event_generator():
        """Generate server-sent events for each token."""
        try:
            async for token in agent.generate_response_stream(
                conversation_id, decoded_message
            ):
                # Yield token as a server-sent event
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n"

            # Signal completion
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception:
            # Send error event
            error_traceback = traceback.format_exc()
            yield f"data: {json.dumps({'error': error_traceback})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/model_config", response_model=ModelConfigResponse)
async def set_model_config(config: ModelConfigData):
    """
    Update the configuration of the underlying language model.

    Args:
        config: Contains provider, model name, and API base URL

    Returns:
        ModelConfigResponse: The updated configuration with timestamp
    """
    assert agent is not None
    agent.model_config = {
        "provider": config.provider,
        "model": config.model,
        "api_base": config.api_base,
    }
    return ModelConfigResponse(
        provider=config.provider,
        model=config.model,
        api_base=config.api_base,
        timestamp=ModelConfigResponse.now(),
        api_key_set=True,
    )


@app.post("/api/conversation")
async def create_conversation():
    """
    Create a new conversation and return its ID.

    Returns:
        dict: Contains the new conversation ID and creation timestamp
    """
    conversation_id = agent.memory_db.create_conversation()
    return {
        "conversation_id": conversation_id,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/api/conversations")
async def get_conversations():
    """
    Retrieve a list of all conversations.

    Returns:
        dict: Contains a list of conversation objects with their metadata,
              or an empty list and error information if retrieval fails
    """
    try:
        conversations = agent.memory_db.get_conversation_history()
        return {
            "conversations": [
                {
                    "id": conv["conversation_id"],
                    "title": conv.get("title")
                    or f"Conversation {conv['conversation_id']}",
                    "message_count": conv.get("message_count", 0),
                    "start_time": conv.get("start_time"),
                }
                for conv in conversations or []
            ]
        }
    except Exception as e:
        return {"conversations": [], "error": str(e)}


@app.delete("/api/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation and its associated messages.

    Args:
        conversation_id: ID of the conversation to delete

    Returns:
        dict: Success message if deletion succeeds

    Raises:
        HTTPException: If deletion fails
    """
    assert agent is not None
    success = agent.memory_db.delete_conversation(conversation_id)
    if success:
        return {
            "status": "success",
            "message": f"Conversation {conversation_id} deleted",
        }
    else:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete conversation {conversation_id}"
        )


@app.get("/api/model_info")
async def get_model_info():
    """
    Get information about the current model configuration.

    Returns:
        dict: Contains model name and current timestamp
    """
    return {
        "model": agent.model_config.get("model", "unknown"),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


if __name__ == "__main__":
    import uvicorn  # type: ignore

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
