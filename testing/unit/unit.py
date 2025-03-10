import pytest

def test_generate_response(llm_generator):
    """Test that generate_response creates tokens and updates state"""
    # Define test data
    conversation_id = "test-conversation"
    query = "Tell me about RAG systems"
    
    # Call the method
    llm_generator.generate_response(conversation_id, query)
    
    # Verify state was updated correctly
    assert conversation_id in llm_generator.generations
    assert llm_generator.generations[conversation_id]["is_complete"]
    assert len(llm_generator.generations[conversation_id]["tokens"]) > 0
    assert llm_generator.generations[conversation_id]["final_response"] != ""


def test_update_memory(llm_generator):
    """Test that _update_memory adds messages to memory database"""
    # Define test data
    conversation_id = "test-conversation"
    query = "What is vector search?"
    response = "Vector search is a technique for finding similar vectors."
    
    # Call the method
    llm_generator._update_memory(conversation_id, query, response)
    
    # Verify memory was updated
    messages = llm_generator.memory_db.get_messages(conversation_id)
    
    # Find user message
    user_message = next((m for m in messages if m["role"] == "user"), None)
    assert user_message is not None
    assert user_message["content"] == query
    
    # Find assistant message
    assistant_message = next((m for m in messages if m["role"] == "assistant"), None)
    assert assistant_message is not None
    assert assistant_message["content"] == response


def test_retrieve_context(llm_generator, embedding_generator, vector_store):
    """Test that _retrieve_context returns context for the query"""
    # Define test data
    conversation_id = "test-conversation"
    query = "Tell me about vector databases"
    
    # Mock the embedding generator to return a specific embedding
    original_generate = embedding_generator.generate_embeddings
    embedding_generator.generate_embeddings = lambda x: [[0.5] * 1023]
    
    # Call the method
    context = llm_generator._retrieve_context(conversation_id, query)
    
    # Verify context was retrieved
    assert context is not None
    assert isinstance(context, str)
    
    # Restore original method
    embedding_generator.generate_embeddings = original_generate


def test_generate_tokens(llm_generator):
    """Test that generate_tokens yields tokens one by one"""
    # Define test data
    conversation_id = "test-conversation"
    query = "How do RAG systems work?"
    
    # First generate a response
    llm_generator.generate_response(conversation_id, query)
    
    # Now test token generation
    tokens = []
    for token in llm_generator.generate_tokens(conversation_id):
        tokens.append(token)
    
    # Verify all tokens were yielded
    assert len(tokens) == len(llm_generator.generations[conversation_id]["tokens"])
    assert ''.join(tokens) == llm_generator.generations[conversation_id]["final_response"]


def test_nonexistent_conversation(llm_generator):
    """Test behavior with non-existent conversation ID"""
    # Try to get tokens for a conversation that doesn't exist
    token = next(llm_generator.generate_tokens("nonexistent-conversation"), None)
    
    # Should return None for non-existent conversations
    assert token is None