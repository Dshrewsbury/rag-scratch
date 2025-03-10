import pytest
import time
import asyncio
import sys

def test_llm_generator_with_vector_store(llm_generator):
 
    """Test integration between LLM generator and vector retrieval system"""
    # Create a conversation ID
    conversation_id = "test-conversation-1"
    
    # Generate a response about RAG (should retrieve from vector store)
    query = "Explain RAG systems to me"
    
    # Override the _retrieve_context method to verify it's called
    original_retrieve = llm_generator._retrieve_context
    context_retrieved = []
    
    def mock_retrieve(conv_id, q):
        context = original_retrieve(conv_id, q)
        context_retrieved.append(context)
        return context
    
    llm_generator._retrieve_context = mock_retrieve
    
    # Execute generation
    llm_generator.generate_response(conversation_id, query)
    
    # Give it time to process
    time.sleep(1)
    
    # Verify context was retrieved
    #assert len(context_retrieved) > 0
    
    # Restore original method
    llm_generator._retrieve_context = original_retrieve
    
    # Verify generation happened
    assert conversation_id in llm_generator.generations
    
    # Check generation has tokens
    #assert len(llm_generator.generations[conversation_id]["tokens"]) > 0
    
    # If using a mock, we can't check content quality, but we can check it completed
    #assert llm_generator.generations[conversation_id]["is_complete"]


# def test_streaming_token_generation(llm_generator):

#     import logging
#     logger = logging.getLogger("tests")
    
#     logger.debug(f"Vector store URL: {llm_generator.vector_store.url}")
#     logger.debug(f"Vector store client: {llm_generator.vector_store.client}")

#     """Test token streaming with the generate_tokens generator"""
#     # Create conversation
#     conversation_id = "test-conversation-streaming"
    
#     # Generate a response
#     llm_generator.generate_response(conversation_id, "Tell me about embeddings")
    
#     # Start streaming tokens
#     tokens = []
    
#     # Set a timeout to avoid infinite loops
#     start_time = time.time()
#     timeout = 3  # seconds
    
#     # Collect tokens while not complete and not timed out
#     while not llm_generator.generations[conversation_id]["is_complete"]:
#         if time.time() - start_time > timeout:
#             break
            
#         # Get next token
#         token_gen = llm_generator.generate_tokens(conversation_id)
#         token = next(token_gen, None)
        
#         if token is not None:
#             tokens.append(token)
            
#         time.sleep(0.1)
    
#     # We should have collected some tokens
#     assert len(tokens) > 0
    
#     # The concatenated tokens should match the final response
#     expected = llm_generator.generations[conversation_id]["final_response"]
#     if expected:  # If final response is not empty
#         assert ''.join(tokens) in expected  # Tokens might be a subset if we timed out


# def test_memory_database_integration(llm_generator, memory_db):

#     import logging
#     logger = logging.getLogger("tests")
    
#     logger.debug(f"Vector store URL: {llm_generator.vector_store.url}")
#     logger.debug(f"Vector store client: {llm_generator.vector_store.client}")

#     """Test integration with the memory database"""
#     # Create conversation
#     conversation_id = "test-conversation-memory"
    
#     # Generate a response
#     query = "What's the weather like today?"
#     llm_generator.generate_response(conversation_id, query)
    
#     # Give time for memory to be updated
#     time.sleep(1)
    
#     # Check if memory has the message
#     messages = memory_db.get_messages(conversation_id)
    
#     # Should have at least the user query
#     assert any(m["role"] == "user" and m["content"] == query for m in messages)
    
#     # If the system processes it quickly enough, should also have assistant response
#     assert any(m["role"] == "assistant" for m in messages)