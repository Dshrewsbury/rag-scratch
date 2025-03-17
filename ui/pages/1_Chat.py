import streamlit as st
import requests
import json
import os
from urllib.parse import quote

# Get the backend URL from environment or use default
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

def chat_interface():
    """Main chat interface for interacting with the RAG system."""
    
    # Initialize chat session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Get available conversations
    try:
        response = requests.get(f"{BACKEND_URL}/api/conversations")
        conversations = response.json().get('conversations', [])
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        conversations = []
    
    # Create sidebar for conversation selection
    with st.sidebar:
        st.title("Conversations")
        
        # New conversation button
        if st.button("New Conversation"):
            try:
                response = requests.post(f"{BACKEND_URL}/api/conversation")
                new_conv_id = response.json()['conversation_id']
                st.session_state.conversation_id = new_conv_id
                st.session_state.messages = []
                st.success("Created new conversation")
                st.rerun()
            except Exception as e:
                st.error(f"Error creating conversation: {str(e)}")
        
        # List existing conversations
        for conv in conversations:
            col1, col2 = st.columns([4, 1])  # Adjust ratio as needed
            
            # Conversation button in first column
            with col1:
                if st.button(
                    f"{conv['title']} ({conv['message_count']} messages)", 
                    key=f"conv_{conv['id']}"
                ):
                    st.session_state.conversation_id = conv['id']
                    st.session_state.messages = []
                    st.success(f"Switched to conversation: {conv['title']}")
                    st.rerun()
            
            # Delete button in second column
            with col2:
                if st.button("🗑️", key=f"del_{conv['id']}", help="Delete conversation"):
                    # Show confirmation dialog using session state
                    st.session_state[f"confirm_delete_{conv['id']}"] = True
                    
            # Handle deletion confirmation
            if st.session_state.get(f"confirm_delete_{conv['id']}", False):
                confirm_col1, confirm_col2 = st.columns([1, 1])
                with confirm_col1:
                    if st.button("Confirm delete", key=f"confirm_yes_{conv['id']}"):
                        try:
                            # Make DELETE request to backend
                            response = requests.delete(f"{BACKEND_URL}/api/conversation/{conv['id']}")
                            if response.status_code == 200:
                                # Clear confirmation state
                                st.session_state[f"confirm_delete_{conv['id']}"] = False
                                # If deleted conversation was selected, clear selection
                                if st.session_state.get('conversation_id') == conv['id']:
                                    st.session_state.conversation_id = None
                                    st.session_state.messages = []
                                st.success(f"Deleted conversation: {conv['title']}")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting conversation: {str(e)}")
                with confirm_col2:
                    if st.button("Cancel", key=f"confirm_no_{conv['id']}"):
                        st.session_state[f"confirm_delete_{conv['id']}"] = False
                        st.rerun()
    
    # Set default conversation if none selected
    if "conversation_id" not in st.session_state and conversations:
        st.session_state.conversation_id = conversations[0]['id']
    elif "conversation_id" not in st.session_state and not conversations:
        # Create first conversation if none exist
        try:
            response = requests.post(f"{BACKEND_URL}/api/conversation")
            st.session_state.conversation_id = response.json()['conversation_id']
        except Exception as e:
            st.error(f"Error creating conversation: {str(e)}")
            return

    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Message the assistant..."):
        # Add user message to state and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        is_first_message = len(st.session_state.messages) == 1

        # Process with streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Stream response from the backend
            encoded_message = quote(prompt)
            stream_url = f"{BACKEND_URL}/api/stream/{st.session_state.conversation_id}/{encoded_message}"
            
            try:
                with requests.get(stream_url, stream=True) as response:
                    if response.status_code != 200:
                        st.error(f"Error: Received status code {response.status_code}")
                        return
                        
                    # Process the streaming response
                    for line in response.iter_lines():
                        if not line:
                            continue
                            
                        line_text = line.decode('utf-8')
                        if line_text.startswith("data: "):
                            try:
                                # Parse the JSON data
                                data = json.loads(line_text[6:])  # Remove "data: " prefix
                                
                                # Handle token
                                if "token" in data:
                                    token = data["token"]
                                    full_response += token
                                    # Display with cursor
                                    message_placeholder.markdown(full_response + "▌")
                                
                                # Handle completion
                                if "done" in data and data["done"]:
                                    # Update with final response
                                    message_placeholder.markdown(full_response)
                                    # Save to session state
                                    st.session_state.messages.append({
                                        "role": "assistant", 
                                        "content": full_response
                                    })
                                    break
                                    
                                # Handle errors
                                if "error" in data:
                                    st.error(data)
                                    break
                                    
                            except json.JSONDecodeError:
                                st.error(f"Invalid response format: {line_text}")
                                
                    if is_first_message:
                        try:
                            # Generate a title based on the first message
                            title_url = f"{BACKEND_URL}/api/generate_title/{st.session_state.conversation_id}"
                            title_response = requests.post(title_url)
                            if title_response.status_code == 200:
                                title_data = title_response.json()
                                if "title" in title_data:
                                    st.success(f"Created conversation: {title_data['title']}")
                                st.rerun()  # Refresh to show the new title
                        except Exception as e:
                            st.error(f"Error generating title: {str(e)}")                        
            except Exception as e:
                st.error(f"Connection error: {str(e)}")

def main():
    """Initialize the Streamlit app."""
    st.title("RAG Chat Assistant")
    
    # Show model information
    try:
        response = requests.get(f"{BACKEND_URL}/api/model_info")
        if response.status_code == 200:
            model_info = response.json()
            st.caption(f"Using model: {model_info['model']}")
    except:
        st.caption("Connected to RAG backend")
    
    # Display the chat interface
    chat_interface()

if __name__ == "__main__":
    main()