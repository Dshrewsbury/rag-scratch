"""
About page for the RAG Chat Assistant application.

This module provides information about the application, its features,
and the technology stack used in its implementation.
"""
import streamlit as st

def main():
    """
    Render the about page with application information.
    
    Displays details about:
    - Application overview
    - Key features
    - Technology stack
    - Version information
    """
    st.title("About")
    st.write("""
    ## AI Chat Assistant
   
    This application combines the power of LLaMA-2 with vector-based memory retrieval
    to provide an intelligent conversational experience.
   
    ### Features
    - Real-time streaming responses
    - Long-term memory using vector embeddings
    - Multiple conversation management
    - Customizable settings
   
    ### Technology Stack
    - Frontend: Streamlit
    - Backend: FastAPI
    - AI Model: LLaMA-2
    - Vector Store: Qdrant
   
    ### Version
    1.0.0
    """)

if __name__ == "__main__":
    main()