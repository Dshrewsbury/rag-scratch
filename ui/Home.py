import streamlit as st


def main():
    st.title("AI Chat Assistant")

    st.write("""
    Welcome to the AI Chat Assistant! This application provides:
    
    - Advanced conversational AI powered by LLaMA-2
    - Vector-based memory retrieval
    - Real-time streaming responses
    - Multiple conversation management
    
    Navigate using the sidebar to:
    - Chat: Start conversations with the AI
    - Settings: Configure your chat settings
    - About: Learn more about the application
    """)

    # Add some example features or quick start buttons
    st.subheader("Quick Start")
    if st.button("Start New Chat"):
        st.switch_page("pages/1_Chat.py")

    if st.button("Configure Settings"):
        st.switch_page("pages/2_Settings.py")


if __name__ == "__main__":
    main()
