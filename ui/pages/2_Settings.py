import streamlit as st

def main():
    st.title("Settings")
    
    # Model selection
    st.selectbox(
        "Choose AI Model",
        ["gpt-4-mini", "gpt-4", "gpt-3.5-turbo"],
        key="openai_model"
    )
    
    # API Configuration
    st.text_input("API Base URL", 
                 value="https://models.inference.ai.azure.com",
                 key="api_base")
    
    # Other settings as needed
    
if __name__ == "__main__":
    main()
