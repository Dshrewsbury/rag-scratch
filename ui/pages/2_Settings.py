import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

def update_backend_model(selected_model, api_base_url):
    payload = {
        "provider": "openai",
        "model": selected_model,
        "api_base": api_base_url
    }
    response = requests.post(f"{BACKEND_URL}/api/model_config", json=payload)
    if response.status_code == 200:
        st.success("Model settings updated successfully!")
    else:
        st.error("Failed to update model settings.")

def main():
    st.title("Settings")
    
    # Model selection
    model = st.selectbox(
        "Choose AI Model",
        ["gpt-4-mini", "gpt-4", "gpt-3.5-turbo"],
        key="openai_model"
    )
    
    # API Configuration
    api_base = st.text_input("API Base URL", 
                 value="https://models.inference.ai.azure.com",
                 key="api_base")
    
     # Apply button
    if st.button("Apply Changes"):
        update_backend_model(model, api_base)
    
if __name__ == "__main__":
    main()
