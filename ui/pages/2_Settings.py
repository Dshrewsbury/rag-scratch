"""
Settings management page for the RAG Chat Assistant.

This module provides a UI for configuring the backend model settings,
including model selection and API endpoint configuration.
"""
import requests  # type: ignore
import streamlit as st

BACKEND_URL = "http://localhost:8000"

def update_backend_model(selected_model, api_base_url):
    """
    Update the backend model configuration via API call.
    
    Args:
        selected_model (str): The name of the model to use (e.g., "gpt-4")
        api_base_url (str): The base URL for the API endpoint
        
    Returns:
        None: Displays success or error message in the UI
    """
    payload = {"provider": "openai", "model": selected_model, "api_base": api_base_url}
    response = requests.post(f"{BACKEND_URL}/api/model_config", json=payload)
    
    if response.status_code == 200:
        st.success("Model settings updated successfully!")
    else:
        st.error("Failed to update model settings.")

def main():
    """
    Render the settings page with configuration options.
    
    Provides UI controls for:
    - Model selection dropdown
    - API base URL configuration
    - Apply changes button
    """
    st.title("Settings")
    
    model = st.selectbox(
        "Choose AI Model", ["gpt-4-mini", "gpt-4", "gpt-3.5-turbo"], key="openai_model"
    )
    
    api_base = st.text_input(
        "API Base URL", value="https://models.inference.ai.azure.com", key="api_base"
    )
    
    if st.button("Apply Changes"):
        update_backend_model(model, api_base)

if __name__ == "__main__":
    main()