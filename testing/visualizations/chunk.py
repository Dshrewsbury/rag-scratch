import streamlit as st
import pandas as pd
import json
from qdrant_client import QdrantClient
import plotly.express as px
from collections import Counter

# Connect to your Qdrant collection
@st.cache_resource
def get_client():
    url = "http://qdrant:6333"
    return QdrantClient(url=url)

def load_chunks(collection_name, limit=100):
    client = get_client()
    print(collection_name)
    # Scroll through points to get chunks and metadata
    points = client.scroll(
        collection_name=collection_name,
        limit=limit,
        with_payload=True,
        with_vectors=False
    )[0]
    
    return points

def main():
    st.title("Book Chunk Metadata Visualizer")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    collection_name = st.sidebar.text_input("Collection Name", "recursive")
    chunk_limit = st.sidebar.slider("Number of chunks to load", 10, 500, 100)
    
    # Load data
    if st.sidebar.button("Load Chunks"):
        points = load_chunks(collection_name, chunk_limit)
        
        if not points:
            st.error("No chunks found in the collection.")
            return
            
        # Convert to more usable format
        chunks = []
        for point in points:
            chunk_data = {
                "id": point.id,
                "text": point.payload.get("text", ""),
            }
            
            # Extract metadata
            metadata = point.payload.get("metadata", {})
            # Flatten metadata for display
            for key, value in metadata.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        chunk_data[f"metadata_{key}_{subkey}"] = str(subvalue)
                else:
                    chunk_data[f"metadata_{key}"] = str(value)
            
            chunks.append(chunk_data)
        
        # Store in session state
        st.session_state.chunks = chunks
        st.session_state.metadata_keys = [key for key in chunks[0].keys() if key.startswith("metadata_")]
        
        # Create a dataframe
        chunks_df = pd.DataFrame(chunks)
        st.session_state.chunks_df = chunks_df
    
    # Display visualization if data is loaded
    if "chunks" in st.session_state:
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["Chunk Explorer", "Metadata Statistics", "Relationship View"])
        
        with tab1:
            st.header("Chunk Explorer")
            
            # Chunk selection
            selected_chunk_idx = st.selectbox(
                "Select a chunk to inspect:", 
                range(len(st.session_state.chunks)),
                format_func=lambda i: f"Chunk {i+1}: {st.session_state.chunks[i]['text'][:50]}..."
            )
            
            selected_chunk = st.session_state.chunks[selected_chunk_idx]
            
            # Display chunk content
            st.subheader("Chunk Content")
            st.text_area("Text", selected_chunk["text"], height=200)
            
            # Display metadata
            st.subheader("Metadata")
            metadata_df = pd.DataFrame({
                "Key": [k for k in selected_chunk.keys() if k.startswith("metadata_")],
                "Value": [selected_chunk[k] for k in selected_chunk.keys() if k.startswith("metadata_")]
            })
            st.dataframe(metadata_df)
            
            # If there are next/prev chunks, allow navigation
            if "metadata_prev_chunk_id" in selected_chunk and selected_chunk["metadata_prev_chunk_id"] != "None":
                if st.button("View Previous Chunk"):
                    prev_id = selected_chunk["metadata_prev_chunk_id"]
                    # Find the index
                    for i, chunk in enumerate(st.session_state.chunks):
                        if chunk.get("metadata_chunk_id") == prev_id:
                            st.session_state.selected_chunk_idx = i
            
            if "metadata_next_chunk_id" in selected_chunk and selected_chunk["metadata_next_chunk_id"] != "None":
                if st.button("View Next Chunk"):
                    next_id = selected_chunk["metadata_next_chunk_id"]
                    # Find the index
                    for i, chunk in enumerate(st.session_state.chunks):
                        if chunk.get("metadata_chunk_id") == next_id:
                            st.session_state.selected_chunk_idx = i
        
        with tab2:
            st.header("Metadata Statistics")
            
            # Select a metadata field to analyze
            metadata_field = st.selectbox(
                "Select metadata field to analyze:",
                st.session_state.metadata_keys
            )
            
            # Create a histogram or bar chart of values
            if metadata_field:
                values = st.session_state.chunks_df[metadata_field].tolist()
                
                # Handle list values stored as strings
                if all(isinstance(v, str) and v.startswith('[') and v.endswith(']') for v in values if v != "None" and v != "[]"):
                    # Parse string lists and count items
                    all_items = []
                    for v in values:
                        if v != "None" and v != "[]":
                            try:
                                items = json.loads(v.replace("'", "\""))
                                all_items.extend(items)
                            except Exception:
                                pass
                    
                    item_counts = Counter(all_items)
                    fig = px.bar(
                        x=list(item_counts.keys()), 
                        y=list(item_counts.values()),
                        title=f"Distribution of items in {metadata_field}"
                    )
                else:
                    # For regular values
                    value_counts = Counter(v for v in values if v != "None")
                    fig = px.bar(
                        x=list(value_counts.keys()), 
                        y=list(value_counts.values()),
                        title=f"Distribution of {metadata_field}"
                    )
                
                st.plotly_chart(fig)
        
        with tab3:
            st.header("Chunk Relationships")
            
            # Create a simple network visualization of chunks
            if "metadata_chunk_id" in st.session_state.chunks_df.columns:
                # Create nodes
                nodes = []
                for i, chunk in enumerate(st.session_state.chunks):
                    if "metadata_chunk_id" in chunk:
                        chunk_id = chunk["metadata_chunk_id"]
                        chapter = chunk.get("metadata_chapter", "Unknown")
                        
                        nodes.append({
                            "id": chunk_id,
                            "label": f"Chunk {chunk_id}",
                            "title": chunk["text"][:50],
                            "group": chapter
                        })
                
                # Create edges based on next/prev relationships
                edges = []
                for chunk in st.session_state.chunks:
                    if "metadata_chunk_id" in chunk and "metadata_next_chunk_id" in chunk:
                        source = chunk["metadata_chunk_id"]
                        target = chunk["metadata_next_chunk_id"]
                        
                        if target != "None":
                            edges.append({
                                "from": source,
                                "to": target
                            })
                
                # Use NetworkX or Pyvis for network visualization
                # For this example, we'll use a simpler Plotly scatter plot
                # to show chunk relationships by chapter
                
                if "metadata_chapter" in st.session_state.chunks_df.columns:
                    fig = px.scatter(
                        st.session_state.chunks_df,
                        x="metadata_chunk_id",
                        y="metadata_chapter",
                        hover_data=["text"],
                        title="Chunks by Chapter"
                    )
                    st.plotly_chart(fig)
                else:
                    st.write("No chapter metadata available for relationship view")

if __name__ == "__main__":
    main()