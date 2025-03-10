import time
import uuid
import re

from typing import List, Dict, Any, Tuple
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_cpp import Llama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain.docstore.document import Document

import os
import sys
import spacy
#sys.path.append("/Project/src/")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from processing.chunking.utils import chunk
from config.settings import EMBEDDING_MODEL_PATH, DATA_DIR, EMBEDDINGS_DIR, BATCH_SIZE

nlp = spacy.load("en_core_web_sm")

def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract named entities from text using spaCy."""
    doc = nlp(text)
    entities = {
        "characters": [],
        "locations": [],
        "dates": [],
        "misc": []
    }
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["characters"].append(ent.text)
        elif ent.label_ in ["GPE", "LOC"]:
            entities["locations"].append(ent.text)
        elif ent.label_ == "DATE":
            entities["dates"].append(ent.text)
        else:
            entities["misc"].append(ent.text)
    
    return entities

def extract_book_metadata(md_text: str) -> Dict[str, Any]:
    """
    Extract book metadata from markdown text including title, author, series.
    
    Args:
        md_text: Markdown text content
        
    Returns:
        Dictionary containing extracted metadata
    """
    metadata = {
        "title": None,
        "author": None,
        "series": None
    }
    
    # Split by page separator
    pages = md_text.split("-----")
    
    for page in pages:
        page_content = page.strip()
        if not page_content:
            continue
        
        lines = page_content.split("\n")
        for line in lines:
            line = line.strip()
            
            # Extract title (e.g., "# GHOST OF THE SHADOWFORT")
            if line.startswith("# ") and metadata["title"] is None:
                metadata["title"] = line.replace("# ", "").strip()
                
            # Extract series info (e.g., "### THE BLADEBORN SAGA: BOOK TWO")
            elif "SAGA" in line and "BOOK" in line and line.startswith("### "):
                metadata["series"] = line.replace("### ", "").strip()
                
            # Extract author (e.g., "## T. C. EDGE")
            elif line.startswith("## ") and metadata["author"] is None and not line.startswith("## CHAPTER"):
                metadata["author"] = line.replace("## ", "").strip()
    
    return metadata

def parse_markdown_with_metadata(md_text: str) -> Tuple[List[Document], Dict[str, Any]]:
    """
    Parse markdown text and create documents with proper page and chapter metadata.
    
    Args:
        md_text: Markdown text content
        
    Returns:
        Tuple of (list of documents with metadata, book metadata)
    """
    # Extract book metadata first
    book_metadata = extract_book_metadata(md_text)
    
    # Split content by page separator
    pages = md_text.split("-----")
    
    documents = []
    current_chapter = None
    current_page = 0
    
    for i, page in enumerate(pages):
        page_content = page.strip()
        if not page_content:
            continue
        
        # Increment page counter
        current_page += 1
        
        # Check if this page starts a new chapter
        lines = page_content.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("### ") and re.match(r"^### \d+$", line):
                current_chapter = int(line.replace("### ", "").strip())
                break
        
        # Create a document for this page
        doc = Document(
            page_content=page_content,
            metadata={
                "page": current_page,
                "chapter": current_chapter,
                "title": book_metadata.get("title"),
                "author": book_metadata.get("author"),
                "series": book_metadata.get("series")
            }
        )
        
        documents.append(doc)
    
    return documents, book_metadata

def process_document(file_path=None, chunk_size=512, chunk_overlap=50, batch_size=BATCH_SIZE):
    """
    Process a document using recursive character splitting strategy.
    
    Args:
        file_path: Path to the PDF file
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        batch_size: Batch size for processing
        
    Returns:
        List of document-embedding pairs
    """
    # Initialize the language model for embedding generation
    llm = Llama(
        model_path=EMBEDDING_MODEL_PATH,
        embedding=True,
        verbose=False,
        n_batch=batch_size
    )
    
    # Load the document
    file = f"{DATA_DIR}/{file_path}"

    # Get file name without extension
    #file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Create markdown file path
    #md_file_path = f"{DATA_DIR}/{file_name}_converted.md"
    
    # # Convert to markdown and save to file
    md_text = pymupdf4llm.to_markdown(file)
    
    # # Save the markdown text to a file
    # with open(md_file_path, "w", encoding="utf-8") as f:
    #     f.write(md_text)
    
    # print(f"Saved markdown version to: {md_file_path}")

    # Extract book metadata
    # book_metadata = extract_book_metadata(md_file_path)
    # print(f"Extracted book metadata: {book_metadata}")

     # Parse the markdown with metadata
    documents, book_metadata = parse_markdown_with_metadata(md_text)
    print(f"Extracted book metadata: {book_metadata}")
    print(f"Created {len(documents)} page documents")
    
    # Load the markdown file using UnstructuredMarkdownLoader
    # loader = UnstructuredMarkdownLoader(md_file_path)
    # docs = loader.load()
 

     # Split the documents into chunks while preserving metadata
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    chunked_documents = []
    
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for chunk_text in chunks:
            # Create a new document for each chunk with the same metadata
            chunk_doc = Document(
                page_content=chunk_text,
                metadata=doc.metadata.copy()  # Copy metadata from the original document
            )
            chunked_documents.append(chunk_doc)
    
    print(f"Created {len(chunked_documents)} chunks")
    
    # Process in batches
    documents_embeddings = []
    batches = list(chunk(documents, batch_size))
    
    # Generate embeddings
    start = time.time()
    for batch in batches:
        embeddings = llm.create_embedding([item.page_content for item in batch])
        
        documents_embeddings.extend(
            [
                (document, embeddings['embedding'])
                for document, embeddings in zip(batch, embeddings['data'])
            ]
        )
        break # SLOW AS HELL
    end = time.time()
    
    # Calculate processing speed
    char_per_second = len(''.join([item.page_content for item in documents])) / (end - start)
    print(f"Time taken: {end - start:.2f} seconds / {char_per_second:,.2f} characters per second")
    
    return documents_embeddings, book_metadata

def store_in_qdrant(documents_embeddings, book_metadata=None, collection_name="recursive"):


    """Store document embeddings in Qdrant vector database"""
    # Initialize Qdrant client
    url = "http://127.0.0.1:6333"
    client = QdrantClient(url=url)
    
    # Create collection if it doesn't exist
    try:
        client.get_collection(collection_name)
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
    
    # Create points from document embeddings
    points = []
    for i, (doc, embeddings) in enumerate(documents_embeddings):
        # Extract entities for additional metadata
        entities = extract_entities(doc.page_content)
        
        # Create chunk metadata
        chunk_metadata = {
            "chunk_id": i,
            "prev_chunk_id": i-1 if i > 0 else None,
            "next_chunk_id": i+1 if i < len(documents_embeddings)-1 else None,
            "page_number": doc.metadata.get('page'),
            "chapter": doc.metadata.get('chapter'),
            "title": doc.metadata.get('title', book_metadata.get('title')),
            "author": doc.metadata.get('author', book_metadata.get('author')),
            "series": doc.metadata.get('series', book_metadata.get('series')),
            "entities": entities
        }
        
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings,
                payload={
                    "text": doc.page_content,
                    "metadata": chunk_metadata
                }
            )
        )
        
    
    # Insert points into collection
    operation_info = client.upsert(
        collection_name=collection_name,
        wait=True,
        points=points
    )
    
    return operation_info 


def main():
    """
    Main function to process a document and store its embeddings in Qdrant.
    Can be run directly from the command line.
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process a document and store embeddings')
    parser.add_argument('--file_path', type=str, default='book1.pdf', help='Path to the PDF file to process')
    parser.add_argument('--chunk-size', type=int, default=512, help='Maximum size of each chunk')
    parser.add_argument('--chunk-overlap', type=int, default=50, help='Overlap between chunks')
    parser.add_argument('--collection-name', default='recursive', help='Name of the Qdrant collection')
    
    args = parser.parse_args()
    
    # Process the document
    print(f"Processing document: {args.file_path}")
    documents_embeddings, book_metadata = process_document(
        file_path=args.file_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Store embeddings in Qdrant
    print(f"Storing embeddings in Qdrant collection: {args.collection_name}")
    operation_info = store_in_qdrant(
        documents_embeddings, 
        book_metadata=book_metadata,
        collection_name=args.collection_name
    )
    
    print(f"Storage operation completed: {operation_info}")
    print("Processing complete!")

if __name__ == "__main__":
    main()