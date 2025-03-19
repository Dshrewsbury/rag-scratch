import uuid
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_cpp import Llama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

from config.settings import EMBEDDING_MODEL_PATH, DATA_DIR, EMBEDDINGS_DIR, BATCH_SIZE


def calculate_cosine_distances(chunk_embeddings):
    """Calculate cosine distances between adjacent chunk embeddings"""
    distances = []
    for i in range(len(chunk_embeddings) - 1):
        embedding_current = chunk_embeddings[i]["embedding"]
        embedding_next = chunk_embeddings[i + 1]["embedding"]
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
        distance = 1 - similarity
        distances.append(distance)
    return distances


def process_document(
    file_path=None, chunk_size=1500, chunk_overlap=100, batch_size=BATCH_SIZE
):
    """
    Process a document using semantic chunking strategy

    Args:
        file_path: Path to the PDF file
        chunk_size: Initial chunk size for initial splitting
        chunk_overlap: Overlap between initial chunks
        batch_size: Batch size for processing

    Returns:
        Semantic chunks and their embeddings
    """
    # Load PDF
    file = file_path or f"{DATA_DIR}/documents/llama2.pdf"
    loader = PyPDFLoader(file)
    data = loader.load()

    # Load embedding model
    llm = Llama(
        model_path=EMBEDDING_MODEL_PATH,
        embedding=True,
        verbose=False,
        n_batch=batch_size,
    )

    # Split PDF content into initial chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    documents = text_splitter.split_documents(data)
    print(f"{len(documents)} initial chunks were created")

    # Store chunks as text and generate embeddings
    chunks = [doc.page_content for doc in documents]
    chunk_embeddings = llm.create_embedding(chunks)["data"]

    # Calculate distances between adjacent chunks
    distances = calculate_cosine_distances(chunk_embeddings)

    # Set threshold to identify semantic breaks
    breakpoint_percentile_threshold = 95
    breakpoint_distance_threshold = np.percentile(
        distances, breakpoint_percentile_threshold
    )
    indices_above_thresh = [
        i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
    ]

    # Group chunks into semantic chunks
    start_index = 0
    semantic_chunks = []
    for index in indices_above_thresh:
        end_index = index
        group = chunks[start_index : end_index + 1]
        combined_text = " ".join(group)
        semantic_chunks.append(combined_text)
        start_index = index + 1

    # Append any remaining text
    if start_index < len(chunks):
        combined_text = " ".join(chunks[start_index:])
        semantic_chunks.append(combined_text)

    # Generate embeddings for semantic chunks
    semantic_chunk_embeddings = llm.create_embedding(semantic_chunks)["data"]

    return semantic_chunks, semantic_chunk_embeddings


def store_in_qdrant(
    semantic_chunks, semantic_chunk_embeddings, collection_name="semantic"
):
    """Store semantic chunks in Qdrant vector database"""
    # Initialize Qdrant client
    client = QdrantClient(path=f"{EMBEDDINGS_DIR}/{collection_name}")

    # Create collection
    try:
        client.get_collection(collection_name)
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=len(semantic_chunk_embeddings[0]["embedding"]),
                distance=Distance.COSINE,
            ),
        )

    # Store chunks and embeddings
    points = [
        PointStruct(
            id=str(uuid.uuid4()), vector=embedding["embedding"], payload={"text": chunk}
        )
        for chunk, embedding in zip(semantic_chunks, semantic_chunk_embeddings)
    ]

    # Insert points into collection
    operation_info = client.upsert(
        collection_name=collection_name, wait=True, points=points
    )

    print(f"Stored {len(points)} semantic chunks in the vector database.")
    return operation_info
