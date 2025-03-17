import time
import pandas as pd
import uuid
from tqdm import tqdm
import re
from litellm import completion

# Import your existing components
# Assuming these are in the correct path - adjust imports as needed
from rag.processing.chunking.recursive_chunker import parse_markdown_with_metadata


# ====================================
# Core LLM Generation
# ====================================

def generate_text(prompt, model="gpt-4o"):
    """
    Simple, dedicated function for text generation using litellm
    
    Args:
        prompt: The prompt to send to the LLM
        model: The model to use
        
    Returns:
        Generated text response
    """
    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        api_base="https://models.inference.ai.azure.com",
        stream=False
    )
    return response["choices"][0]["message"]["content"]

# ====================================
# 1. Context Loading and Chunking
# ====================================

def load_and_chunk_documents(file_paths, chunk_size=1500, chunk_overlap=100):
    """
    Load documents and split them into chunks using your recursive_chunker
    
    Args:
        file_paths: List of paths to documents
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of document chunks
    """
    all_chunks = []
    
    for file_path in file_paths:
        # For PDF files, use your existing functionality
        if file_path.endswith('.pdf'):
            # Extract text using pymupdf4llm (imported from recursive_chunker)
            import pymupdf4llm
            md_text = pymupdf4llm.to_markdown(file_path)
            
            # Parse documents with metadata
            documents, book_metadata = parse_markdown_with_metadata(md_text)
            
            # Split into chunks (simplifying the chunking logic from your recursive_chunker)
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            
            for doc in documents:
                chunks = text_splitter.split_text(doc.page_content)
                for i, chunk_text in enumerate(chunks):
                    all_chunks.append({
                        "id": str(uuid.uuid4()),
                        "content": chunk_text,
                        "metadata": {
                            "source": file_path,
                            "page": doc.metadata.get("page", 1),
                            "chunk_number": i
                        }
                    })
    
    print(f"Created {len(all_chunks)} chunks from {len(file_paths)} documents")
    return all_chunks

# ====================================
# 2. Question Generation Functions
# ====================================

def generate_question_from_chunk(chunk):
    """
    Generate a question based on the given context chunk
    
    Args:
        chunk: A document chunk containing context
        
    Returns:
        Generated question
    """
    prompt = """
    <Instructions>
    Here is some context:
    <context>
    {context}
    </context>

    Your task is to generate 1 question that can be answered using the provided context, following these rules:

    <rules>
    1. The question should make sense to humans even when read without the given context.
    2. The question should be fully answered from the given context.
    3. The question should be framed from a part of context that contains important information. It can also be from tables, code, etc.
    4. The answer to the question should not contain any links.
    5. The question should be of moderate difficulty.
    6. The question must be reasonable and must be understood and responded by humans.
    7. Do not use phrases like 'provided context', etc. in the question.
    8. Avoid framing questions using the word "and" that can be decomposed into more than one question.
    9. The question should not contain more than 10 words, make use of abbreviations wherever possible.
    </rules>

    To generate the question, first identify the most important or relevant part of the context. Then frame a question around that part that satisfies all the rules above.

    Output only the generated question with a "?" at the end, no other text or characters.
    </Instructions>
    """.format(context=chunk["content"] if isinstance(chunk, dict) else chunk)
    
    return generate_text(prompt).strip()

def generate_answer(question, chunk):
    """
    Generate an answer to the question based on the given context
    
    Args:
        question: The question to answer
        chunk: The document chunk containing context
        
    Returns:
        Generated answer
    """
    context = chunk["content"] if isinstance(chunk, dict) else chunk
    
    prompt = """
    <Instructions>
    <Task>
    <role>You are an experienced QA Engineer for building large language model applications.</role>
    <task>It is your task to generate an answer to the following question <question>{question}</question> only based on the <context>{context}</context></task>
    The output should be only the answer generated from the context.

    <rules>
    1. Only use the given context as a source for generating the answer.
    2. Be as precise as possible with answering the question.
    3. Be concise in answering the question and only answer the question at hand rather than adding extra information.
    </rules>

    Only output the generated answer as a sentence. No extra characters.
    </Task>
    </Instructions>
    
    Assistant:
    """.format(question=question, context=context)
    
    return generate_text(prompt).strip()

def extract_relevant_source(question, chunk):
    """
    Extract the relevant sentences from the context that answer the question
    
    Args:
        question: The question to extract sources for
        chunk: The document chunk containing context
        
    Returns:
        Extracted source sentences
    """
    context = chunk["content"] if isinstance(chunk, dict) else chunk
    
    prompt = """
    <Instructions>
    Here is the context:
    <context>
    {context}
    </context>

    Your task is to extract the relevant sentences from the given context that can potentially help answer the following question. You are not allowed to make any changes to the sentences from the context.

    <question>
    {question}
    </question>

    Output only the relevant sentences you found, one sentence per line, without any extra characters or explanations.
    </Instructions>
    """.format(question=question, context=context)
    
    return generate_text(prompt).strip()

def compress_question(question):
    """
    Create a more compressed/evolved version of the question
    
    Args:
        question: The original question
        
    Returns:
        Compressed question
    """
    prompt = """
    <Instructions>
    <role>You are an experienced linguistics expert for building testsets for large language model applications.</role>

    <task>It is your task to rewrite the following question in a more indirect and compressed form, following these rules:

    <rules>
    1. Make the question more indirect
    2. Make the question shorter
    3. Use abbreviations if possible
    </rules>

    <question>
    {question}
    </question>

    Your output should only be the rewritten question with a question mark "?" at the end. Do not provide any other explanation or text.
    </task>
    </Instructions>
    """.format(question=question)
    
    return generate_text(prompt).strip()

# ====================================
# 3. Retrieval-Focused Dataset Generation
# ====================================

def generate_retrieval_dataset_from_qdrant(qdrant_client, collection_name, limit=100):
    """
    Generate a dataset specifically for testing retrieval by creating questions
    from existing Qdrant chunks
    
    Args:
        qdrant_client: Initialized Qdrant client
        collection_name: Name of the collection to query
        limit: Maximum number of chunks to process
        
    Returns:
        DataFrame with questions and source chunks for retrieval testing
    """
    # Fetch chunks from Qdrant
    search_result = qdrant_client.scroll(
        collection_name=collection_name,
        limit=limit,
        with_payload=True,
        with_vectors=False
    )
    
    chunks = search_result[0]  # First element contains the points
    
    print(f"Retrieved {len(chunks)} chunks from Qdrant collection '{collection_name}'")
    
    # Initialize dataset
    retrieval_dataset = pd.DataFrame(columns=[
        "question",
        "question_compressed",
        "source_chunk_id",
        "source_content",
        "source_metadata"
    ])
    
    # Generate questions for each chunk
    for i, chunk in tqdm(enumerate(chunks), total=len(chunks)):
        chunk_id = chunk.id
        chunk_content = chunk.payload.get("text")
        chunk_metadata = chunk.payload.get("metadata", {})
        
        # Skip if no content
        if not chunk_content:
            continue
        
        # Generate question
        question = generate_question_from_chunk({"content": chunk_content})
        
        # Generate compressed version of the question
        compressed_question = compress_question(question)
        
        # Add to dataset
        retrieval_dataset.at[i, "question"] = question
        retrieval_dataset.at[i, "question_compressed"] = compressed_question
        retrieval_dataset.at[i, "source_chunk_id"] = chunk_id
        retrieval_dataset.at[i, "source_content"] = chunk_content
        retrieval_dataset.at[i, "source_metadata"] = str(chunk_metadata)  # Convert to string for DataFrame storage
    
    print(f"Generated {len(retrieval_dataset)} questions for retrieval testing")
    
    return retrieval_dataset

# ====================================
# 4. Full Synthetic Dataset Generation (End-to-End Testing)
# ====================================

def generate_qa_dataset_for_chunk(chunk, dataset, chunk_number):
    """
    Generate a question-answer pair for a single chunk and add it to the dataset
    
    Args:
        chunk: The document chunk to process
        dataset: The pandas DataFrame to add the data to
        chunk_number: The index of the chunk
        
    Returns:
        Updated dataset
    """
    # Generate initial question
    question = generate_question_from_chunk(chunk)
    dataset.at[chunk_number, "question"] = question
    
    # Generate compressed question
    compressed_question = compress_question(question)
    dataset.at[chunk_number, "question_compressed"] = compressed_question
    
    # Generate reference answer
    answer = generate_answer(question, chunk)
    dataset.at[chunk_number, "reference_answer"] = answer
    
    # Extract source sentences
    source_sentence = extract_relevant_source(question, chunk)
    dataset.at[chunk_number, "source_sentence"] = source_sentence
    
    # Add source information
    dataset.at[chunk_number, "source_raw"] = chunk["content"]
    dataset.at[chunk_number, "source_document"] = chunk["metadata"]["source"]
    dataset.at[chunk_number, "source_chunk_id"] = chunk["id"]
    
    return dataset

def generate_full_dataset(chunks, subset_size=None):
    """
    Generate a complete dataset of question-answer pairs for all chunks
    
    Args:
        chunks: List of document chunks
        subset_size: Optional number of chunks to process (for testing)
        
    Returns:
        DataFrame containing the generated dataset
    """
    # Initialize dataset
    dataset = pd.DataFrame(columns=[
        "question", 
        "question_compressed", 
        "reference_answer", 
        "source_sentence",
        "source_raw",
        "source_document",
        "source_chunk_id"
    ])
    
    # Use subset if specified
    if subset_size and subset_size < len(chunks):
        chunks_to_process = chunks[:subset_size]
    else:
        chunks_to_process = chunks
    
    print(f"Generating dataset from {len(chunks_to_process)} chunks")
    generation_time_start = time.time()
    
    for i, chunk in tqdm(enumerate(chunks_to_process), total=len(chunks_to_process)):
        q_generation_time_start = time.time()
        
        dataset = generate_qa_dataset_for_chunk(chunk, dataset, i)
        
        q_generation_time_end = time.time()
        total_elapsed_time_generation = q_generation_time_end - q_generation_time_start
        
        print(f"Finished creating evaluation data for chunk {i+1}")
        print(f"Generation time for chunk: {total_elapsed_time_generation:.2f}s")
        print("---")
    
    generation_time_end = time.time()
    total_elapsed_time = generation_time_end - generation_time_start
    print(f"Generation time for all chunks: {total_elapsed_time:.2f}s")
    
    return dataset

# ====================================
# 5. Quality Assessment
# ====================================

def evaluate_groundedness(question, context):
    """
    Evaluate how well the question can be answered using the given context
    
    Args:
        question: The question to evaluate
        context: The context to evaluate against
        
    Returns:
        Evaluation result with rating and reasoning
    """
    prompt = """
    <Instructions>
    You will be given a context and a question related to that context.

    Your task is to provide an evaluation of how well the given question can be answered using only the information provided in the context. Rate this on a scale from 1 to 5, where:

    1 = The question cannot be answered at all based on the given context
    2 = The context provides very little relevant information to answer the question
    3 = The context provides some relevant information to partially answer the question 
    4 = The context provides substantial information to answer most aspects of the question
    5 = The context provides all the information needed to fully and unambiguously answer the question

    First, read through the provided context carefully:

    <context>
    {context}
    </context>

    Then read the question:

    <question>
    {question}
    </question>

    Evaluate how well you think the question can be answered using only the context information. Provide your reasoning first in an <evaluation> section, explaining what relevant or missing information from the context led you to your evaluation score in only one sentence.

    Provide your evaluation in the following format:

    <rating>(Your rating from 1 to 5)</rating>
    
    <evaluation>(Your evaluation and reasoning for the rating)</evaluation>
    </Instructions>
    """.format(question=question, context=context)
    
    return generate_text(prompt).strip()

def evaluate_relevance(question):
    """
    Evaluate how relevant the question is for the intended use case
    
    Args:
        question: The question to evaluate
        
    Returns:
        Evaluation result with rating and reasoning
    """
    prompt = """
    <Instructions>
    You will be given a question related to a document. Your task is to evaluate how useful this question would be for a person trying to understand the document.

    To evaluate the usefulness of the question, consider the following criteria:

    1. Relevance: Is the question directly relevant to understanding the document? Questions that are too broad or unrelated should receive a lower rating.

    2. Practicality: Does the question address a practical problem or use case that readers might encounter? Theoretical or overly academic questions may be less useful.

    3. Clarity: Is the question clear and well-defined? Ambiguous or vague questions are less useful.

    4. Depth: Does the question require a substantive answer that demonstrates understanding of the topic? Surface-level questions may be less useful.

    5. Applicability: Would answering this question provide insights or knowledge that could be applied to understanding the document? Questions with limited applicability should receive a lower rating.

    Provide your evaluation in the following format:

    <rating>(Your rating from 1 to 5)</rating>
    
    <evaluation>(Your evaluation and reasoning for the rating)</evaluation>

    Here is the question:

    {question}
    </Instructions>
    """.format(question=question)
    
    return generate_text(prompt).strip()

def extract_rating(text):
    """Extract rating value from evaluation text"""
    pattern = r'<rating>(.*?)</rating>'
    match = re.search(pattern, text)
    if match:
        rating = match.group(1)
        return rating
    else:
        return None

def extract_reasoning(text):
    """Extract reasoning from evaluation text"""
    pattern = r'<evaluation>(.*?)</evaluation>'
    match = re.search(pattern, text)
    if match:
        rating = match.group(1)
        return rating
    else:
        return None

def evaluate_dataset(dataset):
    """
    Evaluate the quality of questions in the dataset
    
    Args:
        dataset: DataFrame containing the generated dataset
        
    Returns:
        Dataset with evaluation scores added
    """
    for index, row in dataset.iterrows():
        question = row['question']
        
        # For full dataset, evaluate both groundedness and relevance
        if 'source_raw' in row:
            source_raw = row['source_raw']
            groundedness_check = evaluate_groundedness(question, source_raw)
            groundedness_score = extract_rating(groundedness_check)
            groundedness_reason = extract_reasoning(groundedness_check)
            
            dataset.at[index, 'groundedness_score'] = groundedness_score
            dataset.at[index, 'groundedness_score_reasoning'] = groundedness_reason
        
        # For both types of datasets, evaluate relevance
        relevance_check = evaluate_relevance(question)
        relevance_score = extract_rating(relevance_check)
        relevance_reason = extract_reasoning(relevance_check)
        
        dataset.at[index, 'relevancy_score'] = relevance_score
        dataset.at[index, 'relevancy_score_reasoning'] = relevance_reason
    
    return dataset

# ====================================
# 6. Retrieval Evaluation Metrics
# ====================================

def evaluate_retrieval(dataset, retriever_fn, k_values=[1, 3, 5, 10]):
    """
    Evaluate retrieval metrics using the dataset
    
    Args:
        dataset: DataFrame with questions and source chunk IDs
        retriever_fn: Function that takes a question and returns list of retrieved chunk IDs
        k_values: List of k values to evaluate for precision/recall
        
    Returns:
        Dictionary of evaluation metrics
    """
    results = []
    
    for index, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Evaluating retrieval"):
        query = row['question']
        source_chunk_id = row['source_chunk_id']
        
        # Run retrieval
        retrieved_ids = retriever_fn(query)
        
        # Calculate metrics per query
        query_results = {
            'query_id': index,
            'query': query,
            'source_chunk_id': source_chunk_id,
            'retrieved_ids': retrieved_ids[:max(k_values)]  # Keep only top-k for max k
        }
        
        # Calculate performance at different k values
        for k in k_values:
            top_k_ids = retrieved_ids[:k]
            
            # Hit@k - whether the source chunk was retrieved in the top k
            hit_at_k = source_chunk_id in top_k_ids
            query_results[f'hit@{k}'] = hit_at_k
            
            # Rank - position of source chunk in results (1-indexed)
            try:
                rank = retrieved_ids.index(source_chunk_id) + 1
            except ValueError:
                rank = 0  # Not found
            query_results['rank'] = rank
            
            # Reciprocal Rank - 1/rank (0 if not found)
            reciprocal_rank = 1/rank if rank > 0 else 0
            query_results['reciprocal_rank'] = reciprocal_rank
            
            # For precision/recall, we need to know all relevant chunks
            # In this simple case, we consider only the source chunk as relevant
            relevant_chunks = [source_chunk_id]
            
            # Precision@k - proportion of retrieved chunks that are relevant
            precision_at_k = sum(1 for cid in top_k_ids if cid in relevant_chunks) / len(top_k_ids) if top_k_ids else 0
            query_results[f'precision@{k}'] = precision_at_k
            
            # Recall@k - proportion of relevant chunks that are retrieved
            recall_at_k = sum(1 for cid in relevant_chunks if cid in top_k_ids) / len(relevant_chunks) if relevant_chunks else 0
            query_results[f'recall@{k}'] = recall_at_k
        
        results.append(query_results)
    
    # Aggregate metrics
    metrics = {}
    
    # Overall metrics
    metrics['mrr'] = sum(r['reciprocal_rank'] for r in results) / len(results)
    metrics['mean_rank'] = sum(r['rank'] for r in results if r['rank'] > 0) / sum(1 for r in results if r['rank'] > 0) if any(r['rank'] > 0 for r in results) else float('inf')
    
    # Metrics at different k values
    for k in k_values:
        metrics[f'hit_rate@{k}'] = sum(r[f'hit@{k}'] for r in results) / len(results)
        metrics[f'precision@{k}'] = sum(r[f'precision@{k}'] for r in results) / len(results)
        metrics[f'recall@{k}'] = sum(r[f'recall@{k}'] for r in results) / len(results)
    
    return metrics, results

# ====================================
# Main Functions
# ====================================

def generate_retrieval_test_dataset(qdrant_client, collection_name, output_path=None, limit=100, evaluate=True):
    """
    Generate a dataset specifically for testing retrieval performance
    
    Args:
        qdrant_client: Initialized Qdrant client
        collection_name: Name of the collection to query
        output_path: Optional path to save the dataset
        limit: Maximum number of chunks to process
        evaluate: Whether to evaluate question quality
        
    Returns:
        DataFrame containing the generated dataset
    """
    # Generate dataset from Qdrant chunks
    dataset = generate_retrieval_dataset_from_qdrant(
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        limit=limit
    )
    
    # Evaluate dataset if requested
    if evaluate:
        dataset = evaluate_dataset(dataset)
    
    # Save dataset if output path provided
    if output_path:
        dataset.to_csv(output_path, index=False)
        print(f"Retrieval test dataset saved to {output_path}")
    
    return dataset

def generate_full_qa_dataset(file_paths, output_path=None, subset_size=None, evaluate=True):
    """
    Generate a complete synthetic Q&A dataset for end-to-end testing
    
    Args:
        file_paths: List of paths to documents to process
        output_path: Optional path to save the dataset
        subset_size: Optional number of chunks to process
        evaluate: Whether to evaluate the dataset quality
        
    Returns:
        DataFrame containing the generated dataset
    """
    # Load and chunk documents
    chunks = load_and_chunk_documents(file_paths)
    
    # Generate full dataset
    dataset = generate_full_dataset(chunks, subset_size)
    
    # Evaluate dataset if requested
    if evaluate:
        dataset = evaluate_dataset(dataset)
    
    # Save dataset if output path provided
    if output_path:
        dataset.to_csv(output_path, index=False)
        print(f"Full Q&A dataset saved to {output_path}")
    
    return dataset

def run_retrieval_evaluation(dataset, retriever_fn, output_path=None):
    """
    Run retrieval evaluation on a dataset and save results
    
    Args:
        dataset: DataFrame with questions and source chunk IDs
        retriever_fn: Function that takes a question and returns list of retrieved chunk IDs
        output_path: Optional path to save the results
        
    Returns:
        Tuple of (metrics dict, detailed results list)
    """
    # Run evaluation
    metrics, detailed_results = evaluate_retrieval(dataset, retriever_fn)
    
    # Print summary metrics
    print("\nRetrieval Evaluation Results:")
    print("-----------------------------")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save results if output path provided
    if output_path:
        # Save metrics
        pd.DataFrame([metrics]).to_csv(f"{output_path}_metrics.csv", index=False)
        
        # Save detailed results
        results_df = pd.DataFrame(detailed_results)
        results_df.to_csv(f"{output_path}_detailed.csv", index=False)
        
        print(f"Evaluation results saved to {output_path}_metrics.csv and {output_path}_detailed.csv")
    
    return metrics, detailed_results

# Example usage
if __name__ == "__main__":
    # Example: Generate retrieval test dataset from Qdrant
    from retrieval.vector_store import VectorStore  # Import your vector store
    
    vector_store = VectorStore()  # Initialize your vector store
    
    retrieval_dataset = generate_retrieval_test_dataset(
        qdrant_client=vector_store.client,  # Your Qdrant client
        collection_name="your_collection",  # Your collection name
        output_path="retrieval_test_dataset.csv",
        limit=100  # Process 100 chunks
    )
    
    # Example: Generate full Q&A dataset
    file_paths = ["path/to/your/document1.pdf", "path/to/your/document2.pdf"]
    
    full_dataset = generate_full_qa_dataset(
        file_paths=file_paths,
        output_path="full_qa_dataset.csv",
        subset_size=50,  # Process only 50 chunks
        evaluate=True
    )
    
    # Example: Evaluate retrieval
    def example_retriever_fn(query):
        # This is where you would call your actual retrieval system
        # For this example, just return some dummy IDs
        return ["id1", "id2", "id3", "id4", "id5"]
    
    metrics, details = run_retrieval_evaluation(
        dataset=retrieval_dataset,
        retriever_fn=example_retriever_fn,
        output_path="retrieval_evaluation"
    )