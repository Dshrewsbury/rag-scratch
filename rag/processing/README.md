# PDF Processing and Embedding Generation

recursive_chunker.py processes PDF books and generates embeddings using a recursive chunking strategy, storing the results in a Qdrant vector database.

This will be modified soon to accept any number of pdfs and potentially other file types. At the moment, we are using llama-2-7b-chat.Q4_K_M.gguf as the embedding model which can be found and downloaded: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF, and should be placed in the models directory. This will be updated soon to allow you to select an embedding model.

## Prerequisites

1. Ensure you have the required models in your `assets/models` directory:
   - LLaMA embedding model (default path: `assets/models/llama-2-7b-chat.Q4_K_M.gguf`)

2. Make sure Qdrant is running (usually handled by Docker Compose)

## Usage

1. Place your PDF book in the `assets/documents` directory

2. Run the script using Python:
   ```bash
   python recursive_chunker.py --file_path your_book.pdf
   ```

### Optional Arguments

- `--chunk-size`: Size of text chunks (default: 512)
- `--chunk-overlap`: Overlap between chunks (default: 50)
- `--collection-name`: Name of Qdrant collection (default: 'recursive')

Example with custom settings:
```bash
python recursive_chunker.py \
  --file_path book.pdf \
  --chunk-size 1000 \
  --chunk-overlap 100 \
  --collection-name my_book_collection
```

## Output

The script will:
1. Convert the PDF to markdown format
2. Extract book metadata (title, author, series)
3. Split content into chunks
4. Generate embeddings for each chunk
5. Store embeddings and metadata in Qdrant

Progress and results will be logged to the console. 