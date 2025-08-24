# Document Embedding Streamlit App

A Streamlit application that allows users to upload documents (PDF, DOCX, or JSON), split them into chunks, generate embeddings using sentence-transformers, and store them in Pinecone VectorDB with a progress bar.

## Features

- 📄 Support for PDF, DOCX, and JSON file formats
- 🔗 Integration with Pinecone VectorDB
- 🤖 Uses BAAI/bge-base-en-v1.5 for embeddings
- 📊 Real-time progress tracking
- 🎯 Configurable document chunking
- 💾 Reusable API input fields

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Download NLTK data (will be done automatically on first run):
```python
import nltk
nltk.download('punkt')
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

3. Fill in the required information:
   - **Pinecone API Key**: Your Pinecone API key
   - **Pinecone Environment**: Your Pinecone environment (e.g., 'us-west1-gcp')
   - **Index Name**: Name for your Pinecone index (default: 'document-embeddings')
   - **Document ID**: A unique identifier for your document

4. Upload your document and click "Upload and Embed"

## Configuration

- **Chunk Size**: 500 words (configurable in code)
- **Embedding Model**: BAAI/bge-base-en-v1.5
- **Embedding Dimension**: 768
- **Supported File Types**: PDF, DOCX, JSON

## File Structure

```
PineconEmbedder/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## How It Works

1. **Text Extraction**: The app extracts text from uploaded documents
2. **Text Chunking**: Documents are split into manageable chunks using NLTK sentence tokenization
3. **Embedding Generation**: Each chunk is converted to a 768-dimensional vector using BAAI/bge-base-en-v1.5
4. **Vector Storage**: Embeddings are stored in Pinecone with metadata including source file and chunk index
5. **Progress Tracking**: Real-time progress bar shows upload status

## Requirements

- Python 3.7+
- Pinecone account and API key
- Internet connection for downloading models and connecting to Pinecone

## Troubleshooting

- **NLTK Error**: The app will automatically download required NLTK data on first run
- **Pinecone Connection**: Ensure your API key and environment are correct
- **File Upload**: Check that your file is in a supported format (PDF, DOCX, JSON)
- **Memory Issues**: For large documents, consider reducing chunk size in the code

## Notes

- Each chunk is stored with a unique ID in the format `{doc_id}_{chunk_number}`
- The first 1000 characters of each chunk are stored as metadata for reference
- If the specified index doesn't exist, it will be created automatically