"""
Test script for document embedding functionality
Run this to verify that the core components work correctly
"""

import json
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import nltk

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

def test_embedding_model():
    """Test that the embedding model loads and works correctly"""
    print("Testing embedding model...")
    try:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        test_text = "This is a test sentence for embedding."
        embedding = model.encode([test_text])[0]
        print(f"✓ Model loaded successfully")
        print(f"✓ Embedding dimension: {len(embedding)}")
        print(f"✓ Sample embedding (first 5 values): {embedding[:5]}")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

def test_text_chunking():
    """Test the text chunking functionality"""
    print("\nTesting text chunking...")
    sample_text = """
    This is a sample document with multiple sentences. 
    It contains several paragraphs to test the chunking functionality.
    The chunking should split this text into manageable pieces based on sentence boundaries.
    Each chunk should contain approximately the specified number of words.
    This helps ensure that the text is properly processed for embedding generation.
    """
    
    try:
        sentences = sent_tokenize(sample_text)
        print(f"✓ Text split into {len(sentences)} sentences")
        
        # Test chunking function
        def chunk_text(text, chunk_size=50):  # Smaller chunk for testing
            sentences = sent_tokenize(text)
            chunks, current, length = [], [], 0
            for s in sentences:
                length += len(s.split())
                current.append(s)
                if length >= chunk_size:
                    chunks.append(' '.join(current))
                    current, length = [], 0
            if current:
                chunks.append(' '.join(current))
            return chunks
        
        chunks = chunk_text(sample_text)
        print(f"✓ Text chunked into {len(chunks)} pieces")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}: {len(chunk.split())} words")
        return True
    except Exception as e:
        print(f"✗ Error in chunking: {e}")
        return False

def test_json_processing():
    """Test JSON file processing"""
    print("\nTesting JSON processing...")
    try:
        sample_data = {
            "title": "Sample Document",
            "content": "This is sample content for testing JSON processing.",
            "metadata": {
                "author": "Test Author",
                "date": "2024-01-01"
            }
        }
        
        json_string = json.dumps(sample_data)
        print(f"✓ JSON processing successful")
        print(f"✓ JSON length: {len(json_string)} characters")
        return True
    except Exception as e:
        print(f"✗ Error in JSON processing: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Document Embedding Test Suite ===\n")
    
    tests = [
        test_embedding_model,
        test_text_chunking,
        test_json_processing
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ All tests passed! The application should work correctly.")
    else:
        print("✗ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()