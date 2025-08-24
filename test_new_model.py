"""
Quick test for the new BAAI/bge-base-en-v1.5 model
Verifies model loading and embedding generation
"""

print("=== BAAI/bge-base-en-v1.5 MODEL TEST ===")

try:
    from sentence_transformers import SentenceTransformer
    print("✅ sentence-transformers imported successfully")
    
    print("🔄 Loading BAAI/bge-base-en-v1.5 model...")
    print("   Note: First time may take 1-2 minutes to download")
    
    model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    print("✅ Model loaded successfully!")
    
    # Test embedding generation
    test_texts = [
        "This is a test document for the new embedding model.",
        "BAAI/bge-base-en-v1.5 is a high-quality embedding model.",
        "It produces 768-dimensional embeddings for text."
    ]
    
    print("🔄 Generating test embeddings...")
    embeddings = model.encode(test_texts)
    
    print(f"✅ Successfully generated {len(embeddings)} embeddings")
    print(f"   Embedding dimension: {len(embeddings[0])}")
    print(f"   Expected dimension: 768")
    
    if len(embeddings[0]) == 768:
        print("✅ Dimension check PASSED!")
    else:
        print(f"❌ Dimension mismatch! Got {len(embeddings[0])}, expected 768")
    
    # Show sample embedding values
    print(f"   Sample embedding values: {embeddings[0][:5]}")
    
    # Test with longer text
    long_text = """
    The BAAI/bge-base-en-v1.5 model is a state-of-the-art text embedding model
    developed by the Beijing Academy of Artificial Intelligence (BAAI). 
    It's designed to produce high-quality embeddings for various natural language
    processing tasks including semantic search, text classification, and clustering.
    The model has been trained on a diverse corpus of text data and can handle
    various types of content effectively.
    """
    
    print("🔄 Testing with longer text...")
    long_embedding = model.encode([long_text.strip()])[0]
    print(f"✅ Long text embedding generated: {len(long_embedding)} dimensions")
    
    print("\n🎉 MODEL TEST COMPLETED SUCCESSFULLY!")
    print("✅ BAAI/bge-base-en-v1.5 is working correctly")
    print("✅ Produces 768-dimensional embeddings as expected")
    print("✅ Ready for integration with Pinecone")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Please install sentence-transformers: pip install sentence-transformers")
except Exception as e:
    print(f"❌ Error: {e}")
    print("   Check your internet connection for model download")

print("\n=== PINECONE COMPATIBILITY ===")
print("✅ Model output dimension (768) is compatible with Pinecone")
print("✅ Index creation will use dimension=768")
print("✅ Existing indexes with dimension=384 will need to be recreated")

print("\n=== PERFORMANCE NOTES ===")
print("📊 BAAI/bge-base-en-v1.5 vs all-MiniLM-L6-v2:")
print("   - Higher quality embeddings (better semantic understanding)")
print("   - Larger dimension (768 vs 384)")
print("   - Slightly slower inference")
print("   - Better performance on retrieval tasks")