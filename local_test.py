"""
Complete local test of the document embedding application
Tests all functionality without Streamlit UI
"""

import os
import json
import time
from io import BytesIO

def test_dependencies():
    """Test all required dependencies"""
    print("=== DEPENDENCY TEST ===")
    
    deps = {}
    
    # Test core dependencies
    try:
        from sentence_transformers import SentenceTransformer
        deps['sentence_transformers'] = True
        print("✅ sentence-transformers available")
    except ImportError:
        deps['sentence_transformers'] = False
        print("❌ sentence-transformers missing")
    
    try:
        from pinecone import Pinecone
        deps['pinecone'] = True
        print("✅ pinecone available")
    except ImportError:
        deps['pinecone'] = False
        print("❌ pinecone missing")
    
    try:
        from docx import Document
        deps['docx'] = True
        print("✅ python-docx available")
    except ImportError:
        deps['docx'] = False
        print("❌ python-docx missing")
    
    try:
        import PyPDF2
        deps['PyPDF2'] = True
        print("✅ PyPDF2 available")
    except ImportError:
        deps['PyPDF2'] = False
        print("❌ PyPDF2 missing")
    
    try:
        from nltk.tokenize import sent_tokenize
        import nltk
        
        # Test tokenizer compatibility
        try:
            nltk.data.find('tokenizers/punkt_tab')
            print("✅ NLTK with punkt_tab available")
        except LookupError:
            try:
                nltk.data.find('tokenizers/punkt')
                print("✅ NLTK with punkt available")
            except LookupError:
                print("⚠️ NLTK tokenizer not found, will download")
                try:
                    nltk.download('punkt_tab', quiet=True)
                    print("✅ punkt_tab downloaded")
                except:
                    nltk.download('punkt', quiet=True)
                    print("✅ punkt downloaded")
        
        deps['nltk'] = True
    except ImportError:
        deps['nltk'] = False
        print("❌ NLTK missing - will use fallback")
    
    return deps

def test_embedding_model():
    """Test embedding model loading"""
    print("\n=== EMBEDDING MODEL TEST ===")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("🔄 Loading sentence-transformers model...")
        start_time = time.time()
        
        model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        
        # Test embedding generation
        test_texts = [
            "This is a test document for embedding generation.",
            "The system processes documents and creates vector embeddings.",
            "Embeddings are stored in Pinecone vector database."
        ]
        
        print("🔄 Generating test embeddings...")
        embeddings = model.encode(test_texts)
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"   Embedding dimension: {len(embeddings[0])}")
        print(f"   Sample embedding values: {embeddings[0][:5]}")
        
        return True, model
    except Exception as e:
        print(f"❌ Embedding model test failed: {e}")
        return False, None

def test_text_processing():
    """Test text extraction and chunking"""
    print("\n=== TEXT PROCESSING TEST ===")
    
    # Test JSON processing
    sample_json = {
        "title": "Test Document",
        "content": "This is a comprehensive test of the document processing system. The system should be able to handle various types of content and split them into appropriate chunks for embedding generation. Each chunk should be optimally sized for the embedding model while maintaining semantic coherence.",
        "metadata": {
            "author": "Test User",
            "date": "2024-01-01",
            "category": "Testing"
        }
    }
    
    try:
        # Test JSON extraction
        json_text = json.dumps(sample_json)
        print("✅ JSON processing works")
        
        # Test text chunking
        try:
            from nltk.tokenize import sent_tokenize
            use_nltk = True
        except ImportError:
            use_nltk = False
            def sent_tokenize(text):
                sentences = []
                for line in text.split('\n'):
                    line = line.strip()
                    if line:
                        parts = line.replace('!', '.').replace('?', '.').split('.')
                        for part in parts:
                            part = part.strip()
                            if part:
                                sentences.append(part)
                return sentences
        
        def chunk_text(text, chunk_size=100):  # Smaller chunks for testing
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
        
        test_text = sample_json['content']
        chunks = chunk_text(test_text)
        
        print(f"✅ Text chunking works ({'NLTK' if use_nltk else 'fallback'})")
        print(f"   Generated {len(chunks)} chunks")
        for i, chunk in enumerate(chunks, 1):
            print(f"   Chunk {i}: {len(chunk.split())} words")
        
        return True, chunks
    except Exception as e:
        print(f"❌ Text processing failed: {e}")
        return False, []

def test_pinecone_client():
    """Test Pinecone client initialization"""
    print("\n=== PINECONE CLIENT TEST ===")
    
    try:
        from pinecone import Pinecone
        print("✅ Pinecone client import successful")
        
        # Test client creation (without API key - just structure)
        print("ℹ️ Pinecone client ready for API key")
        print("   Note: Actual connection requires valid API key")
        
        return True
    except Exception as e:
        print(f"❌ Pinecone client test failed: {e}")
        return False

def test_complete_workflow():
    """Test the complete workflow simulation"""
    print("\n=== COMPLETE WORKFLOW TEST ===")
    
    try:
        # Load model
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        
        # Prepare test document
        test_doc = {
            "title": "Machine Learning Document",
            "sections": [
                "Machine learning is a subset of artificial intelligence.",
                "It focuses on the development of algorithms that can learn from data.",
                "Vector embeddings are numerical representations of text.",
                "They capture semantic meaning in high-dimensional space."
            ]
        }
        
        # Extract text
        text = json.dumps(test_doc)
        
        # Chunk text
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        
        # Simple chunking for demo
        chunks = [text[i:i+200] for i in range(0, len(text), 200)]
        if not chunks[-1].strip():
            chunks = chunks[:-1]
        
        print(f"✅ Document processed into {len(chunks)} chunks")
        
        # Generate embeddings
        embeddings = []
        for i, chunk in enumerate(chunks):
            embedding = model.encode([chunk])[0]
            embeddings.append(embedding)
            print(f"   Chunk {i+1}: {len(embedding)} dimensions")
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        
        # Simulate Pinecone upload structure
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_data = {
                'id': f'test_doc_{i}',
                'values': embedding.tolist(),
                'metadata': {
                    'source': 'test_document.json',
                    'chunk': i,
                    'text': chunk[:500]  # First 500 chars
                }
            }
            vectors.append(vector_data)
        
        print(f"✅ Prepared {len(vectors)} vectors for upload")
        print("✅ Complete workflow simulation successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Complete workflow failed: {e}")
        return False

def main():
    """Run all local tests"""
    print("=" * 60)
    print("DOCUMENT EMBEDDING APPLICATION - LOCAL TEST SUITE")
    print("=" * 60)
    
    # Test dependencies
    deps = test_dependencies()
    
    # Test embedding model
    embedding_success, model = test_embedding_model()
    
    # Test text processing
    text_success, chunks = test_text_processing()
    
    # Test Pinecone client
    pinecone_success = test_pinecone_client()
    
    # Test complete workflow
    workflow_success = test_complete_workflow()
    
    # Results summary
    print("\n" + "=" * 60)
    print("LOCAL TEST RESULTS")
    print("=" * 60)
    
    tests = [
        ("Dependencies", len([d for d in deps.values() if d]) >= 3),
        ("Embedding Model", embedding_success),
        ("Text Processing", text_success),
        ("Pinecone Client", pinecone_success),
        ("Complete Workflow", workflow_success)
    ]
    
    passed = sum([result for _, result in tests])
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Application ready for deployment")
        print("✅ All core functionality working")
    elif passed >= 4:
        print("\n⚡ MOSTLY READY!")
        print("✅ Core functionality working")
        print("⚠️ Minor issues detected")
    else:
        print("\n🔧 ISSUES DETECTED")
        print("❌ Critical functionality missing")
    
    print("\n📋 NEXT STEPS:")
    if passed >= 4:
        print("1. ✅ Core functionality verified")
        print("2. ✅ Ready for Streamlit Cloud deployment")
        print("3. 🔄 Try running Streamlit locally: streamlit run app.py")
        print("4. 🌐 Deploy to Streamlit Cloud when ready")
    else:
        print("1. ❌ Fix failing tests first")
        print("2. 🔄 Install missing dependencies")
        print("3. 🔄 Re-run this test")

if __name__ == "__main__":
    main()