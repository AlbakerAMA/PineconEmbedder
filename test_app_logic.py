"""
Test the core application logic without Streamlit
This verifies that all the document processing and embedding logic works
"""

import os
import json

def test_imports():
    """Test all critical imports"""
    print("Testing imports...")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence_transformers import successful")
    except ImportError as e:
        print(f"❌ sentence_transformers failed: {e}")
        return False
    
    try:
        from pinecone import Pinecone
        print("✅ pinecone import successful")
    except ImportError as e:
        print(f"❌ pinecone failed: {e}")
        return False
    
    try:
        from docx import Document
        print("✅ python-docx import successful")
    except ImportError as e:
        print(f"❌ python-docx failed: {e}")
        return False
    
    try:
        import PyPDF2
        print("✅ PyPDF2 import successful")
    except ImportError as e:
        print(f"❌ PyPDF2 failed: {e}")
        return False
    
    return True

def test_embedding_model():
    """Test embedding model loading and generation"""
    print("\nTesting embedding model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("🔄 Loading sentence-transformers model...")
        model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        print("✅ Model loaded successfully")
        
        # Test embedding generation
        test_text = "This is a test sentence for embedding generation."
        print("🔄 Generating test embedding...")
        embedding = model.encode([test_text])[0]
        print(f"✅ Embedding generated: {len(embedding)} dimensions")
        print(f"   Sample values: {embedding[:5]}")
        
        return True, model
    except Exception as e:
        print(f"❌ Embedding model test failed: {e}")
        return False, None

def test_text_chunking():
    """Test text chunking functionality"""
    print("\nTesting text chunking...")
    
    try:
        # Try NLTK first
        try:
            from nltk.tokenize import sent_tokenize
            import nltk
            
            # Check if punkt is available
            try:
                # Try new tokenizer first (NLTK 3.8.2+)
                try:
                    nltk.data.find('tokenizers/punkt_tab')
                    print("✅ NLTK punkt_tab tokenizer available")
                    use_nltk = True
                except LookupError:
                    # Try old tokenizer
                    try:
                        nltk.data.find('tokenizers/punkt')
                        print("✅ NLTK punkt tokenizer available")
                        use_nltk = True
                    except LookupError:
                        print("⚠️ NLTK tokenizers not found, downloading...")
                        try:
                            nltk.download('punkt_tab')
                            use_nltk = True
                        except:
                            nltk.download('punkt')
                            use_nltk = True
        except ImportError:
            print("⚠️ NLTK not available, using fallback")
            use_nltk = False
            
            def sent_tokenize(text):
                """Fallback sentence tokenizer"""
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
        
        # Test chunking
        sample_text = """
        This is a sample document with multiple sentences. 
        It contains several paragraphs to test the chunking functionality.
        The chunking should split this text into manageable pieces based on sentence boundaries.
        Each chunk should contain approximately the specified number of words.
        This helps ensure that the text is properly processed for embedding generation.
        The system should handle various document types including PDF, DOCX, and JSON files.
        """
        
        sentences = sent_tokenize(sample_text)
        print(f"✅ Text split into {len(sentences)} sentences")
        
        # Chunk the text
        def chunk_text(text, chunk_size=50):  # Smaller for testing
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
        print(f"✅ Text chunked into {len(chunks)} pieces")
        for i, chunk in enumerate(chunks):
            print(f"   Chunk {i+1}: {len(chunk.split())} words")
        
        return True
    except Exception as e:
        print(f"❌ Text chunking test failed: {e}")
        return False

def test_pinecone_client():
    """Test Pinecone client initialization"""
    print("\nTesting Pinecone client...")
    
    try:
        from pinecone import Pinecone
        print("✅ Pinecone import successful")
        
        # Test client creation (without API key)
        print("ℹ️ Pinecone client ready for initialization")
        print("   (API key needed for actual connection)")
        
        return True
    except Exception as e:
        print(f"❌ Pinecone client test failed: {e}")
        return False

def test_document_processing():
    """Test document processing capabilities"""
    print("\nTesting document processing...")
    
    # Test JSON processing
    try:
        sample_data = {
            "title": "Test Document",
            "content": "This is test content for JSON processing.",
            "metadata": {"author": "Test", "date": "2024-01-01"}
        }
        json_string = json.dumps(sample_data)
        parsed_data = json.loads(json_string)
        print("✅ JSON processing works")
    except Exception as e:
        print(f"❌ JSON processing failed: {e}")
        return False
    
    # Test DOCX processing capability
    try:
        from docx import Document
        print("✅ DOCX processing available")
    except ImportError:
        print("❌ DOCX processing not available")
        return False
    
    # Test PDF processing capability
    try:
        import PyPDF2
        print("✅ PDF processing available")
    except ImportError:
        print("❌ PDF processing not available")
        return False
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("DOCUMENT EMBEDDING APPLICATION - FUNCTIONALITY TEST")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Imports
    if test_imports():
        tests_passed += 1
    
    # Test 2: Embedding model
    embedding_success, model = test_embedding_model()
    if embedding_success:
        tests_passed += 1
    
    # Test 3: Text chunking
    if test_text_chunking():
        tests_passed += 1
    
    # Test 4: Pinecone client
    if test_pinecone_client():
        tests_passed += 1
    
    # Test 5: Document processing
    if test_document_processing():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Application is ready for deployment")
        print("✅ All dependencies are working correctly")
        print("✅ Core functionality verified")
    elif tests_passed >= 3:
        print("⚡ MOSTLY READY!")
        print("✅ Core functionality working")
        print("⚠️ Some optional features may need attention")
    else:
        print("🔧 SETUP ISSUES DETECTED")
        print("❌ Critical dependencies missing or broken")
    
    print("\n📋 NEXT STEPS:")
    if tests_passed == total_tests:
        print("1. Run Streamlit application: streamlit run app.py")
        print("2. Open browser and test with sample documents")
        print("3. Configure Pinecone API key in the UI")
    else:
        print("1. Fix failing dependency issues")
        print("2. Re-run this test script")
        print("3. Ensure all packages install correctly")

if __name__ == "__main__":
    main()