"""
Simple test script to verify core functionality
This tests the basic components without requiring all dependencies
"""

import sys
import json

def test_basic_imports():
    """Test basic Python functionality"""
    print("=== Testing Basic Python Functionality ===")
    try:
        import os
        import json
        print("✓ Basic imports working")
        return True
    except Exception as e:
        print(f"✗ Basic imports failed: {e}")
        return False

def test_json_processing():
    """Test JSON processing"""
    print("\n=== Testing JSON Processing ===")
    try:
        sample_data = {
            "title": "Test Document",
            "content": "This is test content for JSON processing.",
            "metadata": {"author": "Test", "date": "2024-01-01"}
        }
        json_string = json.dumps(sample_data)
        parsed_data = json.loads(json_string)
        print("✓ JSON processing works")
        print(f"✓ Sample data: {len(json_string)} characters")
        return True
    except Exception as e:
        print(f"✗ JSON processing failed: {e}")
        return False

def test_text_chunking():
    """Test basic text chunking without NLTK"""
    print("\n=== Testing Text Chunking ===")
    try:
        sample_text = """
        This is a sample document with multiple sentences. 
        It contains several paragraphs to test the chunking functionality.
        The chunking should split this text into manageable pieces.
        Each chunk should contain approximately the specified number of words.
        """
        
        # Simple sentence splitting (fallback without NLTK)
        sentences = [s.strip() for s in sample_text.split('.') if s.strip()]
        print(f"✓ Text split into {len(sentences)} sentences")
        
        # Simple chunking
        chunks = []
        current_chunk = []
        word_count = 0
        
        for sentence in sentences:
            words = sentence.split()
            word_count += len(words)
            current_chunk.append(sentence)
            
            if word_count >= 20:  # Small chunk for testing
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = []
                word_count = 0
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
            
        print(f"✓ Text chunked into {len(chunks)} pieces")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}: {len(chunk.split())} words")
        return True
    except Exception as e:
        print(f"✗ Text chunking failed: {e}")
        return False

def test_file_operations():
    """Test file operations"""
    print("\n=== Testing File Operations ===")
    try:
        # Test writing and reading a file
        test_file = "test_temp.txt"
        test_content = "This is a test file for verification."
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        with open(test_file, 'r') as f:
            read_content = f.read()
        
        # Cleanup
        import os
        os.remove(test_file)
        
        if read_content == test_content:
            print("✓ File operations work correctly")
            return True
        else:
            print("✗ File content mismatch")
            return False
    except Exception as e:
        print(f"✗ File operations failed: {e}")
        return False

def check_available_modules():
    """Check which modules are available"""
    print("\n=== Checking Available Modules ===")
    modules_to_check = [
        'streamlit', 'sentence_transformers', 'pinecone', 
        'PyPDF2', 'docx', 'nltk', 'pandas'
    ]
    
    available = []
    missing = []
    
    for module in modules_to_check:
        try:
            __import__(module)
            available.append(module)
            print(f"✓ {module} - Available")
        except ImportError:
            missing.append(module)
            print(f"✗ {module} - Missing")
        except Exception as e:
            missing.append(module)
            print(f"⚠ {module} - Error: {e}")
    
    print(f"\nSummary: {len(available)} available, {len(missing)} missing")
    return available, missing

def main():
    """Run all tests"""
    print("=== Core Functionality Test Suite ===")
    print(f"Python version: {sys.version}")
    print()
    
    tests = [
        test_basic_imports,
        test_json_processing,
        test_text_chunking,
        test_file_operations
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Check available modules
    available, missing = check_available_modules()
    
    print(f"\n=== Final Results ===")
    print(f"Core tests passed: {sum(results)}/{len(results)}")
    print(f"Required modules available: {len(available)}/7")
    
    if all(results):
        print("✓ Core functionality works!")
        if len(available) >= 5:
            print("✓ Most dependencies available - app should work")
        else:
            print("⚠ Some dependencies missing - may need installation")
    else:
        print("✗ Some core tests failed")
    
    return results, available, missing

if __name__ == "__main__":
    main()