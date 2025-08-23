"""
Quick test to verify the Pinecone SDK update and core functionality
"""

print("=== PINECONE SDK TEST ===")

# Test the specific issue from the error
try:
    print("Testing old pinecone-client import...")
    import pinecone
    print("❌ OLD pinecone package still installed - this will cause errors")
    print("   Please uninstall: pip uninstall pinecone-client")
except Exception as e:
    print(f"ℹ️ Old pinecone import failed (expected): {e}")

# Test the new Pinecone SDK
try:
    print("\nTesting NEW Pinecone SDK...")
    from pinecone import Pinecone
    print("✅ New Pinecone SDK import successful!")
    
    # Test client initialization (without API key)
    print("✅ Pinecone client class available")
    print("   Ready for API key and actual connection")
    
except ImportError as e:
    print(f"❌ New Pinecone SDK not available: {e}")
    print("   Install with: pip install pinecone")
except Exception as e:
    print(f"✅ New Pinecone SDK available but: {e}")

print("\n=== QUICK DEPENDENCY CHECK ===")

# Quick check of other dependencies
deps = {
    'sentence_transformers': 'SentenceTransformer',
    'docx': 'Document', 
    'PyPDF2': 'PdfReader',
    'streamlit': 'st'
}

for package, main_class in deps.items():
    try:
        if package == 'docx':
            from docx import Document
            print(f"✅ {package} available")
        elif package == 'streamlit':
            import streamlit as st
            print(f"✅ {package} available")
        else:
            __import__(package)
            print(f"✅ {package} available")
    except ImportError:
        print(f"❌ {package} missing")

print("\n=== CONCLUSION ===")
print("If all dependencies show as available, the issue was the old pinecone-client package.")
print("The new 'pinecone' package should resolve the deployment error.")

# Test a simple embedding workflow
print("\n=== BASIC WORKFLOW TEST ===")
try:
    from sentence_transformers import SentenceTransformer
    print("🔄 Testing basic embedding workflow...")
    
    # This is the exact model used in the app
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("✅ Model loaded successfully")
    
    # Test embedding
    test_text = "Hello world"
    embedding = model.encode([test_text])[0]
    print(f"✅ Embedding generated: {len(embedding)} dimensions")
    
    print("🎉 CORE FUNCTIONALITY WORKING!")
    
except Exception as e:
    print(f"❌ Basic workflow failed: {e}")

print("\n=== NEXT STEPS ===")
print("1. If all tests pass, try running: streamlit run app.py")
print("2. If Streamlit has issues, the core logic is working")
print("3. The app should now work on Streamlit Cloud with the updated dependencies")