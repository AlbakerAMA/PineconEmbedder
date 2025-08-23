"""
Comprehensive dependency check for Document Embedding Project
Checks all required packages and their versions
"""

import sys
print(f"Python version: {sys.version}")
print("=" * 50)

# Check core dependencies
dependencies = {
    'streamlit': 'Streamlit UI framework',
    'sentence_transformers': 'Embedding model (sentence-transformers/all-MiniLM-L6-v2)',
    'pinecone': 'Vector database client (NEW package name)',
    'docx': 'DOCX document processing (python-docx)',
    'PyPDF2': 'PDF document processing',
    'nltk': 'Natural language processing',
    'pandas': 'Data manipulation',
    'json': 'JSON processing (built-in)',
    'os': 'Operating system interface (built-in)'
}

available = []
missing = []
version_info = {}

for package, description in dependencies.items():
    try:
        if package == 'docx':
            # Special case: python-docx imports as 'docx' but we need Document
            from docx import Document
            import docx
            version = getattr(docx, '__version__', 'unknown')
            print(f"✅ {package}: {description} (v{version})")
            available.append(package)
            version_info[package] = version
        elif package == 'pinecone':
            # New Pinecone SDK
            from pinecone import Pinecone
            import pinecone
            version = getattr(pinecone, '__version__', 'unknown')
            print(f"✅ {package}: {description} (v{version})")
            available.append(package)
            version_info[package] = version
        else:
            module = __import__(package)
            version = getattr(module, '__version__', 'built-in' if package in ['json', 'os'] else 'unknown')
            print(f"✅ {package}: {description} (v{version})")
            available.append(package)
            version_info[package] = version
            
    except ImportError as e:
        print(f"❌ {package}: {description} - NOT AVAILABLE")
        print(f"   Error: {e}")
        missing.append(package)
    except Exception as e:
        print(f"⚠️  {package}: {description} - ERROR")
        print(f"   Error: {e}")
        missing.append(package)

print("\n" + "=" * 50)
print("SUMMARY:")
print(f"✅ Available: {len(available)}/{len(dependencies)} packages")
print(f"❌ Missing: {len(missing)} packages")

if missing:
    print(f"\nMissing packages: {', '.join(missing)}")
    print("\nInstall missing packages with:")
    for pkg in missing:
        if pkg == 'docx':
            print(f"  pip install python-docx")
        else:
            print(f"  pip install {pkg}")

print("\n" + "=" * 50)
print("FUNCTIONAL TESTS:")

# Test sentence transformers
if 'sentence_transformers' in available:
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformer import successful")
        
        # Test model loading (this might take time on first run)
        print("🔄 Testing model loading (may take 1-2 minutes on first run)...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✅ Model loaded successfully")
        
        # Test embedding generation
        test_text = "This is a test sentence for embedding."
        embedding = model.encode([test_text])[0]
        print(f"✅ Embedding generated: {len(embedding)} dimensions")
        
    except Exception as e:
        print(f"❌ SentenceTransformer test failed: {e}")

# Test Pinecone
if 'pinecone' in available:
    try:
        from pinecone import Pinecone
        print("✅ Pinecone import successful")
        print("ℹ️  Pinecone client ready (API key needed for actual connection)")
    except Exception as e:
        print(f"❌ Pinecone test failed: {e}")

# Test document processing
if 'docx' in available:
    try:
        from docx import Document
        print("✅ DOCX processing available")
    except Exception as e:
        print(f"❌ DOCX test failed: {e}")

if 'PyPDF2' in available:
    try:
        import PyPDF2
        print("✅ PDF processing available")
    except Exception as e:
        print(f"❌ PDF test failed: {e}")

# Test NLTK
if 'nltk' in available:
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
        print("✅ NLTK available")
        
        # Check if punkt is downloaded
        try:
            nltk.data.find('tokenizers/punkt')
            print("✅ NLTK punkt tokenizer available")
        except LookupError:
            print("⚠️  NLTK punkt tokenizer not downloaded - will download automatically")
            
    except Exception as e:
        print(f"❌ NLTK test failed: {e}")

print("\n" + "=" * 50)
print("DEPLOYMENT READINESS:")

core_deps = ['streamlit', 'sentence_transformers', 'pinecone']
doc_deps = ['docx', 'PyPDF2']
optional_deps = ['nltk', 'pandas']

core_ready = all(dep in available for dep in core_deps)
doc_ready = any(dep in available for dep in doc_deps)

if core_ready and doc_ready:
    print("🎉 ALL SYSTEMS GO! Ready for full deployment")
elif core_ready:
    print("⚡ CORE READY! Can process JSON and generate embeddings")
    print("📄 Install document processors for PDF/DOCX support")
else:
    print("🔧 SETUP NEEDED! Install missing core dependencies")

print(f"\nCore dependencies: {core_ready}")
print(f"Document processing: {doc_ready}")
print(f"Optional features: {all(dep in available for dep in optional_deps)}")