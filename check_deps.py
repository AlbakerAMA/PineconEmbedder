try:
    from docx import Document
    print('✅ python-docx is available')
    docx_available = True
except ImportError:
    print('❌ python-docx not available')
    docx_available = False

try:
    from pinecone import Pinecone
    print('✅ pinecone is available')
    pinecone_available = True
except ImportError:
    print('❌ pinecone not available')
    pinecone_available = False

try:
    from sentence_transformers import SentenceTransformer
    print('✅ sentence-transformers is available')
    st_available = True
except ImportError:
    print('❌ sentence-transformers not available')
    st_available = False

print(f"\nSummary:")
print(f"DOCX support: {docx_available}")
print(f"Pinecone support: {pinecone_available}")
print(f"Embeddings support: {st_available}")

if docx_available and pinecone_available and st_available:
    print("\n🎉 All core dependencies are available!")
elif docx_available and st_available:
    print("\n⚠️ DOCX and embeddings available, but Pinecone missing")
else:
    print(f"\n⚠️ Some dependencies still missing")