import os
import json
import streamlit as st
import time

# Import dependencies with error handling
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.error("PyPDF2 not available. PDF processing disabled.")

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    st.error("python-docx not available. DOCX processing disabled.")

try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    st.error("Pinecone not available. Vector storage disabled.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    st.error("sentence-transformers not available. Embedding generation disabled.")

try:
    from nltk.tokenize import sent_tokenize
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    st.warning("NLTK not available. Using basic text splitting.")

# Download required NLTK data if not already present
if NLTK_AVAILABLE:
    try:
        import nltk
        # Try new tokenizer first (NLTK 3.8.2+)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            # Try old tokenizer (NLTK < 3.8.2)
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                # Download appropriate tokenizer
                with st.spinner('Downloading language models (first time only)...'):
                    try:
                        nltk.download('punkt_tab')
                    except Exception:
                        nltk.download('punkt')
    except Exception as e:
        st.warning(f'NLTK setup issue: {e}. Using fallback tokenizer.')
else:
    # Fallback function for basic sentence splitting
    def sent_tokenize(text):
        """Simple sentence tokenizer fallback"""
        sentences = []
        for line in text.split('\n'):
            line = line.strip()
            if line:
                # Split on sentence endings
                parts = line.replace('!', '.').replace('?', '.').split('.')
                for part in parts:
                    part = part.strip()
                    if part:
                        sentences.append(part)
        return sentences

# Cache the embedding model for better performance
@st.cache_resource
def load_embedding_model():
    """Load and cache the sentence transformer model"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        st.error("sentence-transformers not available. Cannot load embedding model.")
        return None
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

# ---------------- Streamlit UI ----------------
st.title('Document Embedding Uploader with Progress')

# Show model loading status
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

if not st.session_state.model_loaded:
    with st.spinner('🤖 Initializing embedding model (first time may take 1-2 minutes)...'):
        try:
            load_embedding_model()
            st.session_state.model_loaded = True
            st.success('✅ Embedding model ready!')
        except Exception as e:
            st.error(f'❌ Error loading model: {e}')
else:
    st.success('✅ Embedding model ready!')

# Determine available file types based on installed dependencies
available_types = ['json']  # JSON is always available
if PDF_AVAILABLE:
    available_types.append('pdf')
if DOCX_AVAILABLE:
    available_types.append('docx')

# Show dependency status
st.sidebar.markdown("### 📦 Dependency Status")
st.sidebar.markdown(f"📄 PDF Support: {'✅' if PDF_AVAILABLE else '❌'}")
st.sidebar.markdown(f"📝 DOCX Support: {'✅' if DOCX_AVAILABLE else '❌'}")
st.sidebar.markdown(f"🔤 JSON Support: ✅")
st.sidebar.markdown(f"🤖 Embeddings: {'✅' if SENTENCE_TRANSFORMERS_AVAILABLE else '❌'}")
st.sidebar.markdown(f"🗂️ Pinecone: {'✅' if PINECONE_AVAILABLE else '❌'}")
st.sidebar.markdown(f"📚 NLTK: {'✅' if NLTK_AVAILABLE else '⚠️ (basic fallback)'}")

PINECONE_API_KEY = st.text_input('Enter Pinecone API Key', type='password')
INDEX_NAME = st.text_input('Enter Pinecone Index Name', 'document-embeddings')

uploaded_file = st.file_uploader(f'Upload {" / ".join(available_types).upper()}', type=available_types)
doc_id = st.text_input('Enter a unique document ID')

# ---------------- Functions ----------------
def extract_text(file_obj, file_type):
    """Extract text from uploaded file object"""
    if file_type == 'pdf':
        if not PDF_AVAILABLE:
            raise ValueError('PDF processing not available. Please install PyPDF2.')
        import PyPDF2
        text = ''
        reader = PyPDF2.PdfReader(file_obj)
        for page in reader.pages:
            text += page.extract_text() + '\n'
        return text
    elif file_type == 'docx':
        if not DOCX_AVAILABLE:
            raise ValueError('DOCX processing not available. Please install python-docx.')
        from docx import Document
        doc = Document(file_obj)
        return '\n'.join([p.text for p in doc.paragraphs])
    elif file_type == 'json':
        data = json.load(file_obj)
        return json.dumps(data)
    else:
        raise ValueError('Unsupported file type')

def chunk_text(text, chunk_size=500):
    """Split text into chunks based on sentence boundaries"""
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

# ---------------- Main Processing ----------------
def embed_and_upload(chunks, doc_id, source_file):
    """Generate embeddings and upload to Pinecone with progress tracking"""
    try:
        # Load the cached embedding model
        with st.spinner('Loading embedding model...'):
            model = load_embedding_model()
            if model is None:
                st.error("Cannot load embedding model. Please install sentence-transformers.")
                return False
        
        # Initialize Pinecone
        if not PINECONE_AVAILABLE:
            st.error("Pinecone client not available. Cannot upload embeddings.")
            return False
            
        with st.spinner('Connecting to Pinecone...'):
            from pinecone import Pinecone, ServerlessSpec
            pc = Pinecone(api_key=PINECONE_API_KEY)
            
            # Check if index exists, create if it doesn't
            existing_indexes = [index.name for index in pc.list_indexes()]
            if INDEX_NAME not in existing_indexes:
                st.info(f'Creating new index: {INDEX_NAME}')
                pc.create_index(
                    name=INDEX_NAME,
                    dimension=384,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
            
            index = pc.Index(INDEX_NAME)

        # Process chunks with progress bar
        st.info(f'Processing {len(chunks)} chunks...')
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, chunk in enumerate(chunks):
            status_text.text(f'Processing chunk {i + 1} of {len(chunks)}')
            
            # Generate embedding
            embedding = model.encode([chunk])[0].tolist()
            
            # Upload to Pinecone
            index.upsert([(f'{doc_id}_{i}', embedding, {
                'source': source_file, 
                'chunk': i,
                'text': chunk[:1000]  # Store first 1000 chars of chunk for reference
            })])
            
            # Update progress
            progress_bar.progress((i + 1) / len(chunks))
        
        status_text.text('Upload complete!')
        return True
        
    except Exception as e:
        st.error(f'Error during processing: {str(e)}')
        return False

# ---------------- Streamlit Button ----------------
if st.button('Upload and Embed'):
    # Check required dependencies first
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        st.error("😱 sentence-transformers is required but not installed. Please run: pip install sentence-transformers")
    elif not PINECONE_AVAILABLE:
        st.error("😱 pinecone is required but not installed. Please run: pip install pinecone")
    elif uploaded_file and doc_id and PINECONE_API_KEY:
        try:
            # Get file type
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            # Check if file type is supported
            if file_type not in available_types:
                st.error(f"File type '{file_type}' is not supported with current dependencies.")
                st.info("Install missing dependencies: pip install PyPDF2 python-docx")
            else:
                # Extract text from uploaded file
                with st.spinner('Extracting text from document...'):
                    text = extract_text(uploaded_file, file_type)
                
                if not text.strip():
                    st.warning('No text could be extracted from the document.')
                else:
                    # Chunk the text
                    chunks = chunk_text(text)
                    st.info(f'Document split into {len(chunks)} chunks')
                    
                    # Embed and upload
                    if embed_and_upload(chunks, doc_id, uploaded_file.name):
                        st.success(f'Document uploaded successfully! Total chunks: {len(chunks)}')
                        
                        # Display some statistics
                        st.subheader('Upload Statistics')
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric('Total Chunks', len(chunks))
                        with col2:
                            st.metric('Document ID', doc_id)
                        with col3:
                            st.metric('Source File', uploaded_file.name)
                        
        except Exception as e:
            st.error(f'Error processing document: {str(e)}')
    else:
        missing = []
        if not uploaded_file:
            missing.append('file upload')
        if not doc_id:
            missing.append('document ID')
        if not PINECONE_API_KEY:
            missing.append('Pinecone API key')
        
        st.error(f'Please provide: {", ".join(missing)}')

# ---------------- Additional Information ----------------
with st.expander("ℹ️ How to install missing dependencies"):
    st.markdown("""
    If you see missing dependency errors, install them with:
    
    ```bash
    # For PDF support
    pip install PyPDF2
    
    # For DOCX support  
    pip install python-docx
    
    # For embedding generation
    pip install sentence-transformers
    
    # For vector storage
    pip install pinecone
    
    # For better text processing
    pip install nltk
    
    # Install all at once
    pip install -r requirements.txt
    ```
    
    Then restart the application.
    """)
with st.expander("ℹ️ How to use this application"):
    st.markdown("""
    1. **Enter your Pinecone credentials**: You'll need your API key
    2. **Specify an index name**: This will be created if it doesn't exist
    3. **Upload a document**: Supported formats are PDF, DOCX, and JSON
    4. **Provide a unique document ID**: This helps identify your document in the vector database
    5. **Click 'Upload and Embed'**: The app will process your document and show progress
    
    **Note**: The app uses the `sentence-transformers/all-MiniLM-L6-v2` model which creates 384-dimensional embeddings.
    """)

with st.expander("🔧 Configuration"):
    st.markdown("""
    - **Chunk Size**: Documents are split into chunks of approximately 500 words
    - **Model**: sentence-transformers/all-MiniLM-L6-v2 (Free, Local)
    - **Embedding Dimension**: 384
    - **Supported Formats**: PDF, DOCX, JSON
    - **API Requirements**: Only Pinecone (embeddings are generated locally)
    """)

with st.expander("☁️ Deployment Information"):
    st.markdown("""
    **For Streamlit Cloud deployment:**
    - ✅ No additional API keys needed for embeddings
    - ⏱️ First deployment may take 1-2 minutes (model download)
    - 💾 Model is cached after first load
    - 🔒 Documents are processed securely in the cloud
    - 📊 Free tier supports this application
    
    **Memory usage:** ~300MB for the embedding model
    **Model:** sentence-transformers/all-MiniLM-L6-v2 (local, no API)
    """)