import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Page configuration
st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Enterprise Knowledge Assistant")
st.markdown("---")

# Initialize session state for caching
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
    st.session_state.llm = None
    st.session_state.db_loaded = False

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    model_name = st.selectbox(
        "Select LLM Model",
        ["mistral", "llama2", "neural-chat"],
        help="Choose the Ollama model to use"
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Lower values make output more deterministic"
    )
    
    # Number of retrieved documents
    num_docs = st.slider(
        "Number of Retrieved Documents",
        min_value=1,
        max_value=10,
        value=2,
        help="How many relevant documents to retrieve"
    )
    
    st.markdown("---")
    
    # Vector DB status
    if st.session_state.db_loaded:
        st.success("✅ Vector DB Loaded")
    else:
        st.info("⏳ Vector DB not loaded yet")

# Main content area
col1, col2 = st.columns([1, 1], gap="medium")

# Left column: Initialize vector DB
with col1:
    st.subheader("📚 Document Processing")
    
    if st.button("Initialize Vector Database", use_container_width=True, key="init_db"):
        with st.spinner("Loading document and creating embeddings..."):
            try:
                # Load document
                loader = TextLoader("input_data/sample_doc.txt")
                response = loader.load()
                
                # Split into chunks
                text_splitters = RecursiveCharacterTextSplitter(
                    chunk_size=300,
                    chunk_overlap=50
                )
                chunks = text_splitters.split_documents(response)
                
                st.info(f"📊 Total chunks created: {len(chunks)}")
                
                # Create embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                
                # Create and save vector DB
                vector_db = FAISS.from_documents(chunks, embeddings)
                vector_db.save_local("vector_db")
                
                # Store in session state
                st.session_state.vector_db = vector_db
                st.session_state.db_loaded = True
                
                st.success("✅ Vector database initialized successfully!")
                st.rerun()
                
            except FileNotFoundError:
                st.error("❌ Document file not found at 'input_data/sample_doc.txt'")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Load existing vector DB
    if st.button("Load Existing Vector Database", use_container_width=True, key="load_db"):
        with st.spinner("Loading vector database..."):
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                vector_db = FAISS.load_local(
                    "vector_db",
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                st.session_state.vector_db = vector_db
                st.session_state.db_loaded = True
                st.success("✅ Vector database loaded successfully!")
                st.rerun()
            except FileNotFoundError:
                st.error("❌ Vector database not found. Initialize first.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Right column: Query interface
with col2:
    st.subheader("🔍 Query Assistant")
    
    if st.session_state.db_loaded:
        query = st.text_area(
            "Enter your question:",
            placeholder="What does an enterprise knowledge assistant do?",
            height=100,
            key="query_input"
        )
        
        if st.button("Get Answer", use_container_width=True, type="primary", key="submit_query"):
            if not query:
                st.warning("⚠️ Please enter a question")
            else:
                with st.spinner("Searching and generating response..."):
                    try:
                        # Retrieve relevant documents
                        docs = st.session_state.vector_db.similarity_search(query, k=num_docs)
                        
                        # Initialize LLM
                        llm = OllamaLLM(
                            model=model_name,
                            temperature=temperature
                        )
                        
                        # Create context from retrieved documents
                        context = "\n\n".join([doc.page_content for doc in docs])
                        
                        # Create prompt
                        prompt = f"""
You are an enterprise knowledge assistant.
Answer the question strictly using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""
                        
                        # Get response
                        response = llm.invoke(prompt)
                        
                        st.session_state.last_query = query
                        st.session_state.last_docs = docs
                        st.session_state.last_response = response
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    else:
        st.info("📌 Please initialize or load the vector database first to ask questions.")

# Display results
st.markdown("---")
st.subheader("📋 Results")

if "last_response" in st.session_state:
    # Create tabs for organized display
    tab1, tab2 = st.tabs(["Answer", "Retrieved Documents"])
    
    with tab1:
        st.markdown("### Final Answer")
        st.markdown(st.session_state.last_response)
    
    with tab2:
        st.markdown("### Retrieved Documents")
        for i, doc in enumerate(st.session_state.last_docs, 1):
            with st.expander(f"Document {i}"):
                st.markdown(doc.page_content)
else:
    st.info("💡 Results will appear here after you submit a query.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🚀 Powered by LangChain, FAISS, and Ollama</p>
    </div>
    """,
    unsafe_allow_html=True
)
