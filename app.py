import streamlit as st
import os
import time
from rag_chain import RAGAssistant
from utils import save_uploaded_file, display_source_documents, init_session_state

# Page configuration
st.set_page_config(
    page_title="MarketMuse - Content Strategy Assistant",
    page_icon="📝",
    layout="wide"
)

# Initialize session state
init_session_state()

# Sidebar for document upload and API key setup
with st.sidebar:
    st.title("📁 Knowledge Base")
    
    # API key input (always show as fallback)
    api_key = st.text_input("Groq API Key", 
                          value=st.session_state.api_key,
                          type="password",
                          help="Get your API key from https://console.groq.com",
                          key="api_key_input")
    
    if st.button("Set API Key", key="set_api_key"):
        if api_key and api_key.strip():
            st.session_state.api_key = api_key.strip()
            
            # Update the RAG assistant with the new API key
            success = st.session_state.rag_assistant.update_api_key(api_key)
            st.session_state.api_key_set = success
            
            if success:
                st.success("API key set successfully! RAG assistant initialized.")
                time.sleep(1)
                st.rerun()
            else:
                error_msg = st.session_state.rag_assistant.get_initialization_error()
                st.error(f"Failed to set API key: {error_msg}")
        else:
            st.error("Please enter a valid API key")
    
    # Document upload
    st.divider()
    st.subheader("Document Management")
    
    uploaded_files = st.file_uploader(
        "Upload documents for your knowledge base",
        type=["pdf", "txt", "docx", "pptx", "html", "md"],
        accept_multiple_files=True,
        key="doc_uploader"
    )
    
    if uploaded_files and st.button("Process Documents", key="process_docs"):
        if not st.session_state.rag_assistant.is_initialized():
            st.error("Please set your Groq API key first!")
        else:
            with st.spinner("Processing documents..."):
                file_paths = []
                for uploaded_file in uploaded_files:
                    # Skip if already processed
                    if uploaded_file.name in st.session_state.processed_files:
                        st.info(f"Already processed: {uploaded_file.name}")
                        continue
                        
                    file_path = save_uploaded_file(uploaded_file)
                    file_paths.append(file_path)
                    st.session_state.processed_files.add(uploaded_file.name)
                
                if file_paths:
                    try:
                        num_docs = st.session_state.rag_assistant.add_documents(file_paths)
                        st.success(f"Processed {num_docs} document chunks!")
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
                else:
                    st.warning("No new documents to process")
    
    # Display knowledge base stats
    st.divider()
    st.subheader("Knowledge Base Status")
    
    if st.session_state.rag_assistant.is_initialized():
        stats = st.session_state.rag_assistant.get_stats()
        st.info(f"✓ Knowledge base contains {stats.get('collection_count', 0)} document chunks")
        st.success("✓ RAG assistant is ready to answer questions!")
    else:
        st.error("✗ RAG assistant not initialized. Please set your API key.")
    
    # Debug info (can be removed in production)
    st.divider()
    with st.expander("Debug Info"):
        st.write(f"API Key set: {st.session_state.get('api_key_set', False)}")
        st.write(f"API Key length: {len(st.session_state.api_key) if st.session_state.api_key else 0}")
        st.write(f"RAG Initialized: {st.session_state.rag_assistant.is_initialized() if 'rag_assistant' in st.session_state else False}")
        
        if not st.session_state.rag_assistant.is_initialized() and st.session_state.api_key:
            error_msg = st.session_state.rag_assistant.get_initialization_error()
            st.write(f"Initialization Error: {error_msg}")
    
    # Instructions
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **How to use:**
        1. Enter your Groq API key
        2. Click 'Set API Key' button
        3. Upload documents (PDF, TXT, DOCX, etc.)
        4. Click 'Process Documents'
        5. Ask questions about content strategy
        6. View sources in the expanders
        
        **Get API key:** https://console.groq.com
        """
    )

# Main content area
st.title("📝 MarketMuse - Content Strategy Assistant")
st.caption("A RAG-powered assistant for content strategy and creation")

# Display status message based on initialization
if not st.session_state.rag_assistant.is_initialized():
    st.error("""
    **RAG assistant not initialized.**
    
    Please follow these steps:
    1. Get a free API key from [Groq Console](https://console.groq.com)
    2. Enter your API key in the sidebar
    3. Click the 'Set API Key' button
    4. The assistant will initialize automatically
    """)
    
    # Show specific error if available
    if st.session_state.api_key:
        error_msg = st.session_state.rag_assistant.get_initialization_error()
        st.warning(f"Initialization error: {error_msg}")
    
    st.info("""
    **Why do I need an API key?**
    - This assistant uses Groq's powerful language models
    - API keys are free for limited usage
    - Your key is only stored in your browser session
    - No documents are sent to Groq - only your questions
    """)
else:
    st.success("✅ RAG assistant is initialized and ready to use!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message.get("sources"):
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.text(f"Document: {source.metadata.get('source', 'Unknown')}")
                    if 'page' in source.metadata:
                        st.text(f"Page: {source.metadata.get('page', 'N/A')}")
                    st.caption(source.page_content[:200] + "...")

# Chat input (only show if initialized)
if st.session_state.rag_assistant.is_initialized():
    if prompt := st.chat_input("Ask about content strategy..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Researching and generating response..."):
                try:
                    response = st.session_state.rag_assistant.query(prompt)
                    
                    # Display response
                    st.markdown(response["result"])
                    
                    # Display sources
                    if response.get("source_documents"):
                        display_source_documents(response["source_documents"])
                    
                    # Add to message history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response["result"],
                        "sources": response.get("source_documents", [])
                    })
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
else:
    # Show disabled chat input with message
    disabled_chat = st.chat_input("Enter your API key in the sidebar to enable chat...", disabled=True)