import streamlit as st
import os
import time
from rag_chain import RAGAssistant
from utils import save_uploaded_file, display_source_documents, init_session_state

# Page configuration
st.set_page_config(
    page_title="MarketMuse - Document-Based Content Assistant",
    page_icon="📝",
    layout="wide"
)

# Initialize session state
init_session_state()

# Sidebar for document upload and API key setup
with st.sidebar:
    st.title("📁 Document Management")
    
    # API key input
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
                st.success("API key set successfully! Assistant initialized.")
                time.sleep(1)
                st.rerun()
            else:
                error_msg = st.session_state.rag_assistant.get_initialization_error()
                st.error(f"Failed to set API key: {error_msg}")
        else:
            st.error("Please enter a valid API key")
    
    # Document upload section
    st.divider()
    st.subheader("Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload documents for analysis",
        type=["pdf", "txt", "docx", "pptx", "html", "md"],
        accept_multiple_files=True,
        key="doc_uploader",
        help="Upload documents that you want to ask questions about"
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
                        if num_docs > 0:
                            st.success(f"Processed {num_docs} document chunks! You can now ask questions about these documents.")
                        else:
                            st.warning("No new content was extracted from the documents.")
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
                else:
                    st.warning("No new documents to process")
    
    # Document status
    st.divider()
    st.subheader("Document Status")
    
    if st.session_state.rag_assistant.is_initialized():
        stats = st.session_state.rag_assistant.get_stats()
        doc_count = stats.get("collection_count", 0)
        
        if doc_count > 0:
            st.success(f"✓ {doc_count} document chunks ready for analysis")
            st.info("You can ask questions about your uploaded documents.")
        else:
            st.warning("No documents uploaded yet. Please upload documents to ask questions.")
    else:
        st.error("✗ Assistant not initialized. Please set your API key.")
    
    # Clear chat button
    st.divider()
    if st.button("Clear Chat History", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()
    
    # Instructions
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **How to use:**
        1. Set your Groq API key
        2. Upload documents (PDF, TXT, DOCX, etc.)
        3. Click 'Process Documents'
        4. Ask questions **only about the uploaded documents**
        5. View sources in the expanders
        
        **Note:** This assistant will ONLY answer questions based on your uploaded documents.
        """
    )

# Main content area
st.title("📝 Document-Based Content Assistant")
st.caption("Ask questions specifically about your uploaded documents")

# Display status message
if not st.session_state.rag_assistant.is_initialized():
    st.error("""
    **Assistant not initialized.**
    
    Please follow these steps:
    1. Get a free API key from [Groq Console](https://console.groq.com)
    2. Enter your API key in the sidebar
    3. Click the 'Set API Key' button
    """)
    
    if st.session_state.api_key:
        error_msg = st.session_state.rag_assistant.get_initialization_error()
        st.warning(f"**Initialization error:** {error_msg}")

elif not st.session_state.rag_assistant.has_documents():
    st.warning("""
    **No documents uploaded.**
    
    Please upload documents first:
    1. Go to the sidebar
    2. Upload documents (PDF, TXT, DOCX, etc.)
    3. Click 'Process Documents'
    4. Then ask questions about your documents
    """)
else:
    st.success("✅ Assistant is ready! You can ask questions about your uploaded documents.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message.get("sources"):
            with st.expander("View Source Documents"):
                for source in message["sources"]:
                    st.text(f"Document: {source.metadata.get('source', 'Unknown')}")
                    if 'page' in source.metadata:
                        st.text(f"Page: {source.metadata.get('page', 'N/A')}")
                    st.caption(source.page_content[:200] + "...")

# Chat input (only show if initialized and has documents)
if (st.session_state.rag_assistant.is_initialized() and 
    st.session_state.rag_assistant.has_documents()):
    
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching through your documents..."):
                try:
                    response = st.session_state.rag_assistant.query(prompt)
                    
                    # Display response
                    st.markdown(response["result"])
                    
                    # Display sources if available
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
elif st.session_state.rag_assistant.is_initialized():
    # Show message about needing documents
    disabled_chat = st.chat_input("Upload documents in the sidebar to ask questions...", disabled=True)
else:
    # Show disabled chat input
    disabled_chat = st.chat_input("Set your API key in the sidebar to enable chat...", disabled=True)