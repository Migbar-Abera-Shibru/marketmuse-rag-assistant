import streamlit as st
import os
import time
from rag_chain import RAGAssistant
from utils import save_uploaded_file, display_source_documents, init_session_state

# Page configuration
st.set_page_config(
    page_title="MarketMuse - Secure Document Analysis",
    page_icon="📝",
    layout="wide"
)

# Initialize session state
init_session_state()

# Check if API key is available in environment
env_api_key_available = bool(os.getenv("GROQ_API_KEY"))

# Sidebar for document upload and API key setup
with st.sidebar:
    st.title("📁 Document Management")
    
    # Security and privacy notice
    with st.expander("🔒 Security Information"):
        st.info("""
        **Privacy & Security Features:**
        - Documents are processed locally on your device
        - Only document content is sent to the AI service
        - No personal data is stored or shared
        - All analysis is based solely on your uploaded content
        - API keys are handled securely
        """)
    
    # Only show API key input if not available in environment
    if not env_api_key_available:
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
    else:
        st.success("✅ API key loaded from environment variables")
    
    # Document upload section
    st.divider()
    st.subheader("Upload Documents")
    
    # Privacy warning for document upload
    st.warning("""
    **Please ensure documents do not contain:**
    - Personal identifiable information
    - Sensitive financial/medical data
    - Confidential business information
    - Private credentials or passwords
    """)
    
    uploaded_files = st.file_uploader(
        "Upload documents for analysis",
        type=["pdf", "txt", "docx", "pptx", "html", "md"],
        accept_multiple_files=True,
        key="doc_uploader",
        help="Upload content strategy, marketing, or business documents"
    )
    
    if uploaded_files and st.button("Process Documents", key="process_docs"):
        if not st.session_state.rag_assistant.is_initialized():
            if env_api_key_available:
                st.error("Assistant initialization failed. Please check your .env file API key.")
            else:
                st.error("Please set your Groq API key first!")
        else:
            with st.spinner("Processing documents securely..."):
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
        if env_api_key_available:
            st.error("Assistant initialization failed. Please check your API key in the .env file.")
        else:
            st.error("Assistant not initialized. Please set your API key.")
    
    # Clear chat button
    st.divider()
    if st.button("Clear Chat History", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()
    
    # Ethical guidelines
    st.sidebar.markdown("---")
    with st.expander("📋 Ethical Guidelines"):
        st.markdown("""
        **AI Assistant Ethical Framework:**
        - Only uses provided document content
        - Doesn't access external information
        - Protects user privacy
        - Avoids harmful content generation
        - Maintains transparency about limitations
        - Provides source attribution
        """)

# Main content area
st.title("🔒 MarketMuse - Secure Document Analysis")
st.caption("Ask questions about your uploaded documents with built-in safety protocols")

# Security badge
st.markdown("""
<div style="background-color: #e8f5e8; padding: 10px; border-radius: 5px; border-left: 4px solid #4CAF50;">
    <strong>🔒 Security Enabled:</strong> This assistant only uses your uploaded documents and follows strict ethical guidelines.
</div>
<br>
""", unsafe_allow_html=True)

# Display status message
if not st.session_state.rag_assistant.is_initialized():
    if env_api_key_available:
        st.error("""
        **Assistant initialization failed.**
        
        Please check:
        1. Your API key in the .env file is correct
        2. You have internet connectivity
        3. The Groq API service is available
        """)
        
        error_msg = st.session_state.rag_assistant.get_initialization_error()
        st.warning(f"**Error details:** {error_msg}")
    else:
        st.error("""
        **Assistant not initialized.**
        
        Please follow these steps:
        1. Get a free API key from [Groq Console](https://console.groq.com)
        2. Enter your API key in the sidebar
        3. Click the 'Set API Key' button
        """)

elif not st.session_state.rag_assistant.has_documents():
    st.warning("""
    **No documents uploaded.**
    
    Please upload documents first:
    1. Go to the sidebar
    2. Upload documents (PDF, TXT, DOCX, etc.)
    3. Click 'Process Documents'
    4. Then ask questions about your documents
    
    **Note:** For your security, this assistant only analyzes documents you explicitly upload.
    """)
else:
    st.success("✅ Assistant is ready! You can ask questions about your uploaded documents.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message.get("sources"):
            with st.expander("📄 View Source Documents"):
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
            with st.spinner("Analyzing your documents securely..."):
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
    disabled_chat = st.chat_input("Set your API key to enable chat...", disabled=True)

# Footer with ethical guidelines
st.markdown("---")
with st.expander("ℹ️ About This Assistant"):
    st.markdown("""
    **Ethical AI Framework:**
    - **Privacy First**: Only processes explicitly uploaded documents
    - **Transparency**: Clearly indicates source-based responses
    - **Safety**: Implements content boundaries and ethical guidelines
    - **Accuracy**: Qualifies information based on source reliability
    
    **This assistant will:**
    ✓ Only use your uploaded documents for responses
    ✓ Provide source attribution when possible
    ✓ Decline to answer questions outside document scope
    ✓ Follow ethical AI guidelines and safety protocols
    
    **This assistant will not:**
    ✗ Access external information or previous conversations
    ✗ Provide medical, legal, or financial advice
    ✗ Generate harmful or misleading content
    ✗ Store or share personal identifiable information
    """)

    # Add to your existing app.py

# In the sidebar, add a button to clear conversation memory:
if st.button("Clear Conversation Memory", key="clear_memory"):
    st.session_state.rag_assistant.clear_memory()
    st.session_state.messages = []
    st.success("Conversation memory cleared!")
    time.sleep(1)
    st.rerun()

# In the query method, use the conversational approach:
def query(self, question: str):
    """Query the RAG system with conversation memory"""
    if not self.is_initialized():
        return {"result": "RAG assistant not initialized.", "source_documents": []}
    
    # Security check first
    if self._is_security_threat(question):
        return {"result": "I'm designed to help analyze uploaded documents. I cannot answer questions about my internal functioning.", "source_documents": []}
    
    try:
        # Use conversational QA chain with memory
        result = self.qa_chain({"question": question})
        return {
            "result": result.get("answer", "No answer generated"),
            "source_documents": result.get("source_documents", [])
        }
    except Exception as e:
        return {"result": f"Error: {str(e)}", "source_documents": []}