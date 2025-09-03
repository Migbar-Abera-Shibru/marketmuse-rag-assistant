import os
import streamlit as st
from typing import List
from rag_chain import RAGAssistant

def save_uploaded_file(uploaded_file, save_dir: str = "data") -> str:
    """Save uploaded file to directory and return path"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def display_source_documents(source_docs):
    """Display source documents in a user-friendly format"""
    if not source_docs:
        return
    
    with st.expander("Source Documents (References)"):
        for i, doc in enumerate(source_docs):
            st.markdown(f"**Document {i+1}**")
            st.text(f"Source: {doc.metadata.get('source', 'Unknown')}")
            if 'page' in doc.metadata:
                st.text(f"Page: {doc.metadata.get('page', 'N/A')}")
            st.text(f"Content: {doc.page_content[:200]}...")
            st.divider()

def init_session_state():
    """Initialize session state variables"""
    if "rag_assistant" not in st.session_state:
        # Initialize without API key first
        st.session_state.rag_assistant = RAGAssistant()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.getenv("GROQ_API_KEY", "")
    
    if "api_key_set" not in st.session_state:
        st.session_state.api_key_set = False