import os
import tempfile
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader, UnstructuredHTMLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, SUPPORTED_EXTENSIONS

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    def load_document(self, file_path: str):
        """Load document based on file extension"""
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif ext in ['.docx', '.doc']:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif ext in ['.pptx', '.ppt']:
                loader = UnstructuredPowerPointLoader(file_path)
            elif ext in ['.html', '.htm']:
                loader = UnstructuredHTMLLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            
            return loader.load()
        except Exception as e:
            print(f"Error loading document {file_path}: {str(e)}")
            return []
    
    def process_documents(self, file_paths: List[str]):
        """Process multiple documents and split into chunks"""
        all_docs = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
            
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                print(f"Skipping unsupported file type: {file_path}")
                continue
                
            print(f"Processing: {file_path}")
            try:
                docs = self.load_document(file_path)
                if docs:
                    chunks = self.text_splitter.split_documents(docs)
                    # Add source metadata to each chunk
                    for chunk in chunks:
                        if 'source' not in chunk.metadata:
                            chunk.metadata['source'] = file_path
                    all_docs.extend(chunks)
                    print(f"Added {len(chunks)} chunks from {file_path}")
                else:
                    print(f"No content extracted from {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        return all_docs