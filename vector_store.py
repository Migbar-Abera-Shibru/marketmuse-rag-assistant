import os
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from document_processor import DocumentProcessor
from config import VECTOR_STORE_PATH, COLLECTION_NAME

class VectorStoreManager:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.vector_store = None
    
    def initialize_vector_store(self, documents: Optional[List[Document]] = None):
        """Initialize or load the vector store"""
        try:
            if documents and len(documents) > 0:
                # Create new vector store with documents
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.processor.embeddings,
                    persist_directory=VECTOR_STORE_PATH,
                    collection_name=COLLECTION_NAME
                )
                self.vector_store.persist()
                print(f"Created new vector store with {len(documents)} documents")
            else:
                # Load existing vector store
                if os.path.exists(VECTOR_STORE_PATH):
                    self.vector_store = Chroma(
                        persist_directory=VECTOR_STORE_PATH,
                        embedding_function=self.processor.embeddings,
                        collection_name=COLLECTION_NAME
                    )
                    print("Loaded existing vector store")
                else:
                    self.vector_store = None
                    print("No existing vector store found")
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
            self.vector_store = None
        
        return self.vector_store
    
    def add_documents(self, file_paths: List[str]):
        """Add new documents to the vector store"""
        try:
            documents = self.processor.process_documents(file_paths)
            
            if not documents:
                print("No documents processed")
                return 0
                
            if self.vector_store is None:
                self.initialize_vector_store(documents)
            else:
                self.vector_store.add_documents(documents)
                self.vector_store.persist()
            
            print(f"Added {len(documents)} document chunks to vector store")
            return len(documents)
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            return 0
    
    def search_documents(self, query: str, k: int = 4):
        """Search for relevant documents"""
        if self.vector_store is None:
            return []
        
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return []
    
    def get_stats(self):
        """Get statistics about the vector store"""
        if self.vector_store is None:
            return {"collection_count": 0}
        
        try:
            collection = self.vector_store._client.get_collection(COLLECTION_NAME)
            return {"collection_count": collection.count()}
        except:
            return {"collection_count": 0}