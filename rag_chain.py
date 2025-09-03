from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from vector_store import VectorStoreManager
from config import GROQ_API_KEY, GROQ_MODEL
import os
import requests
import json
import sys

class RAGAssistant:
    def __init__(self, api_key=None):
        print("Initializing RAGAssistant...")
        self.vector_store_manager = VectorStoreManager()
        
        # Use provided API key or fall back to config
        self.api_key = api_key or GROQ_API_KEY
        self.llm = None
        self.qa_chain = None
        self.initialization_error = "API key not provided"
        
        # Initialize if API key is available
        if self.api_key and self.api_key.strip():
            self._initialize_llm()
        else:
            print("No API key provided during initialization")
    
    def _test_groq_api_key(self, api_key):
        """Test if the Groq API key is valid"""
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Test with a simple request
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json={
                    "model": GROQ_MODEL,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "temperature": 0.1,
                    "max_tokens": 10
                },
                timeout=15
            )
            
            if response.status_code == 200:
                return True, "API key is valid"
            elif response.status_code == 401:
                return False, "Invalid API key - authentication failed"
            elif response.status_code == 403:
                return False, "API key doesn't have permission to access this model"
            elif response.status_code == 429:
                return False, "Rate limit exceeded or quota exceeded"
            else:
                return False, f"API error: {response.status_code} - {response.text}"
                
        except requests.exceptions.Timeout:
            return False, "Request timeout - check your internet connection"
        except requests.exceptions.ConnectionError:
            return False, "Connection error - check your internet connection"
        except Exception as e:
            return False, f"Error testing API key: {str(e)}"
    
    def _initialize_llm(self):
        """Initialize the LLM with the API key"""
        print("Initializing LLM...")
        
        # Test the API key first
        is_valid, error_msg = self._test_groq_api_key(self.api_key)
        
        if not is_valid:
            self.initialization_error = error_msg
            print(f"API key validation failed: {error_msg}")
            return
        
        try:
            # Set the API key in environment
            os.environ["GROQ_API_KEY"] = self.api_key
            
            # Initialize the Groq client
            print("Creating Groq client...")
            self.llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name=GROQ_MODEL,
                temperature=0.1
            )
            
            # Test the connection with a simple prompt
            print("Testing LLM connection...")
            test_response = self.llm.invoke("Hello")
            print("Groq API connection test successful")
            
            # Initialize vector store
            print("Initializing vector store...")
            self.vector_store_manager.initialize_vector_store()
            
            # Create the QA chain
            print("Creating QA chain...")
            self.qa_chain = self._create_qa_chain()
            
            if self.qa_chain:
                print("RAG assistant initialized successfully")
                self.initialization_error = None
            else:
                self.initialization_error = "Failed to create QA chain"
                print(f"QA chain creation failed: {self.initialization_error}")
                
        except Exception as e:
            error_msg = f"Error initializing Groq client: {str(e)}"
            print(f"Initialization error: {error_msg}")
            import traceback
            traceback.print_exc()
            self.initialization_error = error_msg
            self.llm = None
            self.qa_chain = None
    
    def _create_qa_chain(self):
        """Create the RAG QA chain with custom prompt that strictly uses only provided context"""
        if not self.llm:
            print("No LLM available for QA chain creation")
            return None
            
        try:
            # STRICT prompt that only uses provided context
            prompt_template = """You are a content strategy assistant that answers questions STRICTLY based on the provided context documents. 
            If the answer cannot be found in the context, you must say "I cannot answer that question based on the provided documents."

            Context: {context}

            Question: {question}
            
            Instructions:
            1. ONLY use information from the context provided
            2. If the context doesn't contain relevant information, say "I cannot answer that question based on the provided documents."
            3. Do not make up information or use external knowledge
            4. Provide specific quotes or references from the context when possible
            5. If asked about general topics not in the documents, refer to rule 2
            
            Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )
            
            chain_type_kwargs = {"prompt": PROMPT}
            
            # Check if vector store is available and has documents
            if (self.vector_store_manager.vector_store and 
                self.vector_store_manager.get_stats().get("collection_count", 0) > 0):
                
                retriever = self.vector_store_manager.vector_store.as_retriever(
                    search_kwargs={"k": 4}  # Retrieve more documents for better context
                )
                
                return RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs=chain_type_kwargs,
                    return_source_documents=True
                )
            else:
                # Create a chain that will reject all questions until documents are added
                print("No documents in vector store, creating document-aware chain")
                from langchain.chains import LLMChain
                
                no_docs_prompt = PromptTemplate(
                    input_variables=["question"],
                    template="I cannot answer questions yet because no documents have been uploaded. Please upload documents first and then ask questions about their content."
                )
                return LLMChain(llm=self.llm, prompt=no_docs_prompt)
                
        except Exception as e:
            print(f"Error creating QA chain: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def is_initialized(self):
        """Check if the RAG assistant is properly initialized"""
        return self.llm is not None
    
    def has_documents(self):
        """Check if the vector store has documents"""
        return (self.vector_store_manager.vector_store and 
                self.vector_store_manager.get_stats().get("collection_count", 0) > 0)
    
    def query(self, question: str):
        """Query the RAG system - only answers based on uploaded documents"""
        if not self.is_initialized():
            return {"result": "RAG assistant not initialized. Please check your API key and try again.", "source_documents": []}
        
        # Check if we have documents
        if not self.has_documents():
            return {"result": "No documents have been uploaded yet. Please upload documents first and then ask questions about their content.", "source_documents": []}
        
        try:
            result = self.qa_chain({"query": question})
            
            # Additional check to ensure the answer is based on documents
            if (not result.get("source_documents") or 
                len(result.get("source_documents", [])) == 0):
                result["result"] = "I cannot answer that question based on the provided documents."
            
            return result
        except Exception as e:
            return {"result": f"Error querying the system: {str(e)}", "source_documents": []}
    
    def add_documents(self, file_paths: list):
        """Add documents to the knowledge base and recreate QA chain"""
        result = self.vector_store_manager.add_documents(file_paths)
        if result > 0:
            # Recreate QA chain with updated documents
            self.qa_chain = self._create_qa_chain()
        return result
    
    def get_stats(self):
        """Get statistics about the knowledge base"""
        return self.vector_store_manager.get_stats()
    
    def update_api_key(self, new_api_key):
        """Update the API key and reinitialize"""
        if new_api_key and new_api_key.strip():
            self.api_key = new_api_key.strip()
            self._initialize_llm()
            return self.is_initialized()
        return False
    
    def get_initialization_error(self):
        """Get error message if initialization failed"""
        if self.is_initialized():
            return "No error - initialized successfully"
        
        if not self.api_key or not self.api_key.strip():
            return "API key not provided"
        
        return self.initialization_error or "Unknown initialization error"
    
    def test_api_key_directly(self):
        """Test the API key directly and return detailed results"""
        if not self.api_key or not self.api_key.strip():
            return False, "API key not provided"
        
        return self._test_groq_api_key(self.api_key)