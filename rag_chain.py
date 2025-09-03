from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from vector_store import VectorStoreManager
from config import GROQ_API_KEY, GROQ_MODEL
import os

class RAGAssistant:
    def __init__(self, api_key=None):
        self.vector_store_manager = VectorStoreManager()
        self.vector_store_manager.initialize_vector_store()
        
        # Use provided API key or fall back to config
        self.api_key = api_key or GROQ_API_KEY
        self.llm = None
        self.qa_chain = None
        
        # Initialize if API key is available
        if self.api_key and self.api_key.strip():
            self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM with the API key"""
        try:
            # Set the API key in environment
            os.environ["GROQ_API_KEY"] = self.api_key
            
            # Test the API key by making a simple call
            self.llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name=GROQ_MODEL,
                temperature=0.1
            )
            
            # Test the connection with a simple prompt
            test_response = self.llm.invoke("Hello")
            print("Groq API connection test successful")
            
            # Create the QA chain
            self.qa_chain = self._create_qa_chain()
            print("RAG assistant initialized successfully")
            
        except Exception as e:
            print(f"Error initializing Groq client: {str(e)}")
            self.llm = None
            self.qa_chain = None
            # Remove the invalid API key from environment
            if "GROQ_API_KEY" in os.environ:
                del os.environ["GROQ_API_KEY"]
    
    def _create_qa_chain(self):
        """Create the RAG QA chain with custom prompt"""
        if not self.llm:
            return None
            
        # Check if vector store is available
        if not self.vector_store_manager.vector_store:
            print("Warning: Vector store not available")
            return None
            
        prompt_template = """You are MarketMuse, an AI content strategist assistant. 
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Question: {question}
        
        Provide a comprehensive answer that includes:
        1. Key insights from the context
        2. Content strategy recommendations
        3. Potential subtopics to explore
        4. SEO considerations if relevant
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        chain_type_kwargs = {"prompt": PROMPT}
        
        try:
            return RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store_manager.vector_store.as_retriever(
                    search_kwargs={"k": 3}
                ),
                chain_type_kwargs=chain_type_kwargs,
                return_source_documents=True
            )
        except Exception as e:
            print(f"Error creating QA chain: {str(e)}")
            return None
    
    def is_initialized(self):
        """Check if the RAG assistant is properly initialized"""
        return self.qa_chain is not None and self.llm is not None
    
    def query(self, question: str):
        """Query the RAG system"""
        if not self.is_initialized():
            return {"result": "RAG assistant not initialized. Please check your API key and try again.", "source_documents": []}
        
        try:
            return self.qa_chain({"query": question})
        except Exception as e:
            return {"result": f"Error querying the system: {str(e)}", "source_documents": []}
    
    def add_documents(self, file_paths: list):
        """Add documents to the knowledge base"""
        return self.vector_store_manager.add_documents(file_paths)
    
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
        
        # Try to get more specific error
        try:
            # Test the API key directly
            import requests
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json={
                    "model": GROQ_MODEL,
                    "messages": [{"role": "user", "content": "test"}],
                    "temperature": 0.1
                },
                timeout=10
            )
            
            if response.status_code == 401:
                return "Invalid API key - authentication failed"
            elif response.status_code == 403:
                return "API key doesn't have permission to access this model"
            elif response.status_code >= 400:
                return f"API error: {response.status_code} - {response.text}"
            else:
                return "Unknown initialization error - check API key validity"
                
        except Exception as e:
            return f"Error testing API key: {str(e)}"