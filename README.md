# MarketMuse - RAG-powered Content Strategy Assistant

A Streamlit-based application that uses RAG (Retrieval Augmented Generation) to assist with content strategy using Groq's LLMs.

## Features

- Document ingestion (PDF, TXT, DOCX, PPTX, HTML)
- Vector store using ChromaDB
- RAG-powered question answering
- Source attribution
- Streamlit UI

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env` file:
4. 4. Run the application: `streamlit run app.py`

## Usage

1. Upload content strategy documents
2. Process them using the sidebar
3. Ask questions about content strategy
4. Get AI-powered answers with source attribution

## Technologies Used

- Streamlit
- LangChain
- ChromaDB
- Groq API
- HuggingFace Embeddings
