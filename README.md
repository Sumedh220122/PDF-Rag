# PDF Question Answering with RAG

This application uses Retrieval Augmented Generation (RAG) to answer questions about PDF documents using LangChain and Google's Gemini model.

## Features

- PDF document processing and chunking
- Semantic search using HuggingFace embeddings
- Question answering using Google's Gemini model
- Source attribution for answers

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```
4. Add your Google API key to the `.env` file:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

1. Place your PDF file in the project directory
2. Update the `pdf_path` variable in `main()` function of `pdf_rag.py` to point to your PDF file
3. Run the application:
   ```bash
   python pdf_rag.py
   ```
4. Enter your questions when prompted
5. Type 'quit' to exit the application

## How it Works

1. The PDF is loaded and split into smaller chunks
2. Text chunks are converted into embeddings using HuggingFace's sentence transformers
3. Embeddings are stored in a Chroma vector store
4. When a question is asked:
   - The question is converted to an embedding
   - Similar chunks are retrieved from the vector store
   - Retrieved chunks are sent to Gemini along with the question
   - Gemini generates an answer based on the provided context

## Requirements

- Python 3.8+
- Google API key with access to Gemini
- Internet connection for API calls

## Dependencies

- langchain
- langchain-google-genai
- google-generativeai
- python-dotenv
- pypdf
- chromadb
- sentence-transformers 
