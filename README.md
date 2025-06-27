# PDF Question Answering System

A streamlined application that uses RAG (Retrieval Augmented Generation) to answer questions about PDF documents. Built with Streamlit, LangChain, and Google's Gemini model.

## Features

- 📄 PDF Document Processing
  - Upload and process PDF files
  - Automatic text extraction and chunking
  - Clean context management (fresh context for each new PDF)

- 🔍 Intelligent Question Answering
  - Powered by Google's Gemini model
  - Context-aware responses
  - Real-time processing

- 🎯 User-Friendly Interface
  - Simple drag-and-drop PDF upload
  - Clear success/error messages
  - Currently loaded PDF indicator
  - Easy document switching

## Prerequisites

- Python 3.8+
- Google API key for Gemini
- Internet connection for API calls

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd PDF_RAG
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On Unix or MacOS
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Use the interface:
   - Upload a PDF using the file uploader
   - Wait for the processing confirmation
   - Enter your questions about the document
   - Click "Ask" to get answers
   - Use "New PDF" to switch documents

## Technical Details

The application uses:
- LangChain for document processing and RAG implementation
- Google's Gemini model for question answering
- Chroma as the vector store for document embeddings
- Streamlit for the user interface

## File Structure

```
PDF_RAG/
├── app.py              # Streamlit interface
├── pdf_rag.py         # Core RAG implementation
├── requirements.txt   # Project dependencies
├── .env              # Environment variables (create this)
└── README.md         # This file
```

## Important Notes

- The system automatically clears previous context when loading a new PDF
- Maximum PDF file size: 200MB
- Supports text-based PDFs (scanned documents may not work optimally)
- Requires an active internet connection for Gemini API calls

## Error Handling

The application includes error handling for:
- PDF processing issues
- API connection problems
- Invalid questions
- File management errors

## Contributing

Feel free to submit issues and enhancement requests! 
