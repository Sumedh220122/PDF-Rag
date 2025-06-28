import os
from typing import List
from dotenv import load_dotenv
import pdfplumber
import warnings
import logging
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Configure logging and warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables
load_dotenv()

class PDFQuestionAnswering:
    def __init__(self, pdf_path: str, api_key: str):
        """Initialize the PDF Question Answering system.
        
        Args:
            pdf_path (str): Path to the PDF file
            api_key (str): Google API key for Gemini model
        """
        self.pdf_path = pdf_path
        self.llm = GoogleGenerativeAI(model="gemini-2.5-flash", 
                                    google_api_key=api_key)
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store = None
        
    def load_pdf(self) -> List[Document]:
        """Load PDF and convert to documents."""
        documents = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    doc = Document(
                        page_content=text,
                        metadata={"source": self.pdf_path, "page": page_num + 1}
                    )
                    documents.append(doc)
        return documents
        
    def load_and_process_pdf(self):
        """Load and process the PDF document."""
        # Load PDF
        documents = self.load_pdf()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)
        
        # Create vector store using FAISS
        self.vector_store = FAISS.from_documents(
            documents=texts,
            embedding=self.embeddings
        )
        
    def setup_qa_chain(self):
        """Set up the question-answering chain."""
        # Create prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create chain
        chain_type_kwargs = {"prompt": PROMPT}
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
        )
    
    def answer_question(self, question: str) -> dict:
        """Answer a question about the PDF content.
        
        Args:
            question (str): The question to answer
            
        Returns:
            dict: Contains the answer and source documents
        """
        if not self.vector_store:
            self.load_and_process_pdf()
        
        qa_chain = self.setup_qa_chain()
        response = qa_chain({"query": question})
        
        return {
            "answer": response["result"],
            "source_documents": response["source_documents"]
        }
        
    def cleanup(self):
        """Clean up resources between PDF changes."""
        # Clear the vector store
        if self.vector_store is not None:
            self.vector_store = None
            
        # Force garbage collection to free up memory
        import gc
        gc.collect()

def main():
    # Example usage
    pdf_path = "Sumedh_resume.pdf"  # Replace with your PDF file path
    qa_system = PDFQuestionAnswering(pdf_path, os.getenv("GOOGLE_API_KEY"))
    
    try:
        while True:
            question = input("\nEnter your question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
                
            try:
                result = qa_system.answer_question(question)
                print("\nAnswer:", result["answer"])
                print("\nSources:")
                for i, doc in enumerate(result["source_documents"], 1):
                    print(f"\nSource {i}:")
                    print(f"Page: {doc.metadata.get('page', 'Unknown')}")
                    print(f"Content: {doc.page_content[:200]}...")
            except Exception as e:
                print(f"An error occurred: {str(e)}")
    finally:
        qa_system.cleanup()

if __name__ == "__main__":
    main() 
