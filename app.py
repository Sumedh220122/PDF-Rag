import streamlit as st
import os
import tempfile
import shutil
from pdf_rag import PDFQuestionAnswering

# Set page configuration
st.set_page_config(
    page_title="PDF Question Answering",
    page_icon="📚",
    layout="centered"
)

# Add custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .upload-text {
        text-align: center;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'pdf_uploaded' not in st.session_state:
    st.session_state.pdf_uploaded = False
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = None

def clear_vector_store():
    """Clear the Chroma vector store."""
    try:
        if os.path.exists(".chroma"):
            shutil.rmtree(".chroma")
    except Exception as e:
        st.error(f"Error clearing vector store: {str(e)}")

def process_pdf(uploaded_file):
    """Process the uploaded PDF file."""
    # Clear previous vector store
    clear_vector_store()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    qa_system = PDFQuestionAnswering(tmp_file_path)
    qa_system.load_and_process_pdf()
    
    return qa_system, tmp_file_path

# Main app layout
st.title("📚 PDF Question Answering")
st.markdown("---")

# File upload section
if not st.session_state.pdf_uploaded:
    uploaded_file = st.file_uploader(
        "Upload your PDF document",
        type=['pdf'],
        help="Upload a PDF file to get started"
    )

    if uploaded_file:
        with st.spinner("Processing PDF... This may take a minute..."):
            try:
                qa_system, tmp_file_path = process_pdf(uploaded_file)
                st.session_state.qa_system = qa_system
                st.session_state.pdf_uploaded = True
                st.session_state.tmp_file_path = tmp_file_path
                st.session_state.pdf_name = uploaded_file.name
                st.success(f"Successfully uploaded and processed: {uploaded_file.name}")
                st.rerun()  # Rerun to update the UI
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

# Question input section
if st.session_state.pdf_uploaded:
    st.info(f"Currently loaded PDF: {st.session_state.pdf_name}")
    st.markdown("### Ask Questions")
    question = st.text_input(
        "Enter your question about the PDF:",
        placeholder="e.g., What are the main topics discussed in the document?"
    )
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        if st.button("Ask", type="primary", key="ask_button"):
            if question:
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.qa_system.answer_question(question)
                        st.markdown("### Answer")
                        st.markdown(result["answer"])
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
            else:
                st.warning("Please enter a question.")
    
    with col2:
        if st.button("New PDF", key="new_pdf_button"):
            # Clean up temporary file
            if hasattr(st.session_state, 'tmp_file_path'):
                try:
                    os.unlink(st.session_state.tmp_file_path)
                except:
                    pass
            
            # Clear vector store
            clear_vector_store()
            
            # Reset session state
            st.session_state.qa_system = None
            st.session_state.pdf_uploaded = False
            st.session_state.pdf_name = None
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with Streamlit, LangChain, and Google's Gemini</p>
    </div>
    """,
    unsafe_allow_html=True) 