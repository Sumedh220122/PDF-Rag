import streamlit as st
import os
import tempfile
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
    .api-management {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
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
if 'api_key' not in st.session_state:
    st.session_state.api_key = None

def process_pdf(uploaded_file):
    """Process the uploaded PDF file."""
    # Clean up previous QA system if it exists
    if st.session_state.qa_system:
        st.session_state.qa_system.cleanup()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    qa_system = PDFQuestionAnswering(tmp_file_path, api_key=st.session_state.api_key)
    qa_system.load_and_process_pdf()
    
    return qa_system, tmp_file_path

# Main app layout
st.title("📚 PDF Question Answering")
st.markdown("---")

# API Key input section
if not st.session_state.api_key:
    st.markdown("""
    ### 🔑 Enter Your Google API Key
    To use this application, you need a Google API key with access to the Gemini model. 
    If you don't have one, you can get it from the [Google AI Studio](https://makersuite.google.com/app/apikey).
    """)
    
    api_key = st.text_input(
        "Enter your Google API Key",
        type="password",
        help="Your API key will be stored securely in the session state"
    )
    
    if st.button("Submit API Key"):
        if api_key:
            st.session_state.api_key = api_key
            st.success("API Key saved successfully!")
            st.rerun()
        else:
            st.error("Please enter an API Key")
    
    st.markdown("---")
    st.markdown("""
    #### 🔒 Security Note
    - Your API key is stored securely in the session state
    - It will be cleared when you close the browser
    - Never share your API key with others
    """)
else:
    # Show API status and management
    st.markdown("""
        <div class="api-management">
            <span>🔑 API Key Status: Active</span>
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        if st.button("Clear API Key", type="secondary"):
            st.session_state.api_key = None
            st.session_state.qa_system = None
            st.session_state.pdf_uploaded = False
            st.session_state.pdf_name = None
            if hasattr(st.session_state, 'tmp_file_path'):
                try:
                    os.unlink(st.session_state.tmp_file_path)
                except:
                    pass
            st.rerun()
    
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
                # Clean up QA system and temporary files
                if st.session_state.qa_system:
                    st.session_state.qa_system.cleanup()
                
                if hasattr(st.session_state, 'tmp_file_path'):
                    try:
                        os.unlink(st.session_state.tmp_file_path)
                    except:
                        pass
                
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
