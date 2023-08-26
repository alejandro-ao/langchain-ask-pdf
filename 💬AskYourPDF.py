import os
from dotenv import load_dotenv
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

persist_directory = os.environ.get('PERSIST_DIRECTORY')


# def upload_pdf():
#     st.header("Ask your PDF 💬")
#     pdf = st.file_uploader("Upload your PDF", type="pdf")
#     return pdf

def extract_text_from_pdf(pdf):
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    return None

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore directory exists and is not empty
    """
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        return True
    return False


def create_embeddings(chunks, embeddings):
    knowledge_base = Chroma.from_texts(chunks, embeddings)
    return knowledge_base

def create_persistent_embeddings(chunks, embeddings):
    # Check if vectorstore exists
    if does_vectorstore_exist(persist_directory):
        print(f"Embeddings already exist at {persist_directory}. No need to add anything.")
        knowledge_base = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        # Create and store locally vectorstore using Chroma
        print("Creating new vectorstore")
        knowledge_base = Chroma.from_texts(texts=chunks, embedding=embeddings, persist_directory=persist_directory)
        knowledge_base.persist()
        print(f"Ingestion complete! Embeddings created for uploaded file.")
    return knowledge_base


def ask_question(knowledge_base):
    col1, col2 = st.columns([10, 1])  # Creating two columns with different widths
    col2.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)  # Add a css white space to allign the button
    user_question = col1.text_input("Ask a question about your PDF:")
    submit_button = col2.button("Ask")

    if submit_button and user_question:  # Check if the button is clicked and there's a question
        docs = knowledge_base.similarity_search(user_question)
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
        st.write(response)
        formatted_cost = f"${cb.total_cost:.3f}" 
        st.markdown(f"**Callback Result:**\n\n- **Total Cost:** {formatted_cost}\n- **Prompt Tokens:** {cb.prompt_tokens}")


def upload_pdf():
    st.header("Ask your PDF 💬")

    # If a PDF exists in the session state, display its name and a "Remove PDF" button
    if 'uploaded_pdf' in st.session_state:
        col1, col2 = st.columns([10, 4])  # Creating two columns with different widths
        # col2.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)  # Add a CSS white space to align the button
        col1.write(f"Uploaded file: {st.session_state.uploaded_pdf.name}")

        if col2.button('Remove PDF'):
            del st.session_state.uploaded_pdf
            return None

        return st.session_state.uploaded_pdf
    
    # Otherwise, display the file uploader
    else:
        uploaded_pdf = st.file_uploader("Upload your PDF", type="pdf")

        # If the user uploads a new PDF
        if uploaded_pdf:
            st.session_state.uploaded_pdf = uploaded_pdf
            return uploaded_pdf

        return None


def main():
    load_dotenv()
    embeddings = OpenAIEmbeddings()
    st.set_page_config(
        page_title="Ask your PDF",
        # page_icon="📄",
    )
    
    pdf = upload_pdf()
    
    if pdf:
        text = extract_text_from_pdf(pdf)
        
        if text:
            chunks = get_text_chunks(text)
            
            # Initialize st.session_state if not already
            if not hasattr(st.session_state, "knowledge_base"):
                st.session_state.knowledge_base = None
                
            # Check if embeddings are already in the session state
            if not st.session_state.knowledge_base:
                st.session_state.knowledge_base = create_persistent_embeddings(chunks, embeddings)
                
            ask_question(st.session_state.knowledge_base)


if __name__ == '__main__':
    main()