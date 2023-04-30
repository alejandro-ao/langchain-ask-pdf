# pip install langchain openai pypdf2 faiss-cpu python-dotenv tiktoken

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import streamlit as st


def get_text_chunks_from_pdf(pdf_path):
    # initialize pdf reader
    pdf = PdfReader('gpt4-in-medicine.pdf')

    # get raw text from pdf
    raw_text = ""
    for page in pdf.pages:
        raw_text += page.extract_text()
    
    # split text into chunks for token limit
    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )
    return text_splitter.split_text(raw_text)


def setup_text_search(text_chunks):
    # get embeddings from openai
    embeddings = OpenAIEmbeddings()
    
    # perform docsearch
    docsearch = FAISS.from_texts(text_chunks, embeddings)
    chain = load_qa_chain(OpenAI(), chain_type='stuff')
    
    return docsearch, chain
    
    
def answer_question(question, docsearch, chain):
    # ask question to llm using langchain
    docs = docsearch.similarity_search(question)
    response = chain.run(input_documents=docs, question=question)
    return response


def app():
    # get api key
    load_dotenv()
    st.set_page_config(page_title="Chat with your PDFs", page_icon="🧊", layout="wide")
    st.header("Chat with your PDF")
    
    pdf = st.file_uploader("Upload a PDF", type="pdf")
    
    if pdf is not None:
      text_chunks = get_text_chunks_from_pdf('gpt4-in-medicine.pdf')
      docsearch, chain = setup_text_search(text_chunks)

      user_question = st.text_input("Ask a question")
      if user_question:
        response = answer_question(user_question, docsearch, chain)
        st.write(response)


if __name__ == '__main__':
    app()