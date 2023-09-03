from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import pandas as pd

def process_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def process_text(text):
    return text

def process_csv(csv):
    df = pd.read_csv(csv)
    text = "\n".join(df[df.columns[0]].astype(str))
    return text

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your Document")
    st.header("Ask your Document 💬")
    
    # upload file
    uploaded_file = st.sidebar.file_uploader("Upload your file", type=["pdf", "txt", "csv"])
    
    # extract the text
    if uploaded_file is not None:
        file_type = uploaded_file.type
        
        if file_type == "application/pdf":
            text = process_pdf(uploaded_file)
        elif file_type == "text/plain":
            text = process_text(uploaded_file.getvalue().decode())
        elif file_type == "text/csv":
            text = process_csv(uploaded_file)
        else:
            st.error("Unsupported file type")
            return
        
        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # show user input
        user_question = st.text_input("Ask a question about the document:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
            
            st.write(response)

if __name__ == '__main__':
    main()
