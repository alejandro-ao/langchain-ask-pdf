from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


def upload_pdf():
    st.header("Ask your PDF 💬")
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    return pdf

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

def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base

def ask_question(knowledge_base):
    user_question = st.text_input("Ask a question about your PDF:")
    if user_question:
        docs = knowledge_base.similarity_search(user_question)
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            print(cb)
        st.write(response)

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    pdf = upload_pdf()
    if pdf:
        text = extract_text_from_pdf(pdf)
        if text:
            chunks = get_text_chunks(text)
            knowledge_base = create_embeddings(chunks)
            ask_question(knowledge_base)



if __name__ == '__main__':
    main()