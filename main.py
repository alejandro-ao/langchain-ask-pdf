# pip install langchain openai pypdf2 faiss-cpu python-dotenv tiktoken

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


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


def main():
    # get api key
    load_dotenv()
    
    text_chunks = get_text_chunks_from_pdf('gpt4-in-medicine.pdf')
    
    docsearch, chain = setup_text_search(text_chunks)
    
    response = answer_question("What is this document about?", docsearch, chain)
    
    print(response)


if __name__ == '__main__':
    main()
