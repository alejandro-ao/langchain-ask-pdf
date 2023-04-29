import os
from dotenv import load_dotenv


def main():
    # get api key
    load_dotenv()
    openai_key = os.environ['OPENAI_KEY']
    print(openai_key)
    # initialize pdf reader
    # get raw text from pdf
    # split text into chunks for token limit
    # get embeddings from openai
    # perform docsearch
    # ask question to llm using langchain


if __name__ == '__main__':
    main()
