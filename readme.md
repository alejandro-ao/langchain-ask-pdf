# Langchain Ask PDF 

This is a Python application that allows you to load a PDF and ask questions about it using natural language. The application uses a LLM to generate a response about your PDF. The LLM will not answer questions unrelated to the document.

## How it works

The application reads the PDF and splits the text into smaller chunks that can be then fed into a LLM. It uses OpenAI embeddings to create vector representations of the chunks. The application then finds the chunks that are semantically similar to the question that the user asked and feeds those chunks to the LLM to generate a response.

The application uses Streamlit to create the GUI and Langchain to deal with the LLM.

## Installation

To install the repository, please clone this repository and install the requirements:

```
pip install -r requirements.txt
```

You will also need to **add your OpenAI API key to the `.env` file**.

## Usage

To use the application, run the `main.py` file with the streamlit CLI (after having installed streamlit): 

```
streamlit run app.py
```

## Contributing

Thank you for considering contributing to our project! Your contributions can make a significant impact, and we appreciate your interest in helping us improve.

### Why Contribute?

Our repository is dedicated to analyzing PDF documents using Language Models (LLMs), and your contributions can help us enhance the capabilities and functionalities of our project. By contributing, you can:

- Improve the accuracy and performance of the LLMs for document analysis.
- Add new features and tools to extend the functionality of the project.
- Fix bugs and errors to ensure a smooth experience for users.
- Enhance the documentation to make it more accessible to the community.
