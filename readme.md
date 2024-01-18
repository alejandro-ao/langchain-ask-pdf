# Langchain Ask PDF (Tutorial)

## Introduction
Langchain Ask PDF is a Python application that enables users to load a PDF and ask questions about its content using natural language. The application leverages a Large Language Model (LLM) to generate responses specifically related to the content of the loaded PDF. The LLM will intelligently respond to questions only within the context of the document.

## Tutorial
For a comprehensive step-by-step guide on building and understanding this application, refer to the [video tutorial on YouTube](tutorial_link). The tutorial provides detailed insights into the implementation and functionality of the Langchain Ask PDF application.

## How It Works
1. The application reads the PDF document and intelligently splits the text into smaller, semantically meaningful chunks.
2. OpenAI embeddings are employed to create vector representations of these text chunks.
3. The application identifies chunks that are semantically similar to the user's question.
4. The selected chunks are then fed into the LLM to generate a contextually relevant response.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/langchain-ask-pdf.git
   cd langchain-ask-pdf
   ```
2. Install the necessary dependacies:
   ```
   pip install -r requirements.txt
   ```
3. Add your OpenAI API key to the .env file.
4. Usage
  Run the application using the Streamlit command-line interface:
  ```
  streamlit run app.py
  ```

## Contributing

This repository is intended for educational purposes and serves as supplementary material for the YouTube tutorial. It is not designed to receive further contributions. Feel free to utilize it for learning and experimentation.

For any questions or clarifications related to the tutorial, please refer to the associated YouTube video or raise an issue in this repository.


