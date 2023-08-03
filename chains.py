from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import AzureOpenAI, OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
# from langchain.chains.summarize import load_summarize_chain 

from langchain.chat_models import AzureChatOpenAI


import os
import openai
os.environ['CWD'] = os.getcwd()

# for testing
# import src.constants as constants
import constants 
os.environ['OPENAI_API_KEY'] = constants.AZURE_OPENAI_KEY_FR
os.environ['OPENAI_API_BASE'] = constants.AZURE_OPENAI_ENDPOINT_FR
os.environ['OPENAI_API_VERSION'] = "2023-05-15"
os.environ['OPENAI_API_TYPE'] = "azure"
# openai.api_type = "azure"
# openai.api_base = constants.AZURE_OPENAI_ENDPOINT_FR
# openai.api_version = "2023-05-15"
openai.api_key = constants.OPEN_AI_KEY


import os
from typing import Optional
import hardcoded_data 

class TLDR():
    def __init__(self):
        self.prompt_template_general = constants.prompt_template_general
        self.prompt_template_scientific = constants.prompt_template_scientific
        
        self.llm = AzureChatOpenAI(deployment_name= constants.AZURE_ENGINE_NAME_FR, temperature=0)

    def load_text(self, category_list):
        self.documents = []
        for articles_list in category_list:
            category_name = articles_list['category']
            formatted_list = []
            for article in articles_list['articles']:
                title = article["title"]
                abstract = article["abstract"]
                formatted_article = f"Title: {title}\n\nAbstract: {abstract}"
                formatted_list.append(formatted_article)
            self.documents.append({'category': category_name, 'articles': formatted_list})


    def parse_function(self, input_string):
        # Split the input string into individual entries using the "Title:" pattern
        entries = input_string.split("Title: ")[1:]

        # Initialize an empty list to store the dictionaries
        result_list = []

        # Loop through each entry and extract the title and TLDR
        for entry in entries:
            title, rest = entry.split("\nTldr: ", 1)

            # Find the end of the TLDR by looking for the next "Title:" or reaching the end of the entry
            next_title_index = rest.find("Title: ")
            if next_title_index == -1:
                next_title_index = len(rest)

            # Extract the TLDR
            tldr = rest[:next_title_index].strip()
            entry_dict = {"title": title.strip(), "tldr": tldr}
            result_list.append(entry_dict)

        return result_list



    def summarize(self, target = "general"):

        if target == "general":
            prompt_template = self.prompt_template_general
        elif target == "scientific":
            prompt_template = self.prompt_template_scientific
        else:
            raise ValueError("Error: TLDR.summarize() target must be 'general' or 'scientific'")
        
        prompt = PromptTemplate(template= prompt_template, input_variables= ["field", "context"])

        result = []
        for category in self.documents:
            category_name = category['category']
            joined_articles = "\n\n".join(category['articles'])
            chain = LLMChain(llm=self.llm, prompt=prompt)
            output = chain.run(context=joined_articles, field = category_name)
            result.append({"category": category_name ,"articles": self.parse_function(output)})

        return result




class PDFEmbeddings():
    def __init__(self, path: Optional[str] = None):
        self.path = path or os.path.join(os.environ['CWD'], 'archive')
        self.text_splitter = CharacterTextSplitter(separator="\n", chunk_size=2000, chunk_overlap=200)
        self.embeddings = OpenAIEmbeddings(deployment= constants.AZURE_ENGINE_NAME_US, chunk_size=1,
                                           openai_api_key= constants.AZURE_OPENAI_KEY_US,
                                           openai_api_base= constants.AZURE_OPENAI_ENDPOINT_US,
                                           openai_api_version= "2023-05-15",
                                           openai_api_type= "azure",)
        self.vectorstore = Chroma(persist_directory=constants.persistent_dir, embedding_function=self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_type = "similarity", search_kwags= {"k": 5})
        self.memory = ConversationBufferMemory(memory_key='pdf_memory', return_messages=True)
        self.documents = self.load_documents()  # Load documents during initialization
        # self.process_documents()  # Process documents during initialization (?)

    def load_documents(self):
        # Single responsibility: load the documents
        loader = PyPDFDirectoryLoader(self.path)
        documents = loader.load()
        return documents

    def process_documents(self):
        # Single responsibility: create the embeddings of the document chunks
        chunks = self.text_splitter.split_documents(self.documents)
        self.vectorstore.add_documents(chunks)

    def semantic_search(self, num_queries):
        # Single responsibility: perform a semantic search
        document_sources = set([doc.metadata['source'] for doc in self.documents])
        unique_chunks = set()
        queries = list(constants.similarity_search_queries.values())[:num_queries]

        for source in document_sources:
            for query in queries:
                results = self.vectorstore.similarity_search(query, k=2, filter={'source': source})
                for chunk in results:
                    chunk_str = str(chunk)
                    if chunk_str not in unique_chunks:
                        unique_chunks.add(chunk_str)

        return unique_chunks

    def extract_queries_from_documents(self, num_similarity_search_queries= 3):
        # Perform semantic search
        unique_chunks = self.semantic_search(num_similarity_search_queries)

        output_parser = CommaSeparatedListOutputParser()
        format_instructions = output_parser.get_format_instructions()
        pdf_template = constants.pdf_template
        prompt_template = ChatPromptTemplate.from_template(template= pdf_template)

        chain = LLMChain(
            llm= AzureChatOpenAI(deployment_name= constants.AZURE_ENGINE_NAME_FR),
            prompt=prompt_template,
        )
        # Changed 'context_docs' to 'unique_chunks' as that's what's available in this method
        output = chain.run(context=unique_chunks, format_instructions=format_instructions)

        return output_parser.parse(output)



    def search(self, query: str, chain_type: str = "stuff"):
        chain = RetrievalQA.from_chain_type(llm= AzureChatOpenAI(deployment_name= constants.AZURE_ENGINE_NAME_FR, temperature=0),
                                            retriever= self.retriever, chain_type= chain_type, return_source_documents= True)
        result = chain({"query": query})
        return result

    def conversational_search(self, query: str, chain_type: str = "stuff"):
        chain = ConversationalRetrievalChain.from_llm(llm= AzureChatOpenAI(deployment_name= constants.AZURE_ENGINE_NAME_FR),
                                                      retriever= self.retriever, memory= self.memory, chain_type= chain_type)
        result = chain({"question": query})
        return result['answer']

    def load_and_run_chain(self, query: str, chain_type: str = "stuff"):
        chain = load_qa_chain(llm= AzureChatOpenAI(deployment_name= constants.AZURE_ENGINE_NAME_FR), chain_type= chain_type)
        return chain.run(input_documents = self.retriever, question = query)

if __name__ == '__main__':

    ################ USE CASE 1: PDF EMBEDDINGS ################
    # pdf_embed = PDFEmbeddings()
    # # pdf_embed.process_documents() # This takes a while, so we only do it once, this does the embedding
    # result = pdf_embed.extract_queries_from_documents(num_similarity_search_queries=5)
    # print("type of result: ", type(result))
    # for i in result:
    #     print(i)
    

    ################ USE CASE 2: TLDR Summarize ################
    tldr = TLDR()
    tldr.load_text(hardcoded_data.articles)
    results = tldr.summarize(target= "general") # target can be "general" or "scientific"
    for result in results:
        print("\n\ncategory: ", result['category'])
        for article in result['articles']:
            print("\n\nTitle: ", article['title'])
            print("\n")
            print("Tldr: ", article['tldr'])