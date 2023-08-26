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
from typing import Optional

persist_directory = os.environ.get('PERSIST_DIRECTORY')



class PDFEmbeddings():
    def __init__(self):
        pass

    def semantic_search(self, num_queries):
        unique_chunks = set()
        queries = similarity_search_queries[:num_queries] # fix this

        
        for query in queries:
            results = st.session_state.knowledge_base.similarity_search(query, k=2)
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
    pdf_embed = PDFEmbeddings()
    # pdf_embed.process_documents() # This takes a while, so we only do it once, this does the embedding
    result = pdf_embed.extract_queries_from_documents(num_similarity_search_queries=5)
    print("type of result: ", type(result))
    for i in result:
        print(i)
    

    ################ USE CASE 2: TLDR Summarize ################
    # tldr = TLDR()
    # tldr.load_text(hardcoded_data.articles)
    # results = tldr.summarize(target= "general") # target can be "general" or "scientific"
    # for result in results:
    #     print("\n\ncategory: ", result['category'])
    #     for article in result['articles']:
    #         print("\n\nTitle: ", article['title'])
    #         print("\n")
    #         print("Tldr: ", article['tldr'])