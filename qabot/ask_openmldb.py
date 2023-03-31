#!/usr/bin/env python3

import sys
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI

    
def load_faiss():
    faiss_file_path = "./openmldb_docs_index"
    #faiss_file_path = "./openmldb_pdf_index"
    embeddings = OpenAIEmbeddings()

    vector_store = FAISS.load_local(faiss_file_path, embeddings)
    return vector_store


def markdown_to_html(file_path) -> str:
    return file_path[:-2] + "html"


def file_path_to_web_url(file_path) -> str:
    # "./docs/zh/integration/offline_data_sources/hive.html"
    # "https://openmldb.ai/docs/zh/main/integration/offline_data_sources/hive.html"
    return "https://openmldb.ai/docs/zh/main/" + file_path[10:]


def query_docs(query):

    db = load_faiss()

    #number_of_docs = 2
    number_of_docs = 4
    related_docs = db.similarity_search(query, k=number_of_docs)

    llm = OpenAI(temperature=0)

    chain = load_qa_with_sources_chain(llm, chain_type="stuff")
    result_string = chain({"input_documents": related_docs, "question": query}, return_only_outputs=True)

    results = result_string["output_text"].split("SOURCES:")

    reply = results[0].strip()

    sources_string = results[1].strip()

    sources = []
    for source_string in sources_string.split(","):
        sources.append(file_path_to_web_url(markdown_to_html(source_string.strip())))
        #sources.append(source_string)
    
    print("Reply: ", reply)
    print("Source: ", sources)


def main():
    #query = "How use OpenMLDB with Hive?"
    #query = "如何使用OpenMLDB集成HDFS?"
    query_docs(sys.argv[1])


if __name__ == "__main__":
    main()
