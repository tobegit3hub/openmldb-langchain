#!/usr/bin/env python3

import os
import fnmatch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader




def list_files(path, suffix):
    # Get all files and directories in the path
    files = os.listdir(path)

    # Iterate over files and directories
    for file in files:
        # Construct full path of file or directory
        full_path = os.path.join(path, file)

        # If it's a directory, recursively list Markdown files in that directory
        if os.path.isdir(full_path):
            yield from list_markdown_files(full_path)
        # If it's a Markdown file, yield its path
        elif fnmatch.fnmatch(file, f"*.{suffix}"):
            yield full_path


def list_markdown_files(path):
    return list_files(path, "md")


def list_pdf_files(path):
    return list_files(path, "pdf")


def load_docs_to_faiss():
    markdown_file_paths = list_markdown_files("./docs/zh/")
    faiss_file_path = "./openmldb_docs_index/"

    docs = []
    for file_path in markdown_file_paths:
        print(f"Try to convert {file_path} to faiss index")
        loader = UnstructuredMarkdownLoader(file_path, mode="elements")
        doc = loader.load()
        docs.extend(doc)

    # TODO: split markdown or not

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    db.save_local(faiss_file_path)


def load_pdf_to_faiss():
    pdf_file_paths = list_pdf_files("./paper/")
    faiss_file_path = "./openmldb_pdf_index"

    docs = []
    for file_path in pdf_file_paths:
        print(f"Try to convert {file_path} to faiss index")
        loader = PyPDFLoader(file_path)
        doc = loader.load_and_split()
        docs.extend(doc)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    db.save_local(faiss_file_path)


def main():
    #load_docs_to_faiss()
    load_pdf_to_faiss()


if __name__ == "__main__":
    main()