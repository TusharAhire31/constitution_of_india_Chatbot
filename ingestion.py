import os
import json

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA


def embed_and_save_documents():
    with open("constitution_articles.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = [
        Document(page_content=article["text"], metadata={"article": article["article"]})
        for article in data
    ]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    vectorstore = FAISS.from_documents(split_docs, embeddings)


    vectorstore.save_local("my_vector_store")

if __name__ == "__main__":
    embed_and_save_documents()

    vectorstore.save_local("my_vector_store")
    print("Vector store saved successfully.")

embed_and_save_documents()
