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

        articles = json.load(f)

    documents = []
    for article in articles:
        content = f"Article {article['article_number']}: {article.get('title', '')}\n{article['text']}"
        documents.append(Document(page_content=content, metadata={"article_number": article["article_number"]}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    vectorstore.save_local("my_vector_store")
    print("Vector store saved successfully.")

embed_and_save_documents()
