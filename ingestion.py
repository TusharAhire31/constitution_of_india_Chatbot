import os
import json

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def embed_and_save_documents():
    with open("constitution_articles.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for article in data:
        text = article.get("text")
        if not text:
            print(f"Skipping article: {article.get('article_number', 'Unknown')} due to missing text.")
            continue

        metadata = {
            "article_number": article.get("article_number", "Unknown"),
            "title": article.get("title", "")
        }

        documents.append(Document(page_content=text, metadata=metadata))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    vectorstore.save_local("my_vector_store")
    print("Vector store saved successfully.")


if __name__ == "__main__":
    embed_and_save_documents()
