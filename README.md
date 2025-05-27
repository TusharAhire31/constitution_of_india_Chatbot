# 🇮🇳 Constitution of India Chatbot

This project is an AI-powered chatbot built with **LangChain**, **Streamlit**, and **FAISS**, designed to answer questions about the **Constitution of India** using a structured JSON knowledge base.

---

## Features

- Ask questions about any Article or Part of the Indian Constitution
- Uses vector search (FAISS) for fast and accurate retrieval
- Built with Google Generative AI embeddings + LLaMA model via Groq
- Context-aware chat with memory history
- Clean Streamlit user interface

---

## Tech Stack

- **Python 3.9+**
- **Streamlit** for frontend
- **LangChain** for chaining and memory
- **FAISS** for vector database
- **Google Generative AI** for embeddings
- **Groq API** for LLaMA LLM backend

---
## Folder Structure
constitution-chatbot/
├── app.py # Streamlit chatbot frontend
├── ingestion.py # JSON embedding and vector store builder
├── constitution_articles.json # Knowledge base (Constitution data)
├── my_vector_store/ # FAISS vector database
├── requirements.txt
├── .env # API keys
└── README.md

