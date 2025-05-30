import os
os.environ["HOME"] = os.getcwd()  

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

os.environ["STREAMLIT_HOME"] = os.getcwd()


# Load .env file and Hugging Face API token
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- Prompt Template ---
prompt_template = """
<s>[INST]
You are a helpful and precise legal assistant specializing in the Constitution of India. Your goal is to provide concise, accurate, and context-aware answers based only on the articles provided.

Rules:
- Answer **only** using the provided context.
- If you can't find an answer in the context, say: "I don't have enough information to answer that based on the Constitution."
- Do **not** generate fabricated content.
- Do not repeat the question or chat history unless needed for clarity.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION:
{question}

ANSWER:
[/INST]
"""

# Load vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("my_vector_store", embeddings, allow_dangerous_deserialization=True)

# Use HuggingFaceEndpoint instead of deprecated HuggingFaceHub
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",
    temperature=0.0,
    max_length=512,
    huggingfacehub_api_token=hf_token
)

# Prepare QA chain
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question", "chat_history"]
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# --- Streamlit UI ---
st.set_page_config(page_title="Constitution of India Chatbot", page_icon="🇮🇳")
st.title("🇮🇳 Constitution of India Chatbot")
st.markdown("Ask any question based on the Constitution of India. The assistant will only answer based on the official articles.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Your question:")

if query:
    result = qa_chain({
        "question": query,
        "chat_history": st.session_state.chat_history
    })

    answer = result["answer"]
    st.session_state.chat_history.append((query, answer))
    st.markdown(f"**Answer:** {answer}")

    with st.expander("Context source (for transparency)"):
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Source {i+1}:** {doc.metadata.get('article_number', 'N/A')} - {doc.page_content[:300]}...")
