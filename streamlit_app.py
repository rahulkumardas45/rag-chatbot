import os
from dotenv import load_dotenv

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

# Page Title
st.set_page_config(page_title="PDF RAG Chatbot")

st.title("📚 PDF RAG Chatbot")
st.write("Ask questions from your PDF document")

# Load PDF
loader = PyPDFLoader("notes.pdf")
documents = loader.load()

# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

# Embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# FAISS
vectorstore = FAISS.from_documents(
    docs,
    embedding
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0
)

# User Question
query = st.text_input("Ask a question")

if query:

    retrieved_docs = retriever.invoke(query)

    context = "\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    prompt = f"""
Answer ONLY from the provided context.

If the answer is not present in the context,
reply exactly:

Not found in document

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    st.subheader("Answer")
    st.write(response.content)