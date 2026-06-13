from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# Load PDF
loader = PyPDFLoader("notes.pdf")
documents = loader.load()

print(f"Pages Loaded: {len(documents)}")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

print(f"Chunks Created: {len(docs)}")

# Create embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS vector database
vectorstore = FAISS.from_documents(
    docs,
    embedding
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0
)

print("\nRAG Chatbot Ready!")
print("Type 'exit' to quit.\n")

# Chat loop
while True:
    query = input("Ask Question: ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(query)

    context = "\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

prompt = f"""
You are a RAG chatbot.

Answer ONLY from the provided context.

If the answer is not explicitly present in the context, reply exactly:

Not found in document

Do not use your own knowledge.

Context:
{context}

Question:
{query}
"""

response = llm.invoke(prompt)

print("\nAnswer:")
print(response.content)
print("-" * 60)