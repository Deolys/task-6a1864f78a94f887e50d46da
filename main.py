"""
RAG Agent with ChromaDB and Tavily web search.
Author: Student (task 6a1864f78a94f887e50d46da)
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (TAVILY_API_KEY)
load_dotenv()

# --- Vector Store Setup -----------------------------------------------------
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "nomic-embed-text"
CHROMA_DIR = Path("./chroma_db")
DOCS_DIR = Path("./documents")

# Create or load vector store
vectorstore = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
)

# Load documents if collection is empty
if not list(vectorstore.get_collection().list_documents()):
    # Read all .txt and .md files
    texts = []
    for file_path in DOCS_DIR.rglob("*.txt"):
        texts.append(file_path.read_text(encoding="utf-8"))
    for file_path in DOCS_DIR.rglob("*.md"):
        texts.append(file_path.read_text(encoding="utf-8"))

    if texts:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents([{"page_content": t} for t in texts])
        vectorstore.add_documents(docs)
        # Persist the collection
        vectorstore.persist()

# Retriever for local knowledge base
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- Tools ---------------------------------------------------------------
from langchain.tools import Tool
from langchain_community.utilities.tavily_search import TavilySearchResults

# Local KB search tool
search_local_kb_tool = Tool(
    name="search_local_kb",
    func=lambda query, top_k=3: "\n".join([f"{i+1}. {doc.metadata.get('source', 'unknown')} – {doc.page_content[:200]}..."
                                            for i, doc in enumerate(retriever.invoke(query))]),
    description="Search the local knowledge base using ChromaDB. Provide a query and optional top_k (default 3). Returns formatted snippets.",
)

# Web search tool via Tavily
search_web_tool = Tool(
    name="web_search",
    func=lambda query: "\n".join([f"{i+1}. {res['title']} – {res['url']}"
                                   for i, res in enumerate(TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY")).run(query))],
    description="Perform a web search using Tavily. Provide a query string.",
)

# --- Agent ---------------------------------------------------------------
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOllama

chat = ChatOllama(model="llama3")
agent_executor = initialize_agent(
    tools=[search_local_kb_tool, search_web_tool],
    llm=chat,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# --- CLI ---------------------------------------------------------------
print("RAG Agent ready. Type 'exit' to quit.")
while True:
    user_input = input("\nQuery: ")
    if user_input.strip().lower() == "exit":
        print("Goodbye!")
        break
    try:
        response = agent_executor.run(user_input)
        print(f"\nAnswer:\n{response}")
    except Exception as e:
        print(f"Error: {e}")
