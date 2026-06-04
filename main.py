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
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "nomic-embed-text"
CHROMA_DIR = Path("./chroma_db")
DOCS_DIR = Path("./documents")

# Ensure directory exists for Chroma data (used by local persistence)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Create or load vector store.  The collection name is "local_kb".
vectorstore = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
    collection_name="local_kb",
)

# Load documents if collection is empty
if not vectorstore.get_collection().list_documents():
    texts = []
    for file_path in DOCS_DIR.rglob("*.txt"):
        texts.append(file_path.read_text(encoding="utf-8"))
    for file_path in DOCS_DIR.rglob("*.md"):
        texts.append(file_path.read_text(encoding="utf-8"))

    if texts:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents([{"page_content": t} for t in texts])
        vectorstore.add_documents(docs)
        # Persist is handled by Chroma automatically

# Retriever for local knowledge base
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- Tools ---------------------------------------------------------------
from langchain.tools import tool
from langchain_community.utilities.tavily_search import TavilySearchResults

@tool("search_local_kb")
def search_local_kb(query: str, top_k: int = 3) -> str:
    """
    Search the local knowledge base using ChromaDB. Provide a query and optional top_k (default 3). Returns formatted snippets.
    """
    results = retriever.invoke(query)
    return "\n".join([
        f"{i+1}. {doc.metadata.get('source', 'unknown')} – {doc.page_content[:200]}..."
        for i, doc in enumerate(results)
    ])

@tool("web_search")
def web_search(query: str) -> str:
    """
    Perform a web search using Tavily. Provide a query string.
    """
    tavily = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))
    results = tavily.run(query)
    return "\n".join([
        f"{i+1}. {res['title']} – {res['url']}"
        for i, res in enumerate(results)
    ])

# --- Agent ---------------------------------------------------------------
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOllama

chat = ChatOllama(model="llama3")
agent_executor = initialize_agent(
    tools=[search_local_kb, web_search],
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
        # Append source info based on tool used in the chain (simplified):
        if "search_local_kb" in response:
            source = "chromadb"
        elif "web_search" in response:
            source = "tavily"
        else:
            source = "unknown"
        print(f"\nAnswer:\n{response}\nSource: {source}")
    except Exception as e:
        print(f"Error: {e}")
