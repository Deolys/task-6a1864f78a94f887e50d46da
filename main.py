import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentExecutor, AgentType
from tavily import TavilyClient

# Load environment variables
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not set in .env")

# Embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Qdrant client (assumes local server running on default port 6333)
qdrant_client = QdrantClient(host="localhost", port=6333)
collection_name = "rag_agent_collection"
vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection_name,
    embeddings=embeddings,
)

# Load documents from ./documents/

def load_documents(directory: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for file_path in Path(directory).glob("**/*.*"):
        if file_path.suffix.lower() not in {".txt", ".md"}:
            continue
        content = file_path.read_text(encoding="utf-8")
        splits = text_splitter.split_text(content)
        docs.extend([{"page_content": s, "metadata": {"source": str(file_path)}} for s in splits])
    return docs

# Index documents if collection empty
if not vectorstore.client.count(collection_name=collection_name).count:
    docs = load_documents("documents")
    vectorstore.add_texts([d["page_content"] for d in docs], [d["metadata"] for d in docs])

# Tools
@tool(name="search_local_kb", description="Semantic search in local knowledge base using Qdrant. Returns top_k results.")
def search_local_kb(query: str, top_k: int = 3) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(query)
    return "\n---\n".join([f"{d.page_content[:500]}... (source: {d.metadata.get('source')})" for d in docs])

@tool(name="web_search", description="Search the web using Tavily API.")
def web_search(query: str) -> str:
    client = TavilyClient(api_key=TAVILY_API_KEY)
    results = client.search(query, max_results=3)
    return "\n---\n".join([f"{r.title}\n{r.url}\n{r.content[:500]}..." for r in results])

# Agent setup
tools = [search_local_kb, web_search]
llm = ChatOllama(model="llama3")
agent_executor = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# CLI loop
print("RAG Agent ready. Type 'exit' to quit.")
while True:
    user_input = input("\nQuery: ")
    if user_input.lower() in {"exit", "quit"}:
        break
    response = agent_executor.run(user_input)
    print(f"\nAnswer:\n{response}")
