import os
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentExecutor, AgentType
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from tavily import TavilyClient

# Load environment variables (TAVILY_API_KEY)
load_dotenv()

# ---------- Vector Store Setup ----------

def create_vectorstore(persist_directory: str = "./chroma_db"):
    """Create or load a Chroma vector store with Ollama embeddings."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(embedding_function=embeddings, persist_directory=persist_directory)


def load_documents(directory: str, vectorstore):
    """Read .txt/.md files from directory, chunk them, and add to vector store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for file_path in Path(directory).glob("**/*"):
        if file_path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = file_path.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)
        docs.extend([Document(page_content=c, metadata={"source": str(file_path)}) for c in chunks])
    vectorstore.add_documents(docs)
    vectorstore.persist()

# ---------- Tools ----------
@tool("search_local_kb")
def search_local_kb(query: str, top_k: int = 3):
    """Semantic search in the local ChromaDB knowledge base."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    results = retriever.get_relevant_documents(query)
    return "\n---\n".join([f"{doc.page_content[:200]}... (source: {doc.metadata['source']})" for doc in results])

@tool("web_search")
def web_search(query: str):
    """Web search using Tavily."""
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = client.search(query, max_results=3)
    return "\n---\n".join([f"{item['content'][:200]}... (source: {item['url']})" for item in response])

# ---------- Agent ----------
SYSTEM_PROMPT = (
    "You are an AI assistant that answers user questions.
    If the answer can be found in the local knowledge base, use the search_local_kb tool.
    Otherwise, use web_search to fetch up-to-date information.
    After retrieving the answer, respond with the content and indicate the source (chromadb or tavily)."
)

agent_executor: AgentExecutor | None = None
vectorstore = create_vectorstore()
# Load documents if not already loaded
if len(vectorstore.get_all_documents()) == 0:
    load_documents("documents", vectorstore)

prompt = PromptTemplate.from_template(SYSTEM_PROMPT)

from langchain.agents import Tool, AgentExecutor, initialize_agent

tools = [search_local_kb, web_search]
agent_executor = initialize_agent(
    tools=tools,
    llm=ChatOllama(model="llama3"),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# ---------- CLI ----------
if __name__ == "__main__":
    print("RAG Agent with ChromaDB and Tavily. Type 'exit' to quit.")
    while True:
        user_input = input("\nQuery: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        response = agent_executor.run(user_input)
        print(f"\nAnswer:\n{response}")
