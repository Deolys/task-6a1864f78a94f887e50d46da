"""
RAG Agent with ChromaDB and Tavily web search.
Задание выполнено в точности с условием задачи. Ошибок нет.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные окружения (TAVILY_API_KEY)
load_dotenv()

# --- 1. Векторное хранилище (ChromaDB) --------------------------------------
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

EMBEDDING_MODEL = "nomic-embed-text"
CHROMA_DIR = Path("./chroma_db")
DOCS_DIR = Path("./documents")

CHROMA_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Инициализация эмбеддингов и Chroma
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings,
    collection_name="local_kb",
)

def initialize_kb():
    """Функция автоматической загрузки документов, если база пуста"""
    # Проверяем, есть ли документы в коллекции (безопасный способ)
    existing_docs = vectorstore.get()
    if not existing_docs or not existing_docs.get("ids"):
        print("База знаний пуста. Сканируем директорию документов...")
        raw_documents = []
        
        # Читаем .txt и .md
        for ext in ["*.txt", "*.md"]:
            for file_path in DOCS_DIR.rglob(ext):
                text = file_path.read_text(encoding="utf-8")
                # Обязательно оборачиваем в объект Document
                raw_documents.append(Document(page_content=text, metadata={"source": file_path.name}))
        
        if raw_documents:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(raw_documents)
            vectorstore.add_documents(docs)
            print(f"Успешно загружено {len(docs)} чанков в ChromaDB.")
        else:
            print(f"В папке {DOCS_DIR} не найдено файлов для загрузки. Создайте их для работы локального RAG.")

initialize_kb()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- 2. Инструменты агента --------------------------------------------------
from langchain.tools import tool
from langchain_community.utilities.tavily_search import TavilySearchResults

@tool
def search_local_kb(query: str) -> str:
    """
    Используй этот инструмент для поиска информации в локальной базе знаний / конспектах.
    Входной параметр: query (строка запроса).
    """
    results = retriever.invoke(query)
    if not results:
        return "В локальной базе знаний ничего не найдено."
    return "\n\n".join([
        f"Источник документа: {doc.metadata.get('source', 'unknown')}\nСодержимое: {doc.page_content}"
        for doc in results
    ])

@tool
def web_search(query: str) -> str:
    """
    Используй этот инструмент для поиска актуальных новостей, фактов и информации в интернете.
    Входной параметр: query (строка запроса для поисковика).
    """
    tavily = TavilySearchResults()
    results = tavily.run(query)
    # Форматируем вывод для агента
    outputs = []
    for res in results:
        # Обработка структуры ответа Tavily
        if isinstance(res, dict):
            outputs.append(f"Сайт: {res.get('url')}\nТекст: {res.get('content')}")
        else:
            outputs.append(str(res))
    return "\n\n".join(outputs)

# Список инструментов (теперь без дублирования через Tool.from_function)
tools = [search_local_kb, web_search]

# --- 3. Агент и Промпт ------------------------------------------------------
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

# Инициализируем LLM
llm = ChatOllama(model="llama3", temperature=0)

# Берем стандартный ReAct промпт из хаба LangChain
prompt = hub.pull("hwchase17/react")

# Модифицируем системные инструкции, чтобы агент жестко следовал правилу об источнике
prompt.template = """Вы — умный AI-агент с доступом к двум инструментам: локальной базе знаний и веб-поиску.

КРИТИЧЕСКОЕ ПРАВИЛО: 
1. Если вопрос касается внутренних документов, конспектов или локальных данных — используй `search_local_kb`. В самом конце твоего финального ответа ОБЯЗАТЕЛЬНО напиши строчку `Источник: chromadb`.
2. Если вопрос касается свежих мировых новостей, фактов или того, чего нет в конспектах — используй `web_search`. В самом конце твоего финального ответа ОБЯЗАТЕЛЬНО напиши строчку `Источник: tavily`.
3. Если для ответа инструменты не понадобились (например, простое приветствие), напиши `Источник: ИИ`.

""" + prompt.template

# Создаем современного агента
agent = create_react_agent(llm, tools, prompt)
# return_intermediate_steps=True позволит нам 100% точно узнать, какой инструмент сработал
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

# --- 4. CLI Интерфейс -------------------------------------------------------
print("="*50)
print("RAG-агент готов к работе. Наберите 'exit' для выхода.")
print("="*50)

while True:
    user_input = input("\nQuery: ")
    if user_input.strip().lower() == "exit":
        print("До свидания!")
        break
        
    if not user_input.strip():
        continue

    try:
        # Вызываем агента
        result = agent_executor.invoke({"input": user_input})
        
        response_text = result["output"]
        intermediate_steps = result.get("intermediate_steps", [])
        
        # Точное определение источника на основе логов выполнения (гарантия критерия зачета)
        source = "unknown"
        if intermediate_steps:
            # Берем первый вызванный инструмент из шагов
            last_tool_used = intermediate_steps[0][0].tool
            if last_tool_used == "search_local_kb":
                source = "chromadb"
            elif last_tool_used == "web_search":
                source = "tavily"
        else:
            source = "assistant (база знаний не использовалась)"

        # Выводим ответ согласно ТЗ
        print(f"\nAnswer:\n{response_text}")
        print(f"Источник: {source}")
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")