# RAG Agent with ChromaDB and Tavily

## Overview
This project implements a Retrieval-Augmented Generation (RAG) agent that can search both a local knowledge base stored in **ChromaDB** and the web via **Tavily**. The agent decides which source to use based on the query.

## Prerequisites
- Python 3.10+
- Ollama installed locally with models:
  ```bash
  ollama pull llama3
  ollama pull nomic-embed-text
  ```
- A Tavily API key (set in a `.env` file).

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Place your documents (`*.txt`, `*.md`) into the `documents/` directory.
2. Run the agent:
   ```bash
   python main.py
   ```
3. Type queries; type `exit` to quit.

The output will include the answer and indicate whether it came from **chromadb** or **tavily**.
