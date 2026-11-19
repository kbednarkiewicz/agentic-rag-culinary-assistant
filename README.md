# Agentic RAG Culinary Assistant
An **Agentic RAG** application designed for Titanic survivors on a desert island that searches documents, autonomously decides which tool to use, and cites sources. Powered by LangGraph, the assistant dynamically chooses between two tools: `search_recipes` and `ingredient_converter`.

- **Grounded Responses:** Strictly answers recipe queries from local PDF documents, citing the exact source filename and page number to prevent hallucinations.
- **Local Vector Store:** Uses `PyMuPDFLoader`, `RecursiveCharacterTextSplitter`, and HuggingFace (`all-MiniLM-L6-v2`) embeddings stored locally in **ChromaDB**.
- **Conversational Memory:** Maintains conversation context across multiple interactions using LangGraph's InMemorySaver and configurable thread_id. Each conversation is associated with a unique thread, allowing the agent to retrieve previous messages and maintain context.  Conversation history is stored in memory and is not persisted after the application restarts.
- **Custom Tools:** Includes recipe retrieval and ingredient measurement conversion tools.
- **REST API:**  Exposes the assistant through a FastAPI backend with Pydantic request validation.
- **Automated Testing:** Includes pytest unit tests to verify deterministic tool behavior.

---

## Project Structure

```text
rag-recipe-assistant/
├── data/
│   └── recipes1.pdf
├── notebooks/
│   └── experiments.ipynb     # Initial prototyping and agent evaluation
├── src/
│   ├── __init__.py
│   ├── config.py             # Paths, model names, chunking and retriever settings
│   ├── ingest.py             # PDF extraction, text chunking, and ChromaDB
│   ├── tools.py              # Custom LangChain tools (search_recipes, ingredient_converter)
│   └── agent.py              # CulinaryAgent class with LangGraph and memory checkpointer
├── tests/
│   └── test_agent.py         # Unit tests for tool functionality
├── app.py                    # FastAPI application
├── pytest.ini
├── requirements.txt
├── .env.example
└── README.md
```

## Tech Stack

* LLM: Google Gemini (gemini-3.1-flash-lite), LangChain, LangGraph

* Vector Database & Embeddings: ChromaDB, HuggingFace sentence-transformers/all-MiniLM-L6-v2

* Document Processing: PyMuPDF (PyMuPDFLoader), RecursiveCharacterTextSplitter

* Backend & Testing: FastAPI, Uvicorn, Pydantic, Pytest

## Getting started

**1. Clone the repository and set up the environment**

```bash
git clone https://github.com/kbednarkiewicz/rag-recipe-assistant.git
cd rag-recipe-assistant
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS:
```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

**2. Configure environment variables**

Create a .env file based on the provided template:

```bash
cp .env.example .env
```

Add your Google Gemini API key to the .env file:

```bash
GOOGLE_API_KEY=your_google_api_key
```

**3. Build the vector database**

Extract text from the PDF documents, generate embeddings, and persist the ChromaDB index locally:
```bash
python -m src.ingest
```

*(You can add your own PDF files with recipes - the one in `data/` folder is just an example.)*

**4. Run the FastAPI server**

```bash
uvicorn app:app --reload
```
Open http://127.0.0.1:8000/docs in your browser to test the agent.

**5. Running tests**

Execute the unit tests using pytest:
```bash
pytest
```

> *Originally developed as part of the summer challenge organized by KN Solvro (Wrocław University of Science and Technology).*
