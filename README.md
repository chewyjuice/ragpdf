# RAG PDF Question Answering

A Retrieval-Augmented Generation (RAG) system for querying PDF documents using LangChain, ChromaDB, and Ollama.

## Features

- PDF document loading and processing
- Local embeddings using HuggingFace sentence-transformers
- Vector storage with ChromaDB
- Multiple query interfaces:
  - Simple RAG chain
  - RAG with chat history/memory
  - Agent-based querying (with tool-calling models)
- Local LLM support via Ollama
- Optional OpenAI integration

## Prerequisites

1. **Python 3.13+** - This project requires Python 3.13 or higher
2. **uv** - Fast Python package installer ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))
3. **Ollama** - Local LLM runtime ([download here](https://ollama.ai/))

### Install uv

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### Install Ollama

Download and install Ollama from [https://ollama.ai/](https://ollama.ai/)

Then pull the required models:

```bash
# For basic RAG (smaller, faster)
ollama pull llama3.2:1b

# For agent-based querying (supports tool calling)
ollama pull llama3.1
```

## Installation

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd ragpdf
```

2. **Create a virtual environment and install dependencies**

```bash
# Create virtual environment with uv
uv venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"
```

## Usage

### 1. Prepare Your PDF

Place your PDF file in the project directory (e.g., `annual-report-2024.pdf`)

### 2. Run the Notebook

```bash
# Start Jupyter
jupyter notebook

# Or use Jupyter Lab
jupyter lab
```

Open `RagPdf.ipynb` in your browser.

### 3. Run the Cells

The notebook is organized into sections:

#### **Basic Setup (Cells 1-3)**
- Cell 1: Load and split PDF documents
- Cell 2: Create embeddings and vector store
- Cell 3: Set up basic RAG chain

#### **Simple Querying (Cells 4-5)**
```python
# Run a single query
answer = rag_chain.invoke("What were the company's total revenues?")
print(answer)

# Stream responses
for chunk in rag_chain.stream("What were the main risk factors?"):
    print(chunk, end="", flush=True)
```

#### **Chat with Memory (Cells 6-8)**
```python
# Ask follow-up questions with context
print(ask("What were the company's total revenues?"))
print(ask("How does that compare to last year?"))  # Remembers previous context
```

#### **Agent-Based Querying (Cells 9-10)**

**Option A: With tool-calling model (llama3.1)**
```python
# The agent can reason and use tools
result = agent_executor.invoke({"input": "What were the company's total revenues?"})
print(result["output"])
```

**Option B: Simple RAG (any model)**
```python
# Works with llama3.2:1b or any other model
print(ask_simple("What were the company's total revenues?"))
```

## Project Structure

```
ragpdf/
├── RagPdf.ipynb              # Main notebook
├── pyproject.toml            # Project dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── chroma_db/               # Vector database (auto-generated)
└── *.pdf                    # Your PDF documents
```

## Configuration

### Change the PDF File

Edit Cell 1:
```python
loader = PyPDFLoader("your-document.pdf")
```

### Change the LLM Model

Edit Cell 3:
```python
model = ChatOllama(
    model="llama3.2:3b",  # or llama3.1, mistral, etc.
    temperature=0
)
```

### Adjust Retrieval Settings

Edit Cell 2 or 3:
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # Number of chunks to retrieve
```

### Change Chunk Size

Edit Cell 1:
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Adjust chunk size
    chunk_overlap=200,    # Adjust overlap
)
```

## Troubleshooting

### "No data received from Ollama stream" Error

This means your model doesn't support tool calling. Solutions:
1. Use `llama3.1` or another tool-calling compatible model
2. Use the "Simple RAG" alternative (works with any model)

### ChromaDB Errors

Delete the `chroma_db/` directory and re-run cells 1-2 to rebuild the vector store.

### Memory Issues

If you run out of memory:
1. Use a smaller model (e.g., `llama3.2:1b`)
2. Reduce chunk size in Cell 1
3. Reduce `k` value in retriever settings

### Slow Performance

1. Use a smaller model
2. Reduce the number of retrieved chunks (`k` parameter)
3. Use a GPU-enabled setup for Ollama

## Optional: OpenAI Integration

To use OpenAI instead of Ollama:

1. Install OpenAI package:
```bash
uv pip install langchain-openai
```

2. Set your API key:
```bash
export OPENAI_API_KEY="your-api-key"
```

3. Update the model in Cell 3:
```python
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o", temperature=0)
```

## Models Comparison

| Model | Size | Speed | Tool Calling | Best For |
|-------|------|-------|--------------|----------|
| llama3.2:1b | ~1GB | Fast | ❌ | Simple RAG, testing |
| llama3.2:3b | ~3GB | Medium | ✅ | Balanced performance |
| llama3.1 | ~4GB | Medium | ✅ | Agent-based querying |
| mistral | ~4GB | Medium | ✅ | Alternative to llama3.1 |

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]
