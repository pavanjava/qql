# Healthcare Conversation RAG

A medical RAG (Retrieval-Augmented Generation) application that demonstrates semantic search over doctor-patient conversations using QQL and a local LLM.

## Overview

This application loads a dataset of doctor-patient conversations into Qdrant and provides a natural language search interface via an AI agent. It showcases how to:

- Load medical conversation data from Hugging Face
- Index it in Qdrant using QQL's bulk INSERT
- Build a semantic search tool that the agent calls for context

## Prerequisites

- Python 3.12+
- A running Qdrant instance (local or cloud)
- [Ollama](https://ollama.com) running locally with the `qwen3.5:latest` model
- QQL CLI installed

## Installation

```bash
# Install dependencies
pip install -e ../..  # install qql from the repo root

# Or install from pyproject.toml
pip install qql-cli datasets agno ollama openai
```

## Setup

### 1. Start Qdrant

```bash
# Via Docker
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Or use Qdrant Cloud — update the URL/secret in main.py
```

### 2. Start Ollama

```bash
ollama serve  # starts on localhost:11434 by default
ollama pull qwen3.5:latest
```

### 3. Load the data

```bash
python create_dataset.py
```

This generates `data_sets/source_data.qql` from the `pavanmantha/doctor_patient_conversation` Hugging Face dataset.

### 4. Run the QQL script to populate Qdrant

```bash
qql run data_sets/source_data.qql
```

Or interactively:

```bash
qql connect --url http://localhost:6333
qql> EXECUTE data_sets/source_data.qql
```

## Running the Application

```bash
python main.py
```

The agent will launch and you can ask medical questions in natural language. For example:

```
hi doctor I am just wondering what does abutment of the nerve root mean in a back issue please explain what treatment is required for annular bulging and tear
```

The agent will call `search_medical_records` to retrieve relevant conversation context from Qdrant, then generate a response.

## Project Structure

```
healthcare_conversation_rag/
├── main.py              # AI agent with search tool
├── create_dataset.py    # Generates .qql file from HuggingFace dataset
├── data_sets/
│   ├── source_data.qql  # Bulk INSERT script (generated)
│   └── ground_truth.jsonl  # Sample queries for evaluation
├── pyproject.toml
└── __init__.py
```

## How It Works

1. `create_dataset.py` loads a HuggingFace dataset and writes QQL `INSERT BULK` statements
2. Running that script populates the `doctor_patient_conversation` collection in Qdrant
3. `main.py` defines `search_medical_records()` — a function that:
   - Takes a natural language question
   - Runs a QQL `SEARCH ... SIMILAR TO` query with MMR diversity
   - Aggregates matching conversation texts into a context string
4. The Agno agent calls this function as a tool and uses the returned context to answer

## Key Files

| File | Purpose |
|---|---|
| `main.py` | Agent + search function — the entry point |
| `create_dataset.py` | Data pipeline — HuggingFace → `.qql` file |
| `data_sets/source_data.qql` | Ready-to-run bulk INSERT script |
| `pyproject.toml` | Dependencies |

## Customizing

**Change the collection name** — update `COLLECTION` in `create_dataset.py` and the query in `main.py`.

**Adjust search params** — modify the `WITH { ... }` block in `main.py:search_medical_records()`:

```python
query = f"SEARCH medical_records SIMILAR TO '{question}' LIMIT {LIMIT} WITH {{ hnsw_ef: 128, mmr_diversity: 0.5, mmr_candidates: 50}}"
```

**Use a different model** — change the `Ollama(id=...)` in `main.py`.

**Connect to Qdrant Cloud** — update the connection URL and secret in `main.py:Connection(...)`.