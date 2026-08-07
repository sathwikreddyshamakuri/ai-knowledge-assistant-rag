# AI Knowledge Assistant — Retrieval-Augmented Generation (RAG)

A Retrieval-Augmented Generation (RAG) backend built with **FastAPI**, the **OpenAI Responses API**, and **ChromaDB** for answering natural language questions over user-uploaded documents.

Instead of relying solely on an LLM's internal knowledge, the system retrieves semantically relevant document chunks from a vector database and injects them into the prompt, allowing responses to remain grounded in uploaded content.

---

## Project Overview

This project implements the core components of a RAG pipeline from scratch: document chunking, embedding generation, vector indexing, semantic retrieval, prompt construction, and streaming responses.

A deliberate design decision was made **not** to use orchestration frameworks such as LangChain or LlamaIndex. Building each component directly against the OpenAI API and ChromaDB made it possible to understand retrieval behavior, vector indexing, prompt construction, and streaming mechanics without framework abstractions.

---

## System Architecture

```
                Upload Documents
                        │
                        ▼
              Text Extraction (.txt)
                        │
                        ▼
             Overlapping Chunking
                        │
                        ▼
         OpenAI Embedding Generation
                        │
                        ▼
              Chroma Vector Database
                        │
                        ▼
           Semantic Similarity Search
                        │
                        ▼
         Top-K Relevant Chunks Retrieved
                        │
                        ▼
      OpenAI Responses API (Grounded Prompt)
                        │
                        ▼
          Streaming Response via FastAPI
```

---

## Features

- Semantic document search using vector embeddings
- Multi-document indexing
- Retrieval-Augmented Generation (RAG)
- Streaming AI responses
- Automatic document chunking
- OpenAI embedding generation
- ChromaDB vector storage
- Metadata-aware document indexing
- FastAPI REST API backend

---

## Engineering Decisions

### Building without RAG frameworks

Rather than using LangChain or LlamaIndex, the retrieval pipeline was implemented directly against the OpenAI API and ChromaDB.

This exposed every stage of the retrieval process:

- document chunking
- embedding generation
- vector indexing
- similarity search
- prompt construction
- streaming responses

The goal was to understand how modern RAG systems work internally before introducing orchestration frameworks.

### Chunk size (500 characters)

Documents are divided into approximately 500-character chunks. This size was selected because:

- chunks remain small enough to retrieve precise information
- embeddings remain focused on a single topic
- context windows are used efficiently
- retrieval avoids returning unnecessarily large passages

Very large chunks reduce retrieval precision, because multiple unrelated concepts become embedded together.

### Overlap (100 characters)

A 100-character overlap is maintained between consecutive chunks.

Without overlap, important sentences located near chunk boundaries may be split across two chunks, causing incomplete retrieval. Overlapping preserves context while introducing only a small amount of redundancy.

### Top-K retrieval

The system retrieves the top 3 most semantically similar chunks. Three chunks generally provide enough context for accurate responses while keeping prompts compact.

Retrieving too few risks missing relevant information; retrieving too many increases token usage and may introduce unrelated context.

### Embedding model selection

The project currently uses `text-embedding-3-small` for:

- lower latency
- lower API cost
- strong semantic retrieval quality for small and medium-sized document collections

Although `text-embedding-3-large` provides higher-quality embeddings, the additional cost was not justified at the current scale. The implementation is model-agnostic and can be upgraded without architectural changes.

---

## Technology Stack

| Category | Technology |
|---|---|
| Language | Python |
| Backend | FastAPI |
| LLM | OpenAI Responses API |
| Embeddings | `text-embedding-3-small` |
| Vector Database | ChromaDB |
| API Testing | Swagger UI |
| Streaming | FastAPI `StreamingResponse` |

---

## API Endpoints

### `POST /upload-document`

Uploads one or more text documents. Each uploaded document is extracted, chunked, embedded, and indexed in ChromaDB.

### `POST /ask-ai`

General-purpose AI endpoint without retrieval.

### `POST /ask-ai-document`

Question-answering endpoint backed by semantic retrieval.

Workflow:

1. Embed the user's question
2. Perform semantic similarity search in ChromaDB
3. Retrieve the top-K matching chunks
4. Build a grounded prompt
5. Generate a streamed response

---

## Current Capabilities

- Multi-document indexing
- Semantic vector search
- Retrieval-Augmented Generation
- Streaming responses
- Metadata-aware indexing
- OpenAI embedding generation
- ChromaDB integration
- FastAPI backend

---

## Challenges Encountered

Several implementation issues were identified and resolved during development:

- **Manual similarity search.** Initial retrieval used hand-written cosine similarity before migrating to ChromaDB's built-in vector search.
- **Chunk ID collisions.** Chunk IDs originally collided when multiple documents were uploaded, silently overwriting data. Unique document-scoped chunk identifiers were introduced to prevent this.
- **Single-document retrieval.** Retrieval was initially limited to individual documents. The architecture was redesigned to support semantic retrieval across the entire indexed collection.
- **Missing grounding.** Prompt construction was updated so retrieved context is always supplied to the model before answer generation.

---

## Current Limitations

The current implementation intentionally keeps the architecture simple.

- Supports `.txt` documents only
- No user authentication or session isolation
- No rate limiting or request throttling
- No retrieval evaluation metrics (Recall@K, Precision@K, MRR, hit rate)
- No citation highlighting of retrieved source passages
- Previously uploaded documents remain indexed until the vector database is cleared
- No frontend; interaction is through FastAPI Swagger UI

---

## Planned Enhancements

- PDF document support
- Session-based document isolation
- Metadata filtering
- Conversational memory
- Source citations
- Docker containerization
- React frontend
- AWS deployment

---

## Running the Project

Clone the repository:

```bash
git clone https://github.com/sathwikreddyshamakuri/ai-knowledge-assistant-rag
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure your API key in a `.env` file:

```bash
OPENAI_API_KEY=your_api_key
```

Run the server:

```bash
uvicorn main:app --reload
```

Open Swagger UI at `http://127.0.0.1:8000/docs`.

---

## License

MIT