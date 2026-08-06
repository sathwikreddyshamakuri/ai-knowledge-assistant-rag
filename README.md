# AI Knowledge Assistant – Retrieval-Augmented Generation (RAG)

A Retrieval-Augmented Generation (RAG) application built with **FastAPI**, **OpenAI API**, and **ChromaDB** that answers user questions by retrieving semantically relevant document chunks before generating responses.

Unlike a standard LLM chatbot, this project grounds its answers using uploaded documents, reducing hallucinations and improving factual accuracy.

---

# Project Overview

This project demonstrates the complete workflow of a modern Retrieval-Augmented Generation (RAG) system.

Users can upload one or more text documents, ask natural language questions, and receive AI-generated answers grounded in the most relevant document content.

The project is built incrementally to understand each component of a Retrieval-Augmented Generation (RAG) pipeline from first principles instead of relying on high-level frameworks.

---

# Architecture

```

                Upload Documents
                        │
                        ▼
              Text Extraction (.txt)
                        │
                        ▼
                  Chunking
                        │
                        ▼
             OpenAI Embedding Model
                        │
                        ▼
              Chroma Vector Database
                        │
                        ▼
           Semantic Similarity Search
                        │
                        ▼
          Top-K Semantically Relevant Chunks Retrieved
                        │
                        ▼
      OpenAI LLM Generates Final Answer
                        │
                        ▼
          Streaming Response via FastAPI

```
---
Each uploaded document is split into overlapping chunks, converted into embeddings using OpenAI Embeddings, indexed in ChromaDB with metadata, and retrieved through semantic similarity search before being provided as context to the language model.
---

# Features

- AI-powered question answering using OpenAI Responses API
- Real-time streaming responses
- Upload and index one or more text documents
- Automatic document chunking
- Embedding generation using OpenAI Embeddings
- Semantic similarity search using ChromaDB
- Retrieval-Augmented Generation (RAG)
- Multi-document semantic retrieval
- FastAPI REST API backend
- Metadata-aware document indexing

---

# Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Backend | FastAPI |
| LLM | OpenAI GPT Models (Responses API) |
| Embeddings | text-embedding-3-small |
| Vector Database | ChromaDB |
| API Testing | Swagger UI |
| Streaming | FastAPI StreamingResponse |

---

# API Endpoints

## Upload Documents

```
POST /upload-document
```

Uploads one or more text documents.

Each document is:

- extracted
- chunked
- embedded
- stored inside ChromaDB

---

## Ask AI

```
POST /ask-ai
```

General AI assistant without document retrieval.

---

## Ask AI using RAG

```
POST /ask-ai-document
```

Workflow:

- Convert question into embedding
- Search ChromaDB
- Retrieve Top-K relevant chunks
- Build grounded prompt
- Generate streamed response

---

# Project Workflow

### 1. Upload Documents

Users upload one or more text files.

↓

### 2. Document Processing

Documents are converted into plain text.

↓

### 3. Chunking

Large documents are divided into overlapping chunks.

↓

### 4. Embedding Generation

Each chunk is converted into a high-dimensional vector using OpenAI Embeddings.

↓

### 5. Vector Storage

Chunk embeddings and metadata are stored inside ChromaDB.

↓

### 6. Semantic Search

When a question is asked:

- the question is embedded
- ChromaDB finds the most semantically similar chunks

↓

### 7. Retrieval-Augmented Generation

Retrieved chunks are injected into the LLM prompt.

↓

### 8. Streaming Response

The generated answer is streamed back to the client.

---

# Current Capabilities

- Streaming AI responses
- Prompt engineering
- Document processing
- Chunking with overlap
- Embedding generation
- Vector search
- ChromaDB integration
- Multi-document retrieval
- Retrieval-Augmented Generation (RAG)
- Vector Database Design

---

# Planned Enhancements

- Session-based document isolation
- Metadata filtering for document-specific retrieval
- PDF document support
- React frontend
- Conversational memory
- Source citations
- Docker containerization
- AWS cloud deployment

---

# Running the Project

## Clone

```bash
git clone <repository-url>
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Set your OpenAI API key

```bash
OPENAI_API_KEY=your_api_key
```

## Run

```bash
uvicorn main:app --reload
```

Open Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

# Learning Objectives

This project was built to gain hands-on experience with the core concepts behind modern LLM applications, including:

- Prompt Engineering
- Streaming Responses
- Document Processing
- Chunking Strategies
- Embeddings
- Vector Databases
- Semantic Search
- Retrieval-Augmented Generation (RAG)
- FastAPI Backend Development

---

# License
