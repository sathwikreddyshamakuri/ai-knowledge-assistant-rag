# AI Knowledge Assistant — Retrieval-Augmented Generation (RAG)

A RAG backend built with **FastAPI**, **OpenAI**, **LangChain**, and **ChromaDB** for answering questions about uploaded documents.

The project started as a RAG pipeline built directly with the OpenAI API and ChromaDB. That gave me a chance to understand what happens under the hood — chunking, embeddings, vector search, prompt construction, and response streaming. I'm now introducing LangChain step by step to see where its abstractions are actually useful in a real application.

---

## Project Overview

The assistant takes uploaded documents, breaks them into smaller chunks, creates embeddings for those chunks, and stores them in ChromaDB.

When a user asks a question, the system searches for the most relevant chunks and uses them as context for the LLM instead of relying only on the model's built-in knowledge.

The project is intentionally being built in stages. I first implemented the core RAG flow myself and then started adding LangChain so I can understand both approaches rather than treating the framework as a black box.

---

## System Architecture

```text
                 Upload Document
                        │
                        ▼
              Text Extraction (.txt)
                        │
                        ▼
             Overlapping Chunking
                        │
                        ▼
              Embedding Generation
                        │
                        ▼
                ChromaDB Storage
                        │
                        │
                        ▼
                  User Question
                        │
                        ▼
             LangChain Retriever
                        │
                        ▼
          Top 3 Relevant Document Chunks
                        │
                        ▼
              Conversation History
                        │
                        ▼
                Prompt Construction
                        │
                        ▼
                   OpenAI LLM
                        │
                        ▼
             Streaming API Response
```

---

## Features

- Retrieval-Augmented Generation (RAG)
- Semantic search over uploaded documents
- Multi-document indexing
- Automatic document chunking
- OpenAI embeddings
- ChromaDB vector storage
- LangChain retriever abstraction
- Metadata-based session filtering
- Conversational memory
- Basic prompt-injection validation
- Request rate limiting
- Request ID based logging
- Embedding, retrieval, and LLM timing logs
- Streaming responses with FastAPI
- Swagger API documentation

---

## Engineering Decisions

### Building the first version without LangChain

The first version was intentionally built directly against the OpenAI API and ChromaDB.

This made each part of the RAG pipeline visible:

```text
Document
   ↓
Chunking
   ↓
Embedding
   ↓
Vector Storage
   ↓
Similarity Search
   ↓
Context
   ↓
Prompt
   ↓
LLM
```

Once that flow was working, LangChain was introduced gradually.

The goal is not to use a framework just because it is popular. The goal is to understand what it provides and where it actually makes the application easier to build and maintain.

### LangChain

LangChain is currently being introduced around the retrieval and LLM portions of the application.

Some of the main abstractions being used are:

- `OpenAIEmbeddings` for embedding generation
- `Chroma` for the vector store integration
- `Retriever` for document retrieval
- `Document` for standardized document content and metadata
- Prompt templates for prompt construction
- LLM interfaces for model interaction
- Runnables and LCEL for connecting components

The underlying services have not changed. ChromaDB is still the vector database and OpenAI is still being used for embeddings and generation. LangChain sits between the application and those services and provides common interfaces for working with them.

### Chunking

Documents are currently split into approximately **500-character chunks** with a **100-character overlap**.

The goal is to keep each retrieved passage focused enough to be useful while still preserving some context between neighboring chunks.

Chunking happens before embedding. Each chunk gets its own embedding rather than creating one embedding for the entire document.

### Top-K retrieval

The retriever currently returns the **top 3 most relevant chunks**.

The value is configured on the retriever itself rather than being passed into every retrieval call.

This keeps the retrieval behavior in one place and makes it easier to change later.

### Metadata filtering

Each stored chunk includes metadata such as:

```text
document_id
document_name
chunk_number
session_id
```

The `session_id` is used during retrieval so a question only searches the documents associated with that session.

### Embeddings

The project currently uses:

```text
text-embedding-3-small
```

It provides a good balance between retrieval quality, latency, and API cost for the current size of the project.

The embedding layer is kept separate from the rest of the application so the model or provider can be changed later without rewriting the retrieval logic.

### Conversation history

The application keeps recent conversation history for each session.

This allows follow-up questions to use previous messages as context instead of treating every question as completely independent.

The conversation history is currently kept in memory. LangChain's message-history abstractions will be explored as part of the next stages of the project.

### Prompt-injection validation

A basic validation layer checks incoming requests for common prompt-injection patterns.

This is intentionally a simple guardrail. It is not meant to be a complete security solution.

### Rate limiting

The API currently limits clients to:

```text
5 requests per minute
```

The current implementation is in memory and is intended for the learning project. A distributed production deployment would need a shared mechanism such as Redis.

---

## Technology Stack

| Category | Technology |
|---|---|
| Language | Python |
| Backend | FastAPI |
| LLM | OpenAI |
| LLM API | OpenAI Responses API |
| Embeddings | `text-embedding-3-small` |
| Orchestration | LangChain |
| Vector Database | ChromaDB |
| API Testing | Swagger UI |
| Streaming | FastAPI `StreamingResponse` |
| Configuration | `.env` |
| Logging | Python logging |

---

## API Endpoints

### `POST /upload-document`

Uploads one or more text documents.

The documents are extracted, split into chunks, embedded, and stored in ChromaDB along with their metadata.

### `POST /ask-ai`

General-purpose AI endpoint that does not use document retrieval.

### `POST /ask-ai-document`

Answers questions using the uploaded documents.

The current workflow is:

1. Receive the user's question
2. Search for relevant document chunks
3. Apply the session metadata filter
4. Include recent conversation history
5. Build the grounded prompt
6. Generate and stream the response

---

## Observability

Requests are assigned a unique request ID.

The same request ID is passed through the major parts of the pipeline so that a request can be followed through logs:

```text
Request
   ↓
Embedding
   ↓
Retrieval
   ↓
LLM
   ↓
Response
```

Timing information is also recorded for important operations, which makes it easier to see where latency is coming from during development and debugging.

---

## Current Capabilities

- Multi-document indexing
- Semantic vector search
- Retrieval-Augmented Generation
- LangChain retrieval abstractions
- Session-based retrieval filtering
- Conversational memory
- Prompt-injection validation
- Basic rate limiting
- Request-level logging
- Retrieval and LLM latency tracking
- Streaming responses
- FastAPI REST API
- OpenAI integration
- ChromaDB integration

---

## Challenges Encountered

A few issues came up while building the project:

- **Manual similarity search:** The first version used hand-written similarity logic before moving to ChromaDB's vector search.
- **Chunk ID collisions:** Chunk IDs originally collided across documents, which could cause existing chunks to be overwritten. Document-scoped IDs were added to fix this.
- **Single-document retrieval:** Retrieval initially focused on one document at a time. It was later changed to search across the indexed collection.
- **Grounding:** Prompt construction was updated so retrieved document content is explicitly supplied to the model.
- **Session isolation:** Metadata filtering was added so retrieval results stay within the appropriate session.
- **Observability:** Request IDs were propagated through embedding, retrieval, and LLM operations to make debugging individual requests easier.
- **Framework migration:** LangChain is being introduced gradually after the underlying RAG flow was implemented directly. This makes it easier to compare the two approaches and understand what the framework is actually doing.

---

## Current Limitations

The project is still intentionally focused on the core RAG and LLM application workflow.

- Supports `.txt` documents only
- Conversation history is stored in memory
- Rate limiting is stored in memory
- Prompt-injection protection is basic
- No authentication system
- No frontend
- No automated test suite yet
- No production-grade persistent conversation store
- No advanced reranking pipeline
- No distributed deployment

---

## Planned Enhancements

The next stages of the project will build on the current RAG foundation:

- Complete LangChain integration
- Better prompt and chain orchestration
- LangGraph agent workflows
- Tool calling
- MCP integration
- PDF and Word document support
- Improved document ingestion
- Retrieval improvements and reranking
- Automated testing
- Docker containerization
- CI/CD pipeline
- React frontend
- Cloud deployment
- Production-grade observability
- Persistent conversation storage

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

Create a `.env` file:

```bash
OPENAI_API_KEY=your_api_key
```

Run the server:

```bash
uvicorn main:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## License

MIT
