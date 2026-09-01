from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from typing import List
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse
from services.embedding_provider import get_embedding
from services.llm_provider import generate_response
import chromadb
import uuid
import time
import threading
import logging


app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="documents"
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    request.state.request_id = request_id

    try:
        response = await call_next(request)

        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.info(
            "request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms
        )

        response.headers["X-Request-ID"] = request_id

        return response

    except Exception:
        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.exception(
            "request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms
        )

        raise


class Question(BaseModel):
    question: str = Field(min_length=1)


RATE_LIMIT = 5
RATE_WINDOW = 60

request_history = {}
rate_limit_lock = threading.Lock()

conversation_history = {}
MAX_HISTORY_MESSAGES = 6


def check_rate_limit(client_ip):
    current_time = time.time()

    with rate_limit_lock:
        timestamps = request_history.get(
            client_ip,
            []
        )

        timestamps = [
            timestamp
            for timestamp in timestamps
            if current_time - timestamp < RATE_WINDOW
        ]

        if len(timestamps) >= RATE_LIMIT:
            retry_after = int(
                RATE_WINDOW - (
                    current_time - timestamps[0]
                )
            ) + 1

            raise HTTPException(
                status_code=429,
                detail=(
                    "Rate limit exceeded. "
                    "Please try again later."
                ),
                headers={
                    "Retry-After": str(retry_after)
                }
            )

        timestamps.append(current_time)
        request_history[client_ip] = timestamps


def get_conversation_history(session_id):
    history = conversation_history.get(
        session_id,
        []
    )

    return history[-MAX_HISTORY_MESSAGES:]


def add_to_conversation(
    session_id,
    role,
    content
):
    if session_id not in conversation_history:
        conversation_history[session_id] = []

    conversation_history[session_id].append({
        "role": role,
        "content": content
    })


def validate_prompt_injection(user_question):
    suspicious_patterns = [
        "ignore previous instructions",
        "ignore all previous instructions",
        "ignore the instructions above",
        "reveal the system prompt",
        "show me the system prompt",
        "disregard previous instructions",
        "forget your instructions"
    ]

    normalized_question = user_question.lower()

    return not any(
        pattern in normalized_question
        for pattern in suspicious_patterns
    )


def extract_text_from_txt(file):
    contents = file.file.read()
    return contents.decode("utf-8")


def split_into_chunks(
    text,
    chunk_size=500,
    overlap=100
):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


def store_chunks_in_chroma(
    chunks,
    document_id,
    document_name,
    session_id,
    request_id
):
    for index, chunk in enumerate(chunks):
        start_time = time.perf_counter()

        chunk_embedding = get_embedding(chunk)

        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.info(
            "request_id=%s embedding_completed duration_ms=%.2f",
            request_id,
            duration_ms
        )

        collection.upsert(
            ids=[
                f"{document_id}_chunk_{index}"
            ],
            embeddings=[
                chunk_embedding
            ],
            documents=[
                chunk
            ],
            metadatas=[
                {
                    "document_id": document_id,
                    "document_name": document_name,
                    "chunk_number": index,
                    "session_id": session_id
                }
            ]
        )


def search_chroma(
    user_question,
    session_id,
    top_k=3,
    request_id=None
):
    start_time = time.perf_counter()

    question_embedding = get_embedding(
        user_question
    )

    embedding_duration_ms = (
        time.perf_counter() - start_time
    ) * 1000

    logger.info(
        "request_id=%s embedding_completed duration_ms=%.2f",
        request_id,
        embedding_duration_ms
    )

    retrieval_start = time.perf_counter()

    results = collection.query(
        query_embeddings=[
            question_embedding
        ],
        n_results=top_k,
        where={
            "session_id": session_id
        }
    )

    retrieval_duration_ms = (
        time.perf_counter() - retrieval_start
    ) * 1000

    relevant_chunks = results["documents"][0]
    retrieved_metadata = results["metadatas"][0]

    logger.info(
        "request_id=%s retrieval_completed "
        "embedding_ms=%.2f "
        "chroma_ms=%.2f "
        "top_k=%d "
        "chunks=%d",
        request_id,
        embedding_duration_ms,
        retrieval_duration_ms,
        top_k,
        len(relevant_chunks)
    )

    print("\nTop retrieved chunks from Chroma:")

    for chunk, metadata in zip(
        relevant_chunks,
        retrieved_metadata
    ):
        print("\nDocument:", metadata["document_name"])
        print("Session:", metadata["session_id"])
        print("Chunk number:", metadata["chunk_number"])
        print("Chunk:", chunk)

    return relevant_chunks


def generator_response(
    user_question,
    session_id,
    request_id
):
    history = get_conversation_history(session_id)

    history_text = "\n\n".join(
        f"{message['role'].capitalize()}: "
        f"{message['content']}"
        for message in history
    )

    instructions = """
    You are a helpful AI Knowledge Assistant.

    Use the conversation history to understand
    follow-up questions.

    Explain concepts clearly and accurately,
    adapting to the user's level.

    Use simple language, define technical terms,
    and provide examples when helpful.

    Do not invent facts.

    If uncertain, say so.

    Answer the user's question directly.
    """

    prompt = f"""
    Conversation History:

    {history_text}

    Current User Question:

    {user_question}
    """

    try:
        stream = generate_response(
            instructions,
            prompt
        )

        full_response = ""

        for event in stream:
            if event.type == "response.output_text.delta":
                full_response += event.delta
                yield event.delta

        add_to_conversation(
            session_id,
            "user",
            user_question
        )

        add_to_conversation(
            session_id,
            "assistant",
            full_response
        )

    except Exception:
        logger.exception(
            "request_id=%s OpenAI API error",
            request_id
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "An unexpected error occurred "
                "while processing your request"
            )
        )


@app.post("/ask-ai")
def ask_ai(
    request: Request,
    session_id: str,
    question: Question
):
    request_id = request.state.request_id
    client_ip = request.client.host

    check_rate_limit(client_ip)

    user_question = question.question.strip()

    if not validate_prompt_injection(user_question):
        logger.warning(
            "request_id=%s prompt rejected",
            request_id
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "The request was rejected because "
                "it contains a potentially unsafe "
                "instruction."
            )
        )

    logger.info(
        "request_id=%s AI request accepted session_id=%s",
        request_id,
        session_id
    )

    return StreamingResponse(
        generator_response(
            user_question,
            session_id,
            request_id
        ),
        media_type="text/plain"
    )


def generator_document_response(
    user_question,
    session_id,
    request_id
):
    relevant_chunks = search_chroma(
        user_question,
        session_id,
        top_k=3,
        request_id=request_id
    )

    context = "\n\n".join(relevant_chunks)

    history = get_conversation_history(session_id)

    history_text = "\n\n".join(
        f"{message['role'].capitalize()}: "
        f"{message['content']}"
        for message in history
    )

    instructions = """
    You are a helpful AI Knowledge Assistant.

    Answer the user's question using the
    provided document context and conversation history.

    Use simple language.

    Do not invent facts that are not supported
    by the provided context.

    If the answer cannot be found in the context,
    say so.

    Use conversation history only to understand
    references such as "it", "they", "that",
    or follow-up questions.
    """

    prompt = f"""
    Conversation History:

    {history_text}

    Document Context:

    {context}

    Current User Question:

    {user_question}
    """

    llm_start_time = time.perf_counter()
    first_token_time = None
    full_response = ""

    try:
        stream = generate_response(
            instructions,
            prompt
        )

        for event in stream:
            if event.type == "response.output_text.delta":

                if first_token_time is None:
                    first_token_time = (
                        time.perf_counter()
                        - llm_start_time
                    ) * 1000

                    logger.info(
                        "request_id=%s "
                        "llm_first_token "
                        "duration_ms=%.2f",
                        request_id,
                        first_token_time
                    )

                full_response += event.delta
                yield event.delta

        total_llm_time = (
            time.perf_counter()
            - llm_start_time
        ) * 1000

        logger.info(
            "request_id=%s "
            "llm_completed "
            "duration_ms=%.2f",
            request_id,
            total_llm_time
        )

        add_to_conversation(
            session_id,
            "user",
            user_question
        )

        add_to_conversation(
            session_id,
            "assistant",
            full_response
        )

    except Exception:
        logger.exception(
            "request_id=%s OpenAI API error",
            request_id
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "An unexpected error occurred "
                "while processing your request"
            )
        )


@app.post("/upload-document")
def upload_document(
    request: Request,
    files: List[UploadFile] = File(...),
    session_id: str = ""
):
    request_id = request.state.request_id

    if not session_id:
        session_id = str(uuid.uuid4())

    uploaded_documents = []

    for file in files:
        document_text = extract_text_from_txt(file)
        document_id = str(uuid.uuid4())

        chunks = split_into_chunks(
            document_text,
            chunk_size=500,
            overlap=100
        )

        store_chunks_in_chroma(
            chunks,
            document_id,
            file.filename,
            session_id,
            request_id
        )

        uploaded_documents.append({
            "document_id": document_id,
            "document_name": file.filename
        })

    logger.info(
        "request_id=%s document_upload_completed "
        "session_id=%s documents=%d",
        request_id,
        session_id,
        len(uploaded_documents)
    )

    return {
        "session_id": session_id,
        "documents": uploaded_documents
    }


@app.post("/ask-ai-document")
def ask_ai_document(
    request: Request,
    session_id: str,
    question: str
):
    request_id = request.state.request_id
    client_ip = request.client.host

    check_rate_limit(client_ip)

    user_question = question.strip()

    if not validate_prompt_injection(user_question):
        logger.warning(
            "request_id=%s prompt rejected",
            request_id
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "The request was rejected because "
                "it contains a potentially unsafe "
                "instruction."
            )
        )

    logger.info(
        "request_id=%s document request accepted "
        "session_id=%s",
        request_id,
        session_id
    )

    return StreamingResponse(
        generator_document_response(
            user_question,
            session_id,
            request_id
        ),
        media_type="text/plain"
    )