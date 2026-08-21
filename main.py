from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
from fastapi.responses import StreamingResponse
import chromadb
import uuid
import time
import threading

app = FastAPI()

client = OpenAI()

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="documents"
)


class Question(BaseModel):
    question: str = Field(min_length=1)


RATE_LIMIT = 5
RATE_WINDOW = 60

request_history = {}

rate_limit_lock = threading.Lock()

conversation_history = {}

MAX_HISTORY_MESSAGES = 6


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

    conversation_history[session_id].append(
        {
            "role": role,
            "content": content
        }
    )


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
                detail="Rate limit exceeded. Please try again later.",
                headers={
                    "Retry-After": str(retry_after)
                }
            )

        timestamps.append(current_time)

        request_history[client_ip] = timestamps


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

    for pattern in suspicious_patterns:

        if pattern in normalized_question:
            return False

    return True


def get_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


def extract_text_from_txt(file):

    contents = file.file.read()

    text = contents.decode("utf-8")

    return text


def split_into_chunks(
    text,
    chunk_size=500,
    overlap=100
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start = end - overlap

    return chunks


def store_chunks_in_chroma(
    chunks,
    document_id,
    document_name,
    session_id
):

    for index, chunk in enumerate(chunks):

        chunk_embedding = get_embedding(chunk)

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
    top_k=3
):

    question_embedding = get_embedding(
        user_question
    )

    results = collection.query(
        query_embeddings=[
            question_embedding
        ],
        n_results=top_k,
        where={
            "session_id": session_id
        }
    )

    relevant_chunks = results["documents"][0]

    retrieved_metadata = results["metadatas"][0]

    print("\nTop retrieved chunks from Chroma:")

    for chunk, metadata in zip(
        relevant_chunks,
        retrieved_metadata
    ):

        print(
            "\nDocument:",
            metadata["document_name"]
        )

        print(
            "Session:",
            metadata["session_id"]
        )

        print(
            "Chunk number:",
            metadata["chunk_number"]
        )

        print(
            "Chunk:",
            chunk
        )

    return relevant_chunks


def generator_response(
    user_question,
    session_id
):

    history = get_conversation_history(
        session_id
    )

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

        stream = client.responses.create(
            model="gpt-4.1-mini",
            instructions=instructions,
            input=prompt,
            stream=True
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

    except Exception as e:

        print(
            "OpenAI API error:",
            e
        )

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your request"
        )


@app.post("/ask-ai")
def ask_ai(
    request: Request,
    session_id: str,
    question: Question
):

    client_ip = request.client.host

    check_rate_limit(client_ip)

    user_question = question.question.strip()

    if not validate_prompt_injection(
        user_question
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "The request was rejected because "
                "it contains a potentially unsafe instruction."
            )
        )

    return StreamingResponse(
        generator_response(
            user_question,
            session_id
        ),
        media_type="text/plain"
    )


def generator_document_response(
    user_question,
    session_id
):

    relevant_chunks = search_chroma(
        user_question,
        session_id,
        top_k=3
    )

    context = "\n\n".join(
        relevant_chunks
    )

    history = get_conversation_history(
        session_id
    )

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

    try:

        stream = client.responses.create(
            model="gpt-4.1-mini",
            instructions=instructions,
            input=prompt,
            stream=True
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

    except Exception as e:

        print(
            "OpenAI API error:",
            e
        )

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your request"
        )


@app.post("/upload-document")
def upload_document(
    files: List[UploadFile] = File(...),
    session_id: str = ""
):

    if not session_id:

        session_id = str(
            uuid.uuid4()
        )

    uploaded_documents = []

    for file in files:

        document_text = extract_text_from_txt(
            file
        )

        document_id = str(
            uuid.uuid4()
        )

        chunks = split_into_chunks(
            document_text,
            chunk_size=500,
            overlap=100
        )

        store_chunks_in_chroma(
            chunks,
            document_id,
            file.filename,
            session_id
        )

        uploaded_documents.append(
            {
                "document_id": document_id,
                "document_name": file.filename
            }
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

    client_ip = request.client.host

    check_rate_limit(client_ip)

    user_question = question.strip()

    if not validate_prompt_injection(
        user_question
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "The request was rejected because "
                "it contains a potentially unsafe instruction."
            )
        )

    return StreamingResponse(
        generator_document_response(
            user_question,
            session_id
        ),
        media_type="text/plain"
    )