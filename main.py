from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
from fastapi.responses import StreamingResponse
import chromadb
import uuid

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


def generator_response(user_question):

    instructions = """
    You are a helpful AI Knowledge Assistant.
    Explain concepts clearly and accurately, adapting to the user's level.
    Use simple language, define technical terms, and provide examples when helpful.
    Do not invent facts. If uncertain, say so.
    Answer the user's question directly.
    """

    try:

        stream = client.responses.create(
            model="gpt-4.1-mini",
            instructions=instructions,
            input=user_question,
            stream=True
        )

        for event in stream:

            if event.type == "response.output_text.delta":
                yield event.delta

    except Exception as e:

        print("OpenAI API error:", e)

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your request"
        )


@app.post("/ask-ai")
def ask_ai(request: Question):

    user_question = request.question.strip()

    if not validate_prompt_injection(user_question):

        raise HTTPException(
            status_code=400,
            detail="The request was rejected because it contains a potentially unsafe instruction."
        )

    return StreamingResponse(
        generator_response(user_question),
        media_type="text/plain"
    )

def extract_text_from_txt(file):

    contents = file.file.read()

    text = contents.decode("utf-8")

    return text


def split_into_chunks(text, chunk_size=500, overlap=100):

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


def generator_document_response(
    user_question,
    session_id
):

    relevant_chunks = search_chroma(
        user_question,
        session_id,
        top_k=3
    )

    context = "\n\n".join(relevant_chunks)

    instructions = """
    You are a helpful AI Knowledge Assistant.

    Answer the user's question using the provided document context.

    Explain concepts clearly and accurately.
    Use simple language.
    Do not invent facts that are not supported by the provided context.
    If the answer cannot be found in the context, say so.
    """

    prompt = f"""
    Document Context:
    {context}

    User Question:
    {user_question}
    """

    try:

        stream = client.responses.create(
            model="gpt-4.1-mini",
            instructions=instructions,
            input=prompt,
            stream=True
        )

        for event in stream:

            if event.type == "response.output_text.delta":
                yield event.delta

    except Exception as e:

        print("OpenAI API error:", e)

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
    session_id: str,
    question: str
):

    user_question = question.strip()

    if not validate_prompt_injection(user_question):

        raise HTTPException(
            status_code=400,
            detail="The request was rejected because it contains a potentially unsafe instruction."
        )

    return StreamingResponse(
        generator_document_response(
            user_question,
            session_id
        ),
        media_type="text/plain"
    )

def get_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


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

        print("\nDocument:", metadata["document_name"])

        print("Session:", metadata["session_id"])

        print("Chunk number:", metadata["chunk_number"])

        print("Chunk:", chunk)

    return relevant_chunks

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