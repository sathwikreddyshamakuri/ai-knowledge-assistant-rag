from openai import OpenAI
import os


client = OpenAI()


def get_embedding(text: str):
    provider = os.getenv(
        "EMBEDDING_PROVIDER",
        "openai"
    )

    model = os.getenv(
        "EMBEDDING_MODEL",
        "text-embedding-3-small"
    )

    if provider == "openai":

        response = client.embeddings.create(
            model=model,
            input=text
        )

        return response.data[0].embedding

    raise ValueError(
        f"Unsupported embedding provider: {provider}"
    )