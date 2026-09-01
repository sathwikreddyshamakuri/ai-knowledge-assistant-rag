from openai import OpenAI
import os


client = OpenAI()


def generate_response(
    instructions: str,
    prompt: str
):

    provider = os.getenv(
        "LLM_PROVIDER",
        "openai"
    )

    model = os.getenv(
        "LLM_MODEL",
        "gpt-4.1-mini"
    )

    if provider == "openai":

        return client.responses.create(
            model=model,
            instructions=instructions,
            input=prompt,
            stream=True
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider}"
    )