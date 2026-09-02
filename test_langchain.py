from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-mini")

response = llm.invoke(
    "Explain what AWS Lambda is in one sentence."
)

print(response.content)