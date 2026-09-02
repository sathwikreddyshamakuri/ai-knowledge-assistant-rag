from main import search_chroma

session_id = "02834806-801e-4c68-a15f-309e2caca009"

documents = search_chroma(
    "What is this document about?",
    session_id
)

for doc in documents:
    print(doc.page_content)
    print(doc.metadata)