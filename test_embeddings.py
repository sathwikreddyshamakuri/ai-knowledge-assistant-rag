from openai import OpenAI

client = OpenAI()

text = "Amazon S3 is an object storage service."

response = client.embeddings.create( #generates vector
    model="text-embedding-3-small",
    input=text
)

embedding = response.data[0].embedding

print("Number of dimensions:", len(embedding))
print("First 10 values:", embedding[:10])