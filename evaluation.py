import chromadb
from openai import OpenAI


evaluation_questions = [
    {
        "question": "Which AWS service provides serverless computing?",
        "expected_document": "AWS_and_Its_Services.txt",
        "relevant_chunks": [3, 4]
    },
    {
        "question": "Which AWS service provides object storage?",
        "expected_document": "AWS_and_Its_Services.txt",
        "relevant_chunks": [5]
    },
    {
        "question": "Which AWS service is a fully managed NoSQL database?",
        "expected_document": "sample-txt-file-for-openai-project.txt",
        "relevant_chunks": [0]
    },
    {
        "question": "Which AWS service simplifies running relational databases such as MySQL?",
        "expected_document": "AWS_and_Its_Services.txt",
        "relevant_chunks": [6]
    },
    {
        "question": "Which AWS services provide in-memory caching, graph databases, and data warehousing?",
        "expected_document": "AWS_and_Its_Services.txt",
        "relevant_chunks": [7]
    },
    {
        "question": "Which AWS service provides dedicated network connections between on-premises data centers and AWS?",
        "expected_document": "AWS_and_Its_Services.txt",
        "relevant_chunks": [8]
    },
    {
        "question": "Which AWS service provides access to foundation models for generative AI?",
        "expected_document": "AWS_and_Its_Services.txt",
        "relevant_chunks": [9]
    },
    {
        "question": "Which AWS service controls who can access AWS resources?",
        "expected_document": "AWS_and_Its_Services.txt",
        "relevant_chunks": [10]
    },
    {
        "question": "Which AWS services are used together for continuous integration and delivery?",
        "expected_document": "AWS_and_Its_Services.txt",
        "relevant_chunks": [11]
    },
    {
        "question": "Which AWS service enables interactive SQL queries directly against data stored in S3?",
        "expected_document": "AWS_and_Its_Services.txt",
        "relevant_chunks": [12]
    }
]


chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="documents"
)

client = OpenAI()


def get_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


def evaluate_retrieval():
    
    top_k_values = [1, 3, 5]

    total_questions = len(evaluation_questions)

    for top_k in top_k_values:

        passed_questions = 0

        print("\n" + "=" * 50)
        print(f"Evaluating Top-K = {top_k}")
        print("=" * 50)

        for item in evaluation_questions:

            question = item["question"]
            expected_document = item["expected_document"]
            relevant_chunks = item["relevant_chunks"]

            question_embedding = get_embedding(question)

            results = collection.query(
                query_embeddings=[question_embedding],
                n_results=top_k
            )

            retrieved_metadata = results["metadatas"][0]

            retrieved_chunks = [
                (
                    metadata["document_name"],
                    metadata["chunk_number"]
                )
                for metadata in retrieved_metadata
            ]

            passed = any(
                document_name == expected_document
                and chunk_number in relevant_chunks
                for document_name, chunk_number in retrieved_chunks
            )

            if passed:
                passed_questions += 1

            print("\nQuestion:", question)

            print("Retrieved:")

            for document_name, chunk_number in retrieved_chunks:

                print(
                    "-",
                    document_name,
                    "| Chunk:", chunk_number
                )

            print(
                "Result:",
                "PASS" if passed else "FAIL"
            )

        recall_at_k = passed_questions / total_questions

        print(
            f"\nChunk Recall@{top_k}: "
            f"{recall_at_k:.2%}"
        )


if __name__ == "__main__":
    evaluate_retrieval()