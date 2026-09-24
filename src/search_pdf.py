import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "company_manual"


load_dotenv()

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def get_embedding(text):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


# Chroma DBを開く
chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)


print("Chunks in database:", collection.count())


# ユーザーから質問
question = input("\nQuestion: ")


# 質問をEmbedding
question_embedding = get_embedding(question)


# Chromaで関連Chunkを検索
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)


print("\n--- Top 3 Relevant Chunks ---")


for i, (document, distance) in enumerate(
    zip(
        results["documents"][0],
        results["distances"][0]
    ),
    start=1
):
    print(f"\n--- Result {i} ---")
    print(f"Distance: {distance:.4f}")
    print(document)