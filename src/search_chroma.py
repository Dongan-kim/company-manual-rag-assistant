import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


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


# 既存のChroma DBを開く
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)


# 既存のCollectionを取得
collection = chroma_client.get_collection(
    name="company_documents"
)


print("Documents in database:", collection.count())


question = input("\n質問を入力してください: ")

# 質問だけEmbeddingする
question_embedding = get_embedding(question)


# ChromaでVector Search
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)


print("\n--- Top 3 Results ---")

for document, distance in zip(
    results["documents"][0],
    results["distances"][0]
):
    print(f"Distance: {distance:.4f}")
    print(document)
    print()