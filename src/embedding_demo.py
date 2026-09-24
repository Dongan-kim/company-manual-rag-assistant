import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(vector_a, vector_b):
    a = np.array(vector_a)
    b = np.array(vector_b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


text_a = "新規顧客との契約方法"
text_b = "新しい会社と取引を開始するにはどうすればいいですか？"
text_c = "今日の大阪の天気はどうですか？"

embedding_a = get_embedding(text_a)
embedding_b = get_embedding(text_b)
embedding_c = get_embedding(text_c)

print("Embedding dimension:", len(embedding_a))

print("\nFirst 10 numbers:")
print(embedding_a[:10])

similarity_ab = cosine_similarity(embedding_a, embedding_b)
similarity_ac = cosine_similarity(embedding_a, embedding_c)

print("\n--- Similarity ---")
print(f"A <-> B: {similarity_ab:.4f}")
print(f"A <-> C: {similarity_ac:.4f}")