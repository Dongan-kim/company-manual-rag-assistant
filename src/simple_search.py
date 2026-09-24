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


documents = [
    "新規顧客との契約を開始するには、営業部長の承認が必要です。",
    "有給休暇を取得する場合は、3日前までに上司へ申請してください。",
    "パスワードを忘れた場合は、IT部門にパスワードリセットを依頼してください。",
    "交通費は毎月末までに経費精算システムから申請してください。",
    "在宅勤務を希望する場合は、事前にマネージャーの許可を取得してください。"
]


print("Creating document embeddings...")

document_embeddings = []

for document in documents:
    embedding = get_embedding(document)
    document_embeddings.append(embedding)


question = input("\n質問を入力してください: ")

question_embedding = get_embedding(question)


results = []

for document, document_embedding in zip(
    documents,
    document_embeddings
):
    similarity = cosine_similarity(
        question_embedding,
        document_embedding
    )

    results.append((similarity, document))


results.sort(reverse=True)


print("\n--- Search Results ---")

for similarity, document in results:
    print(f"{similarity:.4f} | {document}")


best_similarity, best_document = results[0]

print("\n--- Best Match ---")
print(f"Similarity: {best_similarity:.4f}")
print(best_document)