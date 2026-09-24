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


documents = [
    "新規顧客との契約を開始するには、営業部長の承認が必要です。",
    "有給休暇を取得する場合は、3日前までに上司へ申請してください。",
    "パスワードを忘れた場合は、IT部門にパスワードリセットを依頼してください。",
    "交通費は毎月末までに経費精算システムから申請してください。",
    "在宅勤務を希望する場合は、事前にマネージャーの許可を取得してください。"
]


# Chroma DBをローカルに作成
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)


# Collectionを作成
collection = chroma_client.get_or_create_collection(
    name="company_documents"
)


print("Creating embeddings...")


embeddings = []

for document in documents:
    embedding = get_embedding(document)
    embeddings.append(embedding)


collection.upsert(
    ids=[
        "doc1",
        "doc2",
        "doc3",
        "doc4",
        "doc5"
    ],
    documents=documents,
    embeddings=embeddings
)


print("Documents stored successfully.")
print("Number of documents:", collection.count())