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


# -------------------------
# 1. Chromaを開く
# -------------------------

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)


# -------------------------
# 2. ユーザーの質問
# -------------------------

question = input("Question: ")


# -------------------------
# 3. 質問をEmbedding
# -------------------------

question_embedding = get_embedding(question)


# -------------------------
# 4. 関連Chunkを検索
# -------------------------

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

retrieved_chunks = results["documents"][0]


# -------------------------
# 5. ChunkをContextにまとめる
# -------------------------

context = "\n\n---\n\n".join(retrieved_chunks)


# -------------------------
# 6. LLMに渡すPrompt
# -------------------------

prompt = f"""
You are an assistant answering questions about a company manual.

Answer the question using ONLY the information in the context below.

If the answer cannot be found in the context, say:
"I could not find that information in the company manual."

Context:
{context}

Question:
{question}
"""


# -------------------------
# 7. LLMで回答生成
# -------------------------

response = openai_client.responses.create(
    model="gpt-5.6-luna",
    input=prompt
)


# -------------------------
# 8. 結果表示
# -------------------------

print("\n--- Retrieved Context ---")

for i, chunk in enumerate(retrieved_chunks, start=1):
    print(f"\n[Chunk {i}]")
    print(chunk)


print("\n--- Answer ---")
print(response.output_text)