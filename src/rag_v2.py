import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "company_manual_v2"

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
# 2. 質問を入力
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

documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]


# -------------------------
# 5. LLM用Contextを作る
# -------------------------

context_parts = []

for document, metadata in zip(
    documents,
    metadatas
):
    context_part = f"""
Source: {metadata["source"]}
Page: {metadata["page"]}

{document}
"""

    context_parts.append(context_part)


context = "\n\n---\n\n".join(context_parts)


# -------------------------
# 6. Prompt
# -------------------------

prompt = f"""
You are an assistant answering questions about a company manual.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, respond with exactly:
NOT_FOUND

Do not invent company policies.

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
# 8. 回答とSources
# -------------------------

answer = response.output_text.strip()

print("\n--- Answer ---")

if answer == "NOT_FOUND":
    print(
        "I could not find that information "
        "in the company manual."
    )

else:
    print(answer)

    print("\n--- Sources ---")

    seen_sources = set()

    for metadata in metadatas:

        source = (
            metadata["source"],
            metadata["page"]
        )

        if source not in seen_sources:

            print(
                f'{metadata["source"]} '
                f'- Page {metadata["page"]}'
            )

            seen_sources.add(source)

# -------------------------
# 10. Debug情報
# -------------------------

print("\n--- Retrieved Chunks ---")

for i, (
    document,
    metadata,
    distance
) in enumerate(
    zip(
        documents,
        metadatas,
        distances
    ),
    start=1
):

    print(f"\n[Result {i}]")
    print(f"Page: {metadata['page']}")
    print(f"Distance: {distance:.4f}")
    print(document)