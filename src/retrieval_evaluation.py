from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


CHROMA_PATH = "./chroma_langchain"
COLLECTION_NAME = "company_manual"


load_dotenv()


# -------------------------
# 1. Embedding model
# -------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# -------------------------
# 2. Chromaを開く
# -------------------------

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)


# -------------------------
# 3. 質問
# -------------------------

question = input("Question: ")


# -------------------------
# 4. Top 5 + Scoreを取得
# -------------------------

results = vector_store.similarity_search_with_score(
    question,
    k=5
)


# -------------------------
# 5. 結果表示
# -------------------------

print("\n--- Retrieval Results ---")

for rank, (document, score) in enumerate(
    results,
    start=1
):

    print(f"\n--- Rank {rank} ---")

    print("Score:", round(score, 4))

    print(
        "Source:",
        document.metadata["source"]
    )

    print(
        "Page:",
        document.metadata["page"]
    )

    print(
        "Preview:",
        document.page_content[:200]
    )