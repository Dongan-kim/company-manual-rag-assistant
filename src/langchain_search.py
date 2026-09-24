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
# 2. 保存済みChromaを開く
# -------------------------

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)


# -------------------------
# 3. Retrieverを作る
# -------------------------

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# -------------------------
# 4. 質問
# -------------------------

question = input("Question: ")


# -------------------------
# 5. Retrieval
# -------------------------

documents = retriever.invoke(question)


# -------------------------
# 6. 結果表示
# -------------------------

print("\n--- Retrieved Documents ---")

for i, document in enumerate(
    documents,
    start=1
):

    print(f"\n--- Result {i} ---")

    print(
        "Source:",
        document.metadata["source"]
    )

    print(
        "Page:",
        document.metadata["page"]
    )

    print()

    print(document.page_content)