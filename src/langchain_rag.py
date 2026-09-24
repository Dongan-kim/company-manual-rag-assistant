from dotenv import load_dotenv

from langchain_openai import (
    OpenAIEmbeddings,
    ChatOpenAI
)

from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate


CHROMA_PATH = "./chroma_langchain"
COLLECTION_NAME = "company_manual"

THRESHOLD = 1.30


load_dotenv()


# -------------------------
# 1. Embedding model
# -------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# -------------------------
# 2. Chroma
# -------------------------

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)


# -------------------------
# 3. Retriever
# -------------------------

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# -------------------------
# 4. Chat Model
# -------------------------

llm = ChatOpenAI(
    model="gpt-5.6-luna"
)


# -------------------------
# 5. Prompt Template
# -------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are an assistant answering questions about
a company manual.

Answer the user's question using ONLY the
provided context.

If the answer cannot be found in the context,
respond with exactly:

NOT_FOUND

Do not invent company policies.

Context:
{context}

Question:
{question}
"""
)


# -------------------------
# 6. User Question
# -------------------------

question = input("Question: ")


# -------------------------
# 7. Retrieval
# -------------------------

results = vector_store.similarity_search_with_score(
    question,
    k=3
)

best_score = results[0][1]

if best_score > THRESHOLD:
    print("\n--- Answer ---")
    print("I could not find that information in the company manual.")
    exit()

documents = [
    document
    for document, score in results
]


# -------------------------
# 8. Contextを作る
# -------------------------

context_parts = []

for document in documents:

    context_part = f"""
Source: {document.metadata["source"]}
Page: {document.metadata["page"]}

{document.page_content}
"""

    context_parts.append(context_part)


context = "\n\n---\n\n".join(
    context_parts
)


# -------------------------
# 9. Promptを作る
# -------------------------

messages = prompt.invoke({
    "context": context,
    "question": question
})


# -------------------------
# 10. LLMへ送る
# -------------------------

response = llm.invoke(messages)

answer = response.content.strip()


# -------------------------
# 11. Answer
# -------------------------

print("\n--- Answer ---")

if answer == "NOT_FOUND":

    print(
        "I could not find that information "
        "in the company manual."
    )

else:

    print(answer)


    # -------------------------
    # 12. Sources
    # -------------------------

    print("\n--- Sources ---")

    seen_sources = set()

    for document in documents:

        source = (
            document.metadata["source"],
            document.metadata["page"]
        )

        if source not in seen_sources:

            print(
                f'{document.metadata["source"]} '
                f'- Page {document.metadata["page"]}'
            )

            seen_sources.add(source)