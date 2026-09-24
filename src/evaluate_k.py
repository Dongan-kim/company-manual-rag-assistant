from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


CHROMA_PATH = "./chroma_langchain"
COLLECTION_NAME = "company_manual"

load_dotenv()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)


# -------------------------
# Evaluation Dataset
# 答えが存在する質問のみ
# -------------------------

test_questions = [
    {
        "question": "What should I do if I forget my password?",
        "expected_page": 3
    },
    {
        "question": "How many days of annual leave do employees receive?",
        "expected_page": 2
    },
    {
        "question": "Can employees work from home?",
        "expected_page": 4
    },
    {
        "question": "How do I submit an expense claim?",
        "expected_page": 6
    },
    {
        "question": "What are the normal working hours?",
        "expected_page": 1
    },
    {
        "question": "What should I do if I receive a suspicious email?",
        "expected_page": 7
    },
    {
        "question": "How much vacation do I get each year?",
        "expected_page": 2
    },
    {
        "question": "I can't get into my account. How can I regain access?",
        "expected_page": 3
    },
    {
        "question": "Am I allowed to do my job outside the office?",
        "expected_page": 4
    },
    {
        "question": "Will the company pay me back for a business lunch?",
        "expected_page": 6
    }
]


# -------------------------
# kを比較
# -------------------------

k_values = [1, 3, 5]

print("\n--- Hit@k Evaluation ---")


for k in k_values:

    hits = 0

    print(f"\n===== k = {k} =====")

    for test in test_questions:

        results = vector_store.similarity_search_with_score(
            test["question"],
            k=k
        )

        retrieved_pages = [
            document.metadata["page"]
            for document, score in results
        ]

        expected_page = test["expected_page"]

        hit = expected_page in retrieved_pages

        if hit:
            hits += 1

        print(
            f"{'PASS' if hit else 'FAIL'} | "
            f"Expected: {expected_page} | "
            f"Retrieved: {retrieved_pages} | "
            f"{test['question']}"
        )

    hit_rate = hits / len(test_questions)

    print(
        f"\nHit@{k}: "
        f"{hits}/{len(test_questions)} "
        f"({hit_rate:.1%})"
    )