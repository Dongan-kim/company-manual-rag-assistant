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
        "question": "How much is the annual employee bonus?",
        "expected_page": None
    },
    {
        "question": "Does the company provide health insurance?",
        "expected_page": None
    },
    {
        "question": "Where can employees park their cars?",
        "expected_page": None
    },
    {
        "question": "Can employees buy company stock?",
        "expected_page": None
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
    },
    {
        "question": "Do employees get a Christmas bonus?",
        "expected_page": None
    },
    {
        "question": "Does the company pay for dental treatment?",
        "expected_page": None
    }
]


# -------------------------
# Retrievalは一度だけ実行
# -------------------------

evaluation_data = []

for test in test_questions:

    results = vector_store.similarity_search_with_score(
        test["question"],
        k=3
    )

    best_document, best_score = results[0]

    retrieved_pages = [
        document.metadata["page"]
        for document, score in results
    ]

    evaluation_data.append({
        "question": test["question"],
        "expected_page": test["expected_page"],
        "retrieved_pages": retrieved_pages,
        "best_score": best_score
    })


# -------------------------
# Threshold比較
# -------------------------

thresholds = [
    1.10,
    1.15,
    1.20,
    1.25,
    1.30,
    1.35,
    1.40
]


print("\n--- Threshold Evaluation ---")
print()

for threshold in thresholds:

    correct = 0
    false_positive = 0
    false_negative = 0

    for item in evaluation_data:

        expected_page = item["expected_page"]
        best_score = item["best_score"]
        retrieved_pages = item["retrieved_pages"]

        actual_found = expected_page is not None
        predicted_found = best_score <= threshold

        if actual_found:

            if predicted_found and expected_page in retrieved_pages:
                correct += 1
            else:
                false_negative += 1

        else:

            if not predicted_found:
                correct += 1
            else:
                false_positive += 1

    accuracy = correct / len(evaluation_data)

    print(
        f"Threshold {threshold:.2f} | "
        f"Accuracy {accuracy:.1%} | "
        f"FP {false_positive} | "
        f"FN {false_negative}"
    )