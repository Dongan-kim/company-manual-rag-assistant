from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

THRESHOLD = 1.3

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

    # 答えがマニュアルに存在する
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

    # 答えがマニュアルに存在しない
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

    # 答えなし
    {
        "question": "Do employees get a Christmas bonus?",
        "expected_page": None
    },
    {
        "question": "Does the company pay for dental treatment?",
        "expected_page": None
    }
]


print("\n--- Retrieval Evaluation ---")


correct = 0

for test in test_questions:

    question = test["question"]
    expected_page = test["expected_page"]

    results = vector_store.similarity_search_with_score(
        question,
        k=3
    )

    best_document, best_score = results[0]

    retrieved_pages = [
        document.metadata["page"]
        for document, score in results
    ]

    predicted_found = best_score <= THRESHOLD
    actual_found = expected_page is not None

    print("\nQuestion:")
    print(question)

    print("Expected page:", expected_page)
    print("Top page:", best_document.metadata["page"])
    print("Best score:", round(best_score, 4))

    if actual_found:

        # 答えが存在する質問
        if predicted_found and expected_page in retrieved_pages:
            result = "PASS"
        else:
            result = "FAIL"

    else:

        # 答えが存在しない質問
        if not predicted_found:
            result = "PASS"
        else:
            result = "FAIL"

    if result == "PASS":
        correct += 1

    print("Predicted:", "FOUND" if predicted_found else "NOT_FOUND")
    print("Result:", result)


accuracy = correct / len(test_questions)

print("\n========================")
print("Evaluation Summary")
print("========================")
print("Threshold:", THRESHOLD)
print("Correct:", correct, "/", len(test_questions))
print("Accuracy:", f"{accuracy:.1%}")