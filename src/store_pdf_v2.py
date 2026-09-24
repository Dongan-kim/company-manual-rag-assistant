import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


PDF_PATH = "data/company_manual.pdf"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "company_manual_v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


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


def split_text(text, chunk_size, overlap):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        chunk = text[start:end]

        # できるだけ単語の途中で切らない
        if end < len(text):
            last_space = chunk.rfind(" ")

            if last_space > chunk_size * 0.7:
                end = start + last_space
                chunk = text[start:end]

        chunks.append(chunk.strip())

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# -------------------------
# 1. PDFを読む
# -------------------------

reader = PdfReader(PDF_PATH)

all_chunks = []
all_metadata = []


# -------------------------
# 2. ページごとにChunking
# -------------------------

for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text()

    if not text:
        continue

    page_chunks = split_text(
        text,
        CHUNK_SIZE,
        CHUNK_OVERLAP
    )

    for chunk_index, chunk in enumerate(page_chunks):

        all_chunks.append(chunk)

        all_metadata.append({
            "source": "company_manual.pdf",
            "page": page_number,
            "chunk": chunk_index
        })


print("Number of chunks:", len(all_chunks))


# -------------------------
# 3. Embedding
# -------------------------

print("Creating embeddings...")

embeddings = []

for i, chunk in enumerate(all_chunks):

    embedding = get_embedding(chunk)

    embeddings.append(embedding)

    print(
        f"Embedded chunk {i + 1}/{len(all_chunks)}"
    )


# -------------------------
# 4. Chromaへ保存
# -------------------------

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)


ids = [
    f"chunk_{i}"
    for i in range(len(all_chunks))
]


collection.upsert(
    ids=ids,
    documents=all_chunks,
    embeddings=embeddings,
    metadatas=all_metadata
)


print("\nPDF stored successfully.")
print("Documents in collection:", collection.count())