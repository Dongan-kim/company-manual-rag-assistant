import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


PDF_PATH = "data/company_manual.pdf"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "company_manual"

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
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# 1. PDFからテキストを抽出
reader = PdfReader(PDF_PATH)

full_text = ""

for page in reader.pages:
    text = page.extract_text()

    if text:
        full_text += text + "\n"


# 2. Chunking
chunks = split_text(
    full_text,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

print("Number of chunks:", len(chunks))


# 3. 各ChunkをEmbedding
print("Creating embeddings...")

embeddings = []

for i, chunk in enumerate(chunks):
    embedding = get_embedding(chunk)
    embeddings.append(embedding)

    print(f"Embedded chunk {i + 1}/{len(chunks)}")


# 4. Chromaを開く
chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


# 5. PDF用Collectionを作る
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)


# 6. IDを作る
ids = [
    f"chunk_{i}"
    for i in range(len(chunks))
]


# 7. Chromaへ保存
collection.upsert(
    ids=ids,
    documents=chunks,
    embeddings=embeddings
)


print("\nPDF stored successfully.")
print("Documents in collection:", collection.count())