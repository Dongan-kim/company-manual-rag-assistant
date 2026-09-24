import os

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


PDF_PATH = "data/company_manual.pdf"
CHROMA_PATH = "./chroma_langchain"
COLLECTION_NAME = "company_manual"


load_dotenv()


# -------------------------
# 1. PDF → LangChain Document
# -------------------------

reader = PdfReader(PDF_PATH)

documents = []

for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text()

    if text:

        document = Document(
            page_content=text,
            metadata={
                "source": "company_manual.pdf",
                "page": page_number
            }
        )

        documents.append(document)


print("PDF pages loaded:", len(documents))


# -------------------------
# 2. Chunking
# -------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)


chunks = text_splitter.split_documents(documents)


print("Chunks created:", len(chunks))


# -------------------------
# 3. Embedding model
# -------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# -------------------------
# 4. Chromaへ保存
# -------------------------

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_PATH
)


print("Documents stored successfully.")
print(
    "Documents in collection:",
    vector_store._collection.count()
)