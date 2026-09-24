from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_PATH = "data/company_manual.pdf"


# -------------------------
# 1. PDFをページごとに読む
# -------------------------

reader = PdfReader(PDF_PATH)

documents = []

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text()

    if text:
        documents.append({
            "text": text,
            "page": page_number
        })


# -------------------------
# 2. LangChain Text Splitter
# -------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)


# -------------------------
# 3. ページごとにChunking
# -------------------------

all_chunks = []

for document in documents:

    chunks = text_splitter.split_text(
        document["text"]
    )

    for chunk in chunks:
        all_chunks.append({
            "text": chunk,
            "page": document["page"]
        })


# -------------------------
# 4. 結果確認
# -------------------------

print("Number of chunks:", len(all_chunks))


for i, chunk in enumerate(all_chunks[:5], start=1):

    print(f"\n--- Chunk {i} ---")
    print("Page:", chunk["page"])
    print("Length:", len(chunk["text"]))
    print(chunk["text"])