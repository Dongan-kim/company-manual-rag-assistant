from pypdf import PdfReader


PDF_PATH = "data/company_manual.pdf"


reader = PdfReader(PDF_PATH)

print("Number of pages:", len(reader.pages))


full_text = ""

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text()

    if text:
        full_text += text + "\n"

    print(f"Page {page_number}: {len(text or '')} characters")


print("\n--- First 1000 characters ---")
print(full_text[:1000])


# -------------------------
# Chunking
# -------------------------

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def split_text(text, chunk_size, overlap):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


chunks = split_text(
    full_text,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


print("\n--- Chunking Result ---")
print("Total characters:", len(full_text))
print("Number of chunks:", len(chunks))


for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)
    print(f"\nLength: {len(chunk)}")