import os
import re
import sys
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

DOCS_FOLDER = "documents"
COLLECTION_NAME = "knowledge_base"

if not os.path.isdir(DOCS_FOLDER):
    sys.exit("The 'documents' folder doesn't exist. Create it and put your PDF inside as documents\\upi.pdf")

files = sorted(os.listdir(DOCS_FOLDER))
if not files:
    sys.exit("The 'documents' folder is empty. Move your PDF into it (for example documents\\upi.pdf).")
print("Found files:", files)


def load_file_text(path):
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def chunk_text(text, chunk_size=1000, overlap=150):
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


client = chromadb.PersistentClient(path="./chroma_db")

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)

# Start fresh each time so re-running never creates duplicates
try:
    client.delete_collection(name=COLLECTION_NAME)
except Exception:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

total_chunks = 0

for filename in files:
    path = os.path.join(DOCS_FOLDER, filename)
    text = load_file_text(path)
    if text is None:
        print(f"Skipping {filename} (unsupported file type)")
        continue

    chunks = chunk_text(text)
    ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk": i} for i in range(len(chunks))]

    # Add in batches of 100
    for i in range(0, len(chunks), 100):
        collection.add(
            documents=chunks[i:i + 100],
            ids=ids[i:i + 100],
            metadatas=metadatas[i:i + 100],
        )

    print(f"{filename}: {len(text)} characters -> {len(chunks)} chunks")
    total_chunks += len(chunks)

print(f"\nDone. {total_chunks} chunks stored in '{COLLECTION_NAME}'.")
print("Verifying count:", collection.count())