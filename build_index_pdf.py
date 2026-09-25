from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions
import re

def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text


def chunk_text(text, chunk_size=1000, overlap=150):
    text = re.sub(r'\s+', ' ', text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


pdf_path = "knowledge.pdf"
raw_text = load_pdf_text(pdf_path)
print(f"Extracted {len(raw_text)} characters from PDF.")

chunks = chunk_text(raw_text)
print(f"Split into {len(chunks)} chunks.")

client = chromadb.PersistentClient(path="./chroma_db")

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)

collection = client.get_or_create_collection(
    name="pdf_docs",
    embedding_function=embedding_fn
)

ids = [f"pdf_chunk_{i}" for i in range(len(chunks))]
collection.add(documents=chunks, ids=ids)

print(f"Stored {len(chunks)} chunks in the 'pdf_docs' collection.")
print("Verifying count immediately after insert:", collection.count())