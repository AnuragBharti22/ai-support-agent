import chromadb

# Step 1: Read the document
with open("docs.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Step 2: Split into chunks (by double newline = paragraph breaks)
chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

print(f"Split document into {len(chunks)} chunks.")

# Step 3: Set up ChromaDB (a local vector database)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="docs")

# Step 4: Add chunks to the collection (Chroma auto-generates embeddings)
ids = [f"chunk_{i}" for i in range(len(chunks))]
collection.add(documents=chunks, ids=ids)

print(f"Stored {len(chunks)} chunks in the vector database.")