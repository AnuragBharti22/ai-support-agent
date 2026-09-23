import chromadb

# Connect to the same database we built earlier
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="docs")

# The question we want to test
query = "How do I keep my API key safe?"

# Search for the most relevant chunks
results = collection.query(
    query_texts=[query],
    n_results=3
)

print(f"Query: {query}\n")
print("Top matching chunks:\n")

for i, chunk in enumerate(results["documents"][0]):
    print(f"--- Match {i+1} ---")
    print(chunk)
    print()