import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
for col in client.list_collections():
    name = col if isinstance(col, str) else col.name
    print(name, "->", client.get_collection(name).count(), "items")