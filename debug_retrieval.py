import sys
import chromadb
from chromadb.utils import embedding_functions

try:
    print("STEP 1: Starting script", flush=True)

    client = chromadb.PersistentClient(path="./chroma_db")
    print("STEP 2: Client created", flush=True)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-mpnet-base-v2"
    )
    print("STEP 3: Embedding function ready", flush=True)

    collection = client.get_or_create_collection(
        name="pdf_docs",
        embedding_function=embedding_fn
    )
    print("STEP 4: Collection loaded", flush=True)

    count = collection.count()
    print(f"STEP 5: Collection has {count} items", flush=True)

    results = collection.query(
        query_texts=["Who developed UPI?"],
        n_results=5
    )
    print("STEP 6: Query completed", flush=True)

    for i, chunk in enumerate(results["documents"][0]):
        print(f"--- Match {i+1} (distance: {results['distances'][0][i]:.3f}) ---", flush=True)
        print(chunk[:200], flush=True)

    print("STEP 7: All done", flush=True)

except Exception as e:
    print(f"ERROR OCCURRED: {e}", flush=True)
    import traceback
    traceback.print_exc()